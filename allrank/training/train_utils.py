import os
from functools import partial

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
import allrank.models.metrics as metrics_module
from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.model_utils import get_num_params, log_num_params, get_model_parameters
from allrank.training.early_stop import EarlyStop
from allrank.utils.ltr_logging import get_logger
from allrank.utils.tensorboard_utils import TensorboardSummaryWriter
# from allrank.min_norm_solvers import MinNormSolver, gradient_normalizers
from itertools import zip_longest
from allrank.methods.weight_methods import WeightMethods

logger = get_logger()

def loss_batch(model, loss_func, xb, yb, indices):
    """Calculate loss for a single batch."""
    xb = xb.detach().requires_grad_(True)
    yb = yb.detach().requires_grad_(True)
    mask = (yb == PADDED_Y_VALUE)
    return loss_func(model(xb, mask, indices), yb)

def metric_on_batch(metric, model, xb, yb, indices):
    """Calculate metric for a single batch."""
    mask = (yb == PADDED_Y_VALUE)
    return metric(model.score(xb, mask, indices), yb)

def metric_on_epoch(metric, model, dl_single, dev):
    metric_values = torch.mean(
        torch.cat(
            [metric_on_batch(metric, model, xb.to(device=dev), yb_single.to(device=dev), indices.to(device=dev))
             for xb, yb_single, indices in dl_single]
        ), dim=0
    ).cpu().numpy()
    return metric_values

def compute_metrics(metrics, model, dl_single, dev):
    metric_values_dict = {}
    for metric_name, ats in metrics.items():
        metric_func = getattr(metrics_module, metric_name)
        metric_func_with_ats = partial(metric_func, ats=ats)
        metrics_values = metric_on_epoch(metric_func_with_ats, model, dl_single, dev)
        metrics_names = ["{metric_name}_{at}".format(metric_name=metric_name, at=at) for at in ats]
        metric_values_dict.update(dict(zip(metrics_names, metrics_values)))
    return metric_values_dict

def epoch_summary(epoch, train_loss, val_loss, train_metrics, val_metrics):
    summary = "Epoch : {epoch} Train loss: {train_loss} Val loss: {val_loss}".format(
        epoch=epoch, train_loss=train_loss, val_loss=val_loss)
    for metric_name, metric_value in train_metrics.items():
        summary += " Train {metric_name} {metric_value}".format(
            metric_name=metric_name, metric_value=metric_value)
    for metric_name, metric_value in val_metrics.items():
        summary += " Val {metric_name} {metric_value}".format(
            metric_name=metric_name, metric_value=metric_value)
    return summary

def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    
def log_metrics(epoch, phase, metrics_dict, loss_values, batch_sizes, task_idx, results_filename):
        """Helper function to log metrics for each task to a file
        
        Args:
            epoch (int): Current epoch
            phase (str): Training phase (Train/Valid)
            metrics_dict (dict): Dictionary of metrics
            loss_values (list): List of loss values
            batch_sizes (list): List of batch sizes
            task_idx (int): Task index
        """
        with open(results_filename, "a") as file:
            loss_result = np.sum([a*b for a, b in zip(loss_values, batch_sizes)]) / np.sum(batch_sizes)
            file.write(f"epoch:{epoch}\ttask:{task_idx}\t{phase} Loss:{loss_result}\t")
            if metrics_dict:
                file.write(f"{phase} Metrics:{metrics_dict}\t")
            file.write("\n")

def fit(epochs, moo_method, main_task_index, task_indices, label_indices,
        results_filename, model, loss_func,
        task_weights, optimizer, scheduler, 
        train_dataloader, val_dataloader, config, 
        gradient_clipping_norm, early_stopping_patience,
        device, output_dir, tensorboard_output_path, 
        epsilon=None, compute_delta_m=False, stl_delta_m=None):
    """Main training loop for multi-task learning with ranking tasks.

    Args:
        moo_method (str): Multi-objective optimization method to use
        main_task_index (int): Index of the main task
        task_indices (list): Indices of tasks to train
        label_indices (list): Indices representing label columns
        results_filename (str): Path to save results
        task_weights (list): Initial weights for different tasks
        epsilon (list, optional): Epsilon values for EC method. Defaults to None.
        compute_delta_m (bool, optional): Whether to compute delta_m metric. Defaults to False.
        stl_delta_m (dict, optional): Single-task learning delta_m values. Required if compute_delta_m is True.
        ... (other args)
    """
    tensorboard_writer = TensorboardSummaryWriter(tensorboard_output_path)
    early_stop = EarlyStop(early_stopping_patience)
    weight_method = WeightMethods(moo_method, 
                                n_tasks=len(task_indices), 
                                device=device, 
                                task_weights=task_weights,
                                epsilon=epsilon)
            
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}, Current learning rate: {get_current_lr(optimizer)}")
        model.train()
        
        # Initialize lists to store losses for each task
        train_loss_values = {task_idx: [] for task_idx in task_indices}
        train_nums = []  # Store batch sizes for weighted averaging

        # Process batches
        for batch_id, batch in enumerate(train_dataloader):
            xb, yb, indices = batch
            # Create a mask of indices to keep for xb, others are candidate labels
            all_indices = torch.arange(xb.shape[-1])
            keep_indices = all_indices[~torch.isin(all_indices, torch.tensor(label_indices))]
            # Select features to keep by indexing xb with keep_indices
            # This removes the label candidate columns from the input tensor
            modified_xb = xb[:, :, keep_indices]
            
            losses = []
            for task_idx in task_indices:
                task_yb = yb if task_idx == 0 else xb[:, :, task_idx]
                task_yb[yb == -1] = -1
                loss = loss_batch(model, loss_func, modified_xb.to(device), task_yb.to(device), indices.to(device))
                losses.append(loss)
                train_loss_values[task_idx].append(loss.item())
            train_nums.append(len(xb))
            
            if optimizer:
                optimizer.zero_grad()
                # Apply multi-objective optimization strategy
                loss_weighted_sum, _ = weight_method.backward(
                    losses=torch.stack(losses),
                    shared_parameters=get_model_parameters(model, 'shared_parameters'),
                    task_specific_parameters=get_model_parameters(model, 'task_specific_parameters'),
                    last_shared_parameters=get_model_parameters(model, 'last_shared_parameters'),
                    task_weights=task_weights
                )
                # Apply gradient clipping if specified
                if gradient_clipping_norm:
                    clip_grad_norm_(model.parameters(), gradient_clipping_norm)
                optimizer.step()

        # Log training metrics
        for task_idx in task_indices:
            temp_dl = []
            for xb, yb, indices in train_dataloader:
                all_indices = torch.arange(xb.shape[-1])
                keep_indices = all_indices[~torch.isin(all_indices, torch.tensor(label_indices))]
                modified_xb = xb[:, :, keep_indices]
                
                task_yb = yb if task_idx == 0 else xb[:, :, task_idx]
                task_yb[yb == -1] = -1
                temp_dl.append((modified_xb, task_yb, indices))
                
            train_metrics = compute_metrics(config.metrics, model, temp_dl, device)
            log_metrics(epoch, "Train", train_metrics, 
                       train_loss_values[task_idx], train_nums, task_idx, results_filename)

        # Validation loop
        model.eval()
        with torch.no_grad():
            valid_loss_values = {task_idx: [] for task_idx in task_indices}
            valid_nums = []
            
            for batch in val_dataloader:
                xb, yb, indices = batch
                all_indices = torch.arange(xb.shape[-1])
                keep_indices = all_indices[~torch.isin(all_indices, torch.tensor(label_indices))]
                modified_xb = xb[:, :, keep_indices]
                
                for task_idx in task_indices:
                    task_yb = yb if task_idx == 0 else xb[:, :, task_idx]
                    task_yb[yb == -1] = -1
                    loss = loss_batch(model, loss_func, modified_xb.to(device), task_yb.to(device), indices.to(device))
                    valid_loss_values[task_idx].append(loss.item())
                valid_nums.append(len(xb))

            # Log validation metrics
            current_result = {}
            for task_idx in task_indices:
                temp_dl = []
                for xb, yb, indices in val_dataloader:
                    all_indices = torch.arange(xb.shape[-1])
                    keep_indices = all_indices[~torch.isin(all_indices, torch.tensor(label_indices))]
                    modified_xb = xb[:, :, keep_indices]
                    
                    task_yb = yb if task_idx == 0 else xb[:, :, task_idx]
                    task_yb[yb == -1] = -1
                    temp_dl.append((modified_xb, task_yb, indices))
                    
                valid_metrics = compute_metrics(config.metrics, model, temp_dl, device)
                current_result[task_idx] = valid_metrics
                log_metrics(epoch, "Valid", valid_metrics, 
                          valid_loss_values[task_idx], valid_nums, task_idx, results_filename)

            # Compute delta m if requested
            if compute_delta_m:
                metric_name = 'get_deltam'
                metric_func = getattr(metrics_module, metric_name)
                delta_m = metric_func(current_result, stl_delta_m)
        # Handle scheduling and early stopping
        current_val_metric = current_result[task_indices[0]].get(config.val_metric)
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_metrics[config.val_metric])
            else:
                scheduler.step()

        early_stop.step(current_val_metric, epoch)
        if early_stop.stop_training(epoch):
            logger.info(
                "early stopping at epoch {} since {} didn't improve from epoch no {}. Best value {}, current value {}".format(
                    epoch, config.val_metric, early_stop.best_epoch, early_stop.best_value, current_val_metric
                ))
            break

    torch.save(model.state_dict(), os.path.join(output_dir, "model.pkl"))
    tensorboard_writer.close_all_writers()

    return {
        "epochs": epoch,
        "train_metrics": train_metrics,
        "val_metrics": valid_metrics,
        "num_params": get_num_params(model)
    } 