# Standard library imports
import os
import sys
import time
import random
from argparse import ArgumentParser, Namespace
from functools import partial
from urllib.parse import urlparse
from pprint import pformat
import itertools

# Third-party imports
import numpy as np
import torch
from torch import optim
from attr import asdict

# Local imports
import allrank.models.losses as losses
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, create_data_loaders
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device, CustomDataParallel
from allrank.training.train_utils import fit
from allrank.utils.command_executor import execute_command
from allrank.utils.experiments import dump_experiment_result, assert_expected_metrics
from allrank.utils.file_utils import create_output_dirs, PathsContainer, copy_local_to_gs
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import dummy_context_mgr


def parse_args() -> Namespace:
    """Parse command line arguments for the allRank training process."""
    parser = ArgumentParser("allRank")
    parser.add_argument("--output-dir", help="Base output path for all experiments", required=True)
    parser.add_argument("--task-indices", type=str, help="Comma-separated indices for tasks")
    parser.add_argument("--task-weights", type=str, help="Comma-separated task weights")
    parser.add_argument("--run-id", help="Unique identifier for this run", required=False)
    parser.add_argument("--moo-method", type=str, help="Multi-objective optimization method")
    parser.add_argument("--config-file-path", required=True, type=str, help="Path to JSON config file")
    parser.add_argument("--dataset-name", type=str, help="Name of dataset to use")
    parser.add_argument("--reduction-method", type=str, help="Loss reduction method")

    return parser.parse_args()

def get_weight_combinations(moo_method: str, num_tasks: int, task_indices: list) -> tuple:
    """Determine weight combinations or epsilon values based on MOO method and number of tasks."""
    if moo_method == "stl":
        # Single task learning - identity matrix for num_tasks
        return np.identity(num_tasks).tolist(), None
    
    elif moo_method == "ec":
        # Epsilon constraint method
        if num_tasks == 2:
            # Define epsilon ranges based on task pairs
            if task_indices[0] == 0 and task_indices[1] == 131:
                epsilon_values = [[x] for x in np.linspace(0.0001, 0.0015, 10)]
            elif task_indices[0] == 0 and task_indices[1] == 135:
                epsilon_values = [[x] for x in np.linspace(0.001, 0.015, 10)]
        else:  # 5 tasks
            sets = [[20, 35, 50], [40, 70, 100], [70, 125, 180], [120, 310, 500]]
            epsilon_values = np.array(list(itertools.product(*sets)))
        return None, epsilon_values
    
    else:  # weighted sum or other methods
        if num_tasks == 2:
            return [[x, 1 - x] for x in np.linspace(0.001, 0.991, 10)], None
        else:  # 5 tasks
            return np.loadtxt("weights-5tasks.txt").tolist(), None

def run():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    args = parse_args()

    # read config
    config = Config.from_json(args.config_file_path)
    # Modify device selection to handle non-CUDA environments
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(42)  # Set CUDA seeds only if CUDA is available
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')
        # logger.info("CUDA is not available, using CPU instead")
    
    config.data.path = 'datasets/MSLR-WEB30K/Fold1_normalized_' + args.dataset_name
    config.loss.args['reduction'] = args.reduction_method
    
    # Parse task indices and weights
    task_indices = list(map(int, args.task_indices.split(",")))
    task_weights = list(map(int, args.task_weights.split(",")))
    min_weight, max_weight = task_weights
    num_tasks = len(task_indices)
    
    # Indices for different types of labels in the MSLR-WEB30K dataset
    label_indices = [131, 132, 133, 135]  # Web30K dataset indices

    # Get weight combinations based on number of tasks and MOO method
    weight_combinations, epsilon_values = get_weight_combinations(args.moo_method, num_tasks, task_indices)

    # Main task is always the first task (index 0)
    main_task_index = 0

    # Data Loading Section
    train_dataset, val_dataset = load_libsvm_dataset(
        input_path=config.data.path,
        slate_length=config.data.slate_length,
        validation_ds_role=config.data.validation_ds_role,
    )

    # Calculate number of features by excluding the label columns
    n_features = train_dataset.shape[-1] - len(label_indices)  # remove label candidates from the input
    assert n_features == val_dataset.shape[-1] - len(label_indices), "Last dimensions of train_dataset and val_dataset do not match!"

    # Create data loaders for batched processing
    train_dataloader, val_dataloader = create_data_loaders(
        train_dataset, val_dataset, num_workers=config.data.num_workers, batch_size=config.data.batch_size)

    # Training Loop
    for weight_index in range(min_weight, max_weight):
        # Set task weights and epsilon based on the specified MOO method
        if args.moo_method == "ec":
            epsilon = epsilon_values[weight_index]
            task_weights_tensor = torch.tensor([1] + [0] * len(epsilon))
        else:
            task_weights_tensor = weight_combinations[weight_index]
            epsilon = None

        # Construct run ID and paths
        task_id_string = "_task_".join(map(str, task_indices))
        run_id = os.path.join(
            f"{num_tasks}tasks",
            f"task_{task_id_string}",
            args.dataset_name,
            args.reduction_method,
            args.moo_method,
            str(weight_index)
        )
        args.run_id = run_id
        # Set up paths and logging for this run
        paths = PathsContainer.from_args(args.output_dir, args.run_id, args.config_file_path)
        create_output_dirs(paths.output_dir)
        
        logger = init_logger(paths.output_dir)
        logger.info(f"Created paths container: {paths}")

        # Copy config file to output directory for reproducibility
        output_config_path = os.path.join(paths.output_dir, "used_config.json")
        execute_command(f"cp {paths.config_path} {output_config_path}")

        # Set up results directory and filename for storing training and evaluation metrics
        results_dir = os.path.join(
            "result",
            f"{num_tasks}tasks",
            f"task_{task_id_string}",
            args.dataset_name,
            args.reduction_method,
            args.moo_method
        )

        results_filename = os.path.join(
            results_dir,
            f"results_task_{task_id_string}_"
            f"in_total_{num_tasks}_tasks_{args.moo_method}_average_weight{weight_index}.txt"
        )

        # Ensure clean results directory exists
        if os.path.exists(results_filename):
            os.remove(results_filename)
        os.makedirs(os.path.dirname(results_filename), exist_ok=True)

        logger.info(f"Model training will execute on device: {device}")

        # Initialize ranking model with the feature dimensions
        model = make_model(n_features=n_features, **asdict(config.model, recurse=False))
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = CustomDataParallel(model)  # Use multiple GPUs if available
        model.to(device)

        # Setup training components: optimizer, loss function, and scheduler
        optimizer = getattr(optim, config.optimizer.name)(params=model.parameters(), **config.optimizer.args)
        loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)
        scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.name)(optimizer, **config.lr_scheduler.args) if config.lr_scheduler.name else None

        # Train model with anomaly detection if configured
        with torch.autograd.detect_anomaly() if config.detect_anomaly else dummy_context_mgr():
            result = fit(
                moo_method=args.moo_method,
                main_task_index=main_task_index,
                task_indices=task_indices,
                label_indices=label_indices,
                results_filename=results_filename,
                model=model,
                loss_func=loss_func,
                task_weights=task_weights_tensor,
                epsilon=epsilon,
                optimizer=optimizer,
                scheduler=scheduler,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                config=config,
                device=device,
                output_dir=paths.output_dir,
                tensorboard_output_path=paths.tensorboard_output_path,
                **asdict(config.training)
            )

        # Save results and verify metrics against expected values
        dump_experiment_result(args, config, paths.output_dir, result)

        if urlparse(args.output_dir).scheme == "gs":
            copy_local_to_gs(paths.local_base_output_path, args.output_dir)

        # Verify results match expected metrics from config
        assert_expected_metrics(result, config.expected_metrics)

if __name__ == "__main__":
    run()