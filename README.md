
# Deep Multi-task Learning to Rank 

## Introduction
DeepMTL2R is a deep learning framework used for multi-task learning to rank tasks. 

## Setup environment
<!-- follow https://github.com/OptMN-Lab/fairgrad -->
<!-- https://github.com/Cranial-XIX/FAMO -->
<!-- https://github.com/allegro/allRank/tree/master/allrank -->
aws s3 sync s3://personal-tests/chaosd/DeepMTL2R-dev/ DeepMTL2R/

### Setup enviroment for running dmtl2r
```
conda create -n dmtl2r python=3.9.7
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dmtl2r

cd DeepMTL2R
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu113

chmod +x *.sh
```

### Setup enviroment for plotting and computing metrics
```
conda create -n pygmo python=3.9.7
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pygmo

cd DeepMTL2R
pip install -r requirements-hvi.txt
conda install pygmo

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pygmo
```

#### Add a Conda environment to Jupyter Notebook
```
conda install ipykernel
python -m ipykernel install --name pygmo --display-name pygmo
```

## Usage 
To train the model, configure the experiment in a config.json file. The code in allrank provides the core components for model training. The task-specific files in DeepMTL2R uses the core modules to run experiments.

We provide one example using MSLR30K data as follows.
```
CUDA_VISIBLE_DEVICES=0 python main_ntasks.py \
            --config-file-path scripts/local_config_web.json \
            --output-dir "allrank/run"
            --task-indices 0,135 \
            --task-weights 0,10 \
            --moo-method ls \
            --dataset-name "original" \
            --reduction-method "mean" 
```
We also provide run_2tasks_web30k.sh and run_5tasks_web30k.sh script to run the experiments in our paper which trains Transformer models on the MSLR30K data for two tasks and five tasks, respectively.

## MTL methods

We support the following MTL methods with a unified API. 

| Method (code name) | Paper (notes) |
| :---: | :---: |
| FAMO (`famo`) | [Fast Adaptive Multitask Optimization](https://arxiv.org/abs/2306.03792.pdf) |
| Nash-MTL (`nashmtl`) | [Multi-Task Learning as a Bargaining Game](https://arxiv.org/pdf/2202.01017v1.pdf) |
| CAGrad (`cagrad`) | [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/pdf/2110.14048.pdf) |
| PCGrad (`pcgrad`) | [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782) |
| IMTL-G (`imtl`) | [Towards Impartial Multi-task Learning](https://openreview.net/forum?id=IMPnRXEWpvr) |
| MGDA (`mgda`) | [Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/abs/1810.04650) |
| DWA (`dwa`) | [End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704) |
| Uncertainty weighting (`uw`) | [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115v3.pdf) |
| Linear scalarization (`ls`) | - (equal weighting) |
| Scale-invariant baseline (`scaleinvls`) | - (see Nash-MTL paper for details) |
| Random Loss Weighting (`rlw`) | [A Closer Look at Loss Weighting in Multi-Task Learning](https://arxiv.org/pdf/2111.10603.pdf) |

## Citation 
If you use this work, or otherwise found our work valuable, please consider citing the paper:

```
@article{chaoshengdong-deepmtl2r2025,
  title={Deep Multi-task Learning to Rank},
  author={Dong, Chaosheng and Xiao, Peiyao and Ji, Kaiyi},
  year={2025}
}

```

## Contact
For any question, you can contact chaosd@amazon.com.


## License

This project is licensed under the Apache-2.0 License.

## Acknowlegement 
We thank authors of the following repositories, upon which we built the present codebase:
[allRank](https://github.com/allegro/allRank/), [FAMO](https://github.com/Cranial-XIX/FAMO), [SDMGrad](https://github.com/OptMN-Lab/sdmgrad/tree/main), [MGDA](https://github.com/isl-org/MultiObjectiveOptimization), [EPO](https://github.com/dbmptr/EPOSearch), [MO-LightGBM](https://github.com/amazon-science/MO-LightGBM).