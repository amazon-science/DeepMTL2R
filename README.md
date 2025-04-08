
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

## Citation 
If you use this work, or otherwise found our work valuable, please consider citing the paper:

```
@article{chaoshengmo-lightgbm2025,
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
We would like to thank the authors of the following repositories, upon which we built the present codebase:
[allRank](https://github.com/allegro/allRank/), [FAMO](https://github.com/Cranial-XIX/FAMO), [SDMGrad](https://github.com/OptMN-Lab/sdmgrad/tree/main), [MGDA](https://github.com/isl-org/MultiObjectiveOptimization).