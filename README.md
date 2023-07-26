# DRGBCN
## Dependencies
* python == 3.6.12
* pytorch == 1.6.0
* pytorch-lightning==1.0.8
* torch_geometric
* scikit-learn 
* seaborn
* matplotlib
* openpyxl
* xlrd

## Installation Guide
Clone this GitHub repo and set up a new conda environment.
# create a new conda environment
* conda create -n drgbcn python=3.6.12
* conda activate drgbcn
# install requried python dependencies
* pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
* pip install pytorch-lightning==1.0.8
* pip install scikit-learn
* pip install pandas


## Datasets
* Fdataset and Cdataset https://github.com/BioinformaticsCSU/BNNR
* LRSSL https://github.com/linwang1982/DRIMC

### Usage
```shell
cd DRGBCN
python demo_t.py
```
### Device
* RTX2080Ti
