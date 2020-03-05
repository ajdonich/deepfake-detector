## Project: kaggle-deepfake-detection 

This project is designed for submission to Kaggle's Deepfake Detection competition.  
See: [Kaggle Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge/overview)

NOTE: you have to download Kaggle's little sample dataset manually (with your browser or kaggle CLI).  
Create a data subdirectory in the repo root directory and unzip the sample dataset there. The full  
dataset can be downloaded using 

___


### Installation:

The repository is currently setup for Python 3.7 miniconda configuration management, with packages governed by deepfake.yml file.    
You can access miniconda download and installation instructions here: [Miniconda Installation Instructions](https://docs.conda.io/en/latest/miniconda.html)

Once miniconda is successfully installed, the following commands may be executes to install the project (for further Conda  
Environment reference, this blog article may be useful: [Getting Started w/Conda Environments](https://towardsdatascience.com/getting-started-with-python-environments-using-conda-32e9f2779307)):

```
$ git clone https://github.com/ajdonich/kaggle-deepfake-detection.git
$ cd kaggle-deepfake-detection
$ conda env create -f ./deepfake.yml
```

___


### Execution:

Currently all implementation is in IPython/Jupyter notebooks.  
To launch a notebook session, excecute the following from the command line:

```
$ conda activate deepfake
$ jupyter-lab
```

The notebooks directory contains the following .ipynb files for downloading and preprocessing data:

1. data_downloader.ipynb
2. deepfake_preprocessor.ipynb

