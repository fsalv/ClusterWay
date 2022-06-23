[![arXiv](http://img.shields.io/badge/arXiv-2001.09136-B31B1B.svg)](https://arxiv.org/abs/2010.16322)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<h1 align="center"> ~ ClusterWay ~ </h1>

<p align="center">
  <img src=media/deep_way_net.png>
</p>


This repository contains all the code related to the paper [Waypoint Generation in Row-based Crops with Deep Learning and Contrastive Clustering](https://arxiv.org/abs/2010.16322), a deep learning model able to predict the position of waypoints in row-crop fields and cluster them with a single forward pass.

# 1 Getting Started
## 1.1 Installation

1. Clone this repository

``` git clone  https://github.com/fsalv/ClusterWay.git ```

2. Install the required packages

``` 
cd ClusterWay
pip install -r requirements.txt
```
We recommend to do it in a separate virtual environment with respect to your main one to avoid compatibility issues for packages versions. In this case, remember to create a jupyter kernel linked to the new environment.

# 2 Dataset

You can find synthetic and real-world datasets used in the paper experimentation under the ```Datasets``` folder. If you want to generate new synthetic images you can run the jupyter notebook ```Artificial Dataset Generator Curved.ipynb``` under the ```CurvedGenerator``` folder.  You can modify useful parameters in the first cells of the notebook.


# 2 Model training

All the code related to model training and testing is inside ```ClusterWay``` folder. To re-train DeepWay or ClusterWay run ```train.py```. You can choose which model to train with the ```--name``` argument. The ```--curved``` argument allows to choose wether to train on straight or curved datasets. Be aware that in order to train the clustering head, you should have already trained a correspondent classic DeepWay model. All the options can be visualized with ```python train.py --help```. As an example, to train ClusterWay on curved dataset:

``` 
python train.py --name deep_way --curved
python train.py --name cluster_way --curved
``` 

You can modify network parameters inside the configuration file  ```config.json```. In particular, by modifying the ```DATA_N``` and ```DATA_N_VAL``` values you can choose to train/validate with fewer images to see how prediction quality changes with dataset dimension. Setting ```accumulating_gradients``` you can train accumulating batches before computing gradients, in order to reach the desired ```BATCH_SIZE```, but requiring ```BATCH_SIZE_ACC``` GPU memory only. This can be useful if you encounter out-of-memory errors. You can also modify the network architecture changing ```R```, the number of ```FILTERS``` per layer, the ```KERNEL_SIZE``` or clustering head space dimensionality ```OUT_FEATS```. Remember to change compression value ```K``` according to the chosen ```R```, as shown in the paper.

You can find a script to train all the models three times to reproduce the complete experimantion inside the ```scripts``` folder:

```
./scripts/training.sh
```


# 2 Model testing


You can test DeepWay on both the satellite and synthethic test datasets with the notebook ```DeepWay Test.ipynb```. This notebooks allows you to compute the AP metric on the selected images. You can change the test set inside the notebook in the section *Import the Test Dataset*. If you set ```name_model = 'deep_way_pretrained.h5'``` in the third cell, you can use the weights pretrained by us.

# 3 Path planning
<p align="center">
  <img src=media/deepway.png>
</p>
To generate the paths with the A* algorithm and compute the coverage metric, you can use the ``` Prediction and Path Planning.ipynb``` notebook. Again, you can change the test set inside the notebook to select satellite or synthethic datasets. Note that the A* execution will require a lot of time, exspecially if it finds some trouble in generating the path for too narrow masks.
<br/><br/>

**Warning:** If you don't have gpu support, comment the fourth cell (*"select a GPU and set memory growth"*).
<br/>

## Citation
If you enjoyed this repository and you want to cite our work, you can refer to the Arxiv version of the paper [here](https://doi.org/10.1016/j.compag.2021.106091).

```
BIBTEX
```

<br/>
<sub> <b> Note on the satellite dataset: </b> </br>
The 150 masks of the real-world remote-sensed dataset (100 straight and 50 curved) have been derived by manual labeling of images taken from Google Maps. Google policy for the products of its satellite service can be found [here](https://www.google.com/permissions/geoguidelines/). Images can be used for reasearch purposes by giving the proper attribution to the owner. However, for this repository we chose to release masks only and not the original satellite images. </sub> 
