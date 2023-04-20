# Machine Learning Project 2 - Road Segmentation - 2021-2022

The purpose of this project is to implement machine learning algorithms for binary classification of pixels on satelite images between road and not road. The resulting predictions are submitted to [AIcrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/).

## Team Members
Sami FERCHIOU : sami.ferchiou@epfl.ch <br/>
Miguel SANCHEZ : miguel-angel.sanchezndoye@epfl.ch <br/>
Etienne BRUNO : etiene.bruno@epfl.ch <br/>

##  Objective
The aim of this project called Road Segmentation is to design a model able to detect roads in a set of satellite images (acquired from Google Maps). We then implemented a convolutional neural network based on the U-NET architecture. This neural network was trained over an original set of 100 images, augmented and filtered to 2300 images with their respective ground truth images, representing the optimal results that we expect from this model. Using a K-Fold cross-validation, we were able to optimize the hyper-parameters of our model in order to realize a final prediction [#169412](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/submissions/169412), over a test set of 50 images, that had an accuracy of 94.2\% and an F1 score of 89.2\% .

## Requirements
To run ou model, we used Google Colab Pro + which give us access to :
- Intel(R) Xeon(R) CPU @ 2.30GHz
- 52 G of RAM
- Tesla P100-PCIE
This configuration is required to train our model (at least). However, you cqn still load our model with less ressources.

## Instructions
First, you need to create a Google Colab Notebook to clone our repository by running command lines in a cell.
Clone the repository in you Google Drive using the following command via ssh. To avoid having to change the path, please create a folder called 'ml_epfl' at the root of your Google Drive. Then clone our repo as follows:
```
!git clone git@github.com:etiennebruno/cs_433_ml_project_2.git ml_road_segmentation
```
Open the u-net.ipynb notebook and run all the cells one by one. You will have the option to run the K-FOLD cross validation for hyper parameters, the K-FOLD for training with bagging or simply to load an existing model. If you want to load an axisting model, pleased directly connect to our [Google Drive](https://drive.google.com/drive/folders/1-R3SQ62_dRcnp_eogn1_oclSGheUOGrz?usp=sharing) and launch our notebook directly from there since we could not (size limit) upload our weights on github.

To obtain the results published in our report, go back to your drive folder and download the csv_file that has been generated (called submission_bagging.csv). You can ow upload it to AiCrowd to get the same score.


## Overview
Here's a list of the relevant source files 

|Source file | Description|
|---|---|
|`unet.ipynb`           | The notebook you can run entrely to train or test a model|
|`data_analysis.ipynb`  | Data analysis notebook after the k-fold cross validation launched to get the best hyperparameters|
|`models.py`            | Three different unet models explained in our report (with one or filters and with a triple convolution)|
|`dataset_loading.py`   | Class that represents our dataset|

You will also find a lot of checkpoint files that corresponds to various models and predictions we experimentes througout this project.
