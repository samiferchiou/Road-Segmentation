# Machine Learning Project 2 - Road Segmentation - 2021-2022

The purpose of this project is to implement machine learning algorithms for binary classification of pixels on sateliitie images between road and not road. The resulting predictions are submitted to [AIcrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/).

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
First, you need to create a Google Colab Notebook to clone our repositor by running command lines in a cell.
Clone the repository in you Google Drive using the following command via ssh:
```
!git clone git@github.com:etiennebruno/cs433_project_2.git
```
Open the u-net.ipynb notebook and run all the cells one by one.

To obtain the results published in our report, go back to your drive folder and download the csv_file that has been generated. You can ow upload it to AiCrowd to get the same score.


## Overview
Here's a list of the relevant source files 

|Source file | Description|
|---|---|
|`implementations.py`   | Regrouping the six machine learning algorithms we hqve developped for this project as well as dependant function|
|`run.py`               | Main script containing the solution of the problem producing our highest prediction score|
|`proj1_helpers.py`     | Containing additional functions used in the project|
|`projet1.ipynb`        | Notebook of the project with all the visualization and the analysis of the training data as weel as the code of the training models|


# Project Road Segmentation

For this choice of project task, we provide a set of satellite images acquired 
from GoogleMaps. We also provide ground-truth images where each pixel is labeled 
as road or background. 

Your task is to train a classifier to segment roads in these images, i.e. 
assigns a label `road=1, background=0` to each pixel.

Submission system environment setup:

1. The dataset is available from the 
[CrowdAI page](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation).

2. Obtain the python notebook `segment_aerial_images.ipynb` from this github 
folder, to see example code on how to extract the images as well as 
corresponding labels of each pixel.

The notebook shows how to use `scikit learn` to generate features from each 
pixel, and finally train a linear classifier to predict whether each pixel is 
road or background. Or you can use your own code as well. Our example code here 
also provides helper functions to visualize the images, labels and predictions. 
In particular, the two functions `mask_to_submission.py` and 
`submission_to_mask.py` help you to convert from the submission format to a 
visualization, and vice versa.

3. As a more advanced approach, try `tf_aerial_images.py`, which demonstrates 
the use of a basic convolutional neural network in TensorFlow for the same 
prediction task.

Evaluation Metric:
 [F1 score](https://en.wikipedia.org/wiki/F1_score)
