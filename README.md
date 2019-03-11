# LIME-KNN

This project is a fork of the original LIME repository (https://github.com/marcotcr/lime), which is based on the following paper (https://arxiv.org/abs/1602.04938).
As of now, the LIME-KNN explanations work on Tabular data and Image data.

## Installation Requirements

The Installation requirements are almost identical to the original LIME repository (https://github.com/marcotcr/lime).
Run "python setup.py install" to install any missing requirements.

## KNN explanation for Tabular Data

# Available options for KNN-LIM 
The "explainInstanceUsingKNN()" requires the following inputs:
* originalClassifier - The original complex classifier. 
* num\_samples - The number of neighbourhood points used to learn a KNN model locally
* num\_neighbours - The 'K' parameter in KNN
* feature\_names - The names of the features in an array
* class\_names - The interpretation of the output labels in an array
* dataPointInterpretation - The meaning of each data point. For example, for the Iris dataset, each datapoint is an "Iris Flower"
* scalar - the scalar used to scale the data to 0 mean and unit standard deviation

# Driver code (knn\_demo.ipynb)
* Run "python setup.py install" to install 
*  This is a driver program which calls the KNN explanation code
* It starts off by loading the IRIS-dataset, creating the test-train split, and scaling the data to a mean of 0 and standard deviation of 1.
* Learn a Random Forest Classifier (which classifier is used does not matter, since LIME is model-agnostic)
* Ask LIME-KNN to provide an explanation for one of the points, and display the HTML output of the explanation.
	
# KNN-explanation code (lime.lime\_knn\_tabular)
* Exposes the "explainInstanceUsingKNN()" function which returns an HTML string of the visual explanation of a data point.
* It follows a similar logic to LIME, wherein we locally learn a simple, interpretable classifier (KNN) to explain a more complex classifier.
* We first generate a set of data points in the neighbourhood of the point to be explained, and learn a KNN classifier on this data. The ground-truth labelling for the neighbourhood data is obtained by querying the complex classifier to make a prediction on that data.
* Finally, using this locally-learned KNN-classifier, to explain the required instance. 	
		

## KNN explanation for Image Data
	
# Available options for KNN-LIM
* Same as the tabular data only feature names are not needed
* Specify the segmentation algorithm (currently support k-means clustering algorithm)
		
# Code (knn\_image\_demo.ipynb)
* It starts off by loading image [dataset](https://drive.google.com/drive/folders/1XaFM8BJFligrqeQdE-_5Id0V_SubJAZe?usp=sharing)
* Learns a simple CNN classifier (with the accuracy of 99.87%)
* Preprocesses the image by calling segmentation algorithm
* Finds corresponding points in vicinity (by removing segments permutatively
* Sends to the explanation function to return k nearest neighbors for explanation with their corresponding labels.

Further details can be found in the report available on the github repo.
