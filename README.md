# LIME-KNN

This project is a fork of the original LIME repository (https://github.com/marcotcr/lime), which is based on the following paper (https://arxiv.org/abs/1602.04938).
<Insert something from the paper's introduction>
As of now, the LIME-KNN explanations work on Tabular data and Image data.

A] Installation Requirements

The Installation requirements are almost identical to the original LIME repository (https://github.com/marcotcr/lime).
Run "python setup.py install" to install any missing requirements.

B] KNN explanation for Tabular Data
	
	1) Available options for KNN-LIM
		The "explainInstanceUsingKNN()" requires the following inputs:
			i) originalClassifier - The original complex classifier. 
			ii) num_samples - The number of neighbourhood points used to learn a KNN model locally
			iii) num_neighbours - The 'K' parameter in KNN
			iv) feature_names - The names of the features in an array
			v) class_names - The interpretation of the output labels in an array
			vi) dataPointInterpretation - The meaning of each data point. For example, for the Iris dataset, each datapoint is an "Iris Flower"
			vii) scalar - the scalar used to scale the data to 0 mean and unit standard deviation
	2) Driver code (knn_demo.ipynb)
		- Run "python setup.py install" to install 
		- This is a driver program which calls the KNN explanation code
		- It starts off by loading the IRIS-dataset, creating the test-train split, and scaling the data to a mean of 0 and standard deviation of 1.
		- Learn a Random Forest Classifier (which classifier is used does not matter, since LIME is model-agnostic)
		- Ask LIME-KNN to provide an explanation for one of the points, and display the HTML output of the explanation.
	
	<Vidushi, should I remove this part?>
	3) KNN-explanation code (lime.lime_knn_tabular)
		- Exposes the "explainInstanceUsingKNN()" function which returns an HTML string of the visual explanation of a data point.
		- It follows a similar logic to LIME, wherein we locally learn a simple, interpretable classifier (KNN) to explain a more complex classifier.
		- We first generate a set of data points in the neighbourhood of the point to be explained, and learn a KNN classifier on this data. The ground-truth labelling for the neighbourhood data is obtained by querying the complex classifier to make a prediction on that data.
		- Finally, using this locally-learned KNN-classifier, to explain the required instance. 	
		