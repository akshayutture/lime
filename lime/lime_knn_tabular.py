import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import subprocess

def explainInstanceUsingKNN(originalClassifier,instancePoint,num_samples,num_neighbours,feature_names,class_names,dataPointInterpretation,scalar):
	#Get the label for the instance point as given by the classifier
	instancePointIn2darray = np.reshape(instancePoint,(1,-1))
	instancePointLabel = (originalClassifier.predict(instancePointIn2darray))[0]

	#Compute the set of points in the vicinity of the instancePoint
	trainingDataInVicinity = getDataPointsInVicinity(instancePoint,num_samples,0.5)

	#Get the labels for these points. For the purpose of KNN , this is the ground truth
	labelsOfDataInVicinity = originalClassifier.predict(trainingDataInVicinity)
	
	#Run the KNN classifier, with this ground truth, and predict the neighbours of the instance point.
	knnClassifier = KNeighborsClassifier(num_neighbours)
	knnClassifier.fit(trainingDataInVicinity,labelsOfDataInVicinity)
	distances,indices = knnClassifier.kneighbors(instancePointIn2darray,num_neighbours)

	#Get all the similar points with the same label.
	similarPointsForExplanation = []
	for index in indices[0]:
		if labelsOfDataInVicinity[index]==instancePointLabel:
			similarPointsForExplanation.append(trainingDataInVicinity[index])
	
	
	#Scale back to the original dimensions
	unscaledSimilarPointsForExplanation = scalar.inverse_transform(similarPointsForExplanation)
	unScaledInstancePointIn2darray = scalar.inverse_transform(instancePointIn2darray)
	

	#TERMINAL OUTPUT
	#Print instance point.
	'''
	print(dataPointInterpretation + " to Explain is")
	for featureIndex in range(len(unScaledInstancePointIn2darray[0])):
		print(feature_names[featureIndex] + ":" + str(round(unScaledInstancePointIn2darray[0][featureIndex],2)))
	print("")

	#Print out all the similar points with the same label in a verbose way
	print("Reason for classification as " + class_names[instancePointLabel] + " is that it is similar to these other " + dataPointInterpretation + ":\n")
	for i in range(len(unscaledSimilarPointsForExplanation)):
		print(dataPointInterpretation + " " + str(i) + ":")
		for featureIndex in range(len(unscaledSimilarPointsForExplanation[i])):
			print(feature_names[featureIndex] + ":" + str(round(unscaledSimilarPointsForExplanation[i][featureIndex],2)))
		print("")
	'''

	#HTML OUTPUT
	#show instance point
	htmlContentString = '<button class="collapsible">' + dataPointInterpretation + " to explain.</button>"
	htmlContentString += '<div class="content">'
	for featureIndex in range(len(unScaledInstancePointIn2darray[0])):
		htmlContentString += "<b>" + feature_names[featureIndex] + "</b> : " + str(round(unScaledInstancePointIn2darray[0][featureIndex],2)) + "<br>"
	htmlContentString += "</div>"

	#Show all the similar points with the same label with a collapsible UI
	htmlContentString += "<h1> Reason for classification as " + class_names[instancePointLabel] + " is that it is similar to these other " + dataPointInterpretation + ":</h1>"

	for i in range(len(unscaledSimilarPointsForExplanation)):
		htmlContentString += '<button class="collapsible">' + dataPointInterpretation + " " + str(i) + "</button>"
		htmlContentString += '<div class="content">'
		for featureIndex in range(len(unscaledSimilarPointsForExplanation[i])):
			htmlContentString += "<b>" + feature_names[featureIndex] + "</b> : " + str(round(unscaledSimilarPointsForExplanation[i][featureIndex],2)) + "<br>"
		htmlContentString += '</div>'

	return HTMLboilerplateCodeHead() + htmlContentString + HTMLboilerplateCodeTail()



def getDataPointsInVicinity(instancePoint,num_samples,stdDev):
	numberOfFeatures = instancePoint.shape[0]
	perturbations = np.random.normal(0,stdDev,num_samples*numberOfFeatures)
	trainingDataInVicinity = np.empty([num_samples,numberOfFeatures], dtype = float)

	for i in range(num_samples):
		for j in range(numberOfFeatures):
			a = instancePoint[j]
			b = perturbations[i*numberOfFeatures + j]
			trainingDataInVicinity[i][j] = a + b

	return trainingDataInVicinity

def HTMLboilerplateCodeHead():
	return """<!DOCTYPE html>
	<html>
	<head>
	<link rel="stylesheet" type="text/css" href="test.css">
	</head>
	<body>
	"""

def HTMLboilerplateCodeTail():
	return '<script src="test.js"></script></body></html>'

def openHTMLStringInBrowser(explanation):
	with open("test.html", "w") as f:
		f.write(explanation)
	subprocess.run(["open", "test.html"])