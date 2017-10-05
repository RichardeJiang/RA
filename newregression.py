import os
import sys
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from collections import Counter
import statsmodels.api as sm
import statsmodels.formula.api as smapi
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def writeScore(dictFile, fileName):
	theFile = open(fileName, "w")
	for item in dictFile:
		theFile.write("%s: " % str(item))
		theFile.write("%s\n" % str(dictFile[item]))
	theFile.close()
	return

def writeList(listFile, fileName):
	theFile = open(fileName, "w")
	for item in listFile:
		theFile.write("%s\n" % str(item))
	theFile.close()
	return

def readTimeSeriesData(fileName):
	fp = open(fileName, "r")
	phraseList = []
	timeSeries = []
	for line in fp:
		temp = line.split(":")
		phraseList.append(temp[0])
		timeSeries.append([float(ele) for ele in temp[1].split(" ")])

	return phraseList, timeSeries

def splitData2(timeSeries, windowSize):
	resultXList = []
	resultYList = []

	tempLine = timeSeries[0]
	size = len(tempLine) - windowSize
	# print size
	seriesSize = len(timeSeries)
	# print seriesSize
	for index in range(size):
		resultX = []
		resultY = []
		for lineIndex in range(seriesSize):
			training = timeSeries[lineIndex][index: index + windowSize]

			# temp = training[-1]
			# training.append(temp ** 2)

			# for windowStep in range(windowSize):
			# 	training.append(training[windowStep] ** 2)
			# for intermediate in range(windowSize - 1):
			# 	training.append(training[intermediate + 1] - training[intermediate])
			training = addFeatures(training, 5)
			training.insert(0, lineIndex)
			resultX.append(training)
			resultY.append(timeSeries[lineIndex][index + windowSize])
		resultXList.append(resultX)
		resultYList.append(resultY)

	return resultXList, resultYList, size

def splitData3(timeSeries, windowSize):
	resultX = []
	resultY = []
	powerTerms = []

	tempLine = timeSeries[0]
	size = len(tempLine) - windowSize

	seriesSize = len(timeSeries)

	for index in range(size):
		for lineIndex in range(seriesSize):
			training = timeSeries[lineIndex][index: index + windowSize]

			# training = addFeaturesKF(training, 4)
			# powerTerms.append(combination(training))
			
			training.insert(0, lineIndex) # format of data: [year index, phrase index, scores]
			training.insert(0, index)
			resultX.append(training)
			resultY.append(timeSeries[lineIndex][index + windowSize])
			
	return resultX, resultY, powerTerms, size

def addFeatures(currentTrainingSample, selectedFeatureType = 0):
	''' feature type: 0: none; 1: x+x**2; 2: x2-x1; 3: log(x); 4: x1*x2 '''
	result = currentTrainingSample
	windowSize = len(currentTrainingSample)
	if selectedFeatureType == 1:
		for windowStep in range(windowSize):
			result.append(result[windowStep] ** 2)
	elif selectedFeatureType == 2:
		for windowStep in range(windowSize - 1):
			result.append(result[windowStep + 1] - result[windowStep])
	elif selectedFeatureType == 3: # Note: log indicates that if the score is <= 1, the final score will be <= 0, and 0 will gives none value
		result = np.log(result).tolist()
	elif selectedFeatureType == 4: # Note: currently only implemented the consecutive terms' cross product
		for windowStep in range(windowSize - 1):
			result.append(result[windowStep] * result[windowStep + 1])
	elif selectedFeatureType == 5:
		for windowStep in range(windowSize):
			result.append(result[windowStep] ** 2)
		for windowStep in range(windowSize - 1):
			result.append(result[windowStep] * result[windowStep + 1])
	elif selectedFeatureType == 6: # x1x1 x2x2 x3x3 x2x3 : obtained by RMSe without year as input
		for windowStep in range(windowSize):
			result.append(result[windowStep] ** 2)
		result.append(result[windowSize - 1] * result[windowSize - 2])
	elif selectedFeatureType == 7: # x2x2 x3x3 : obtained by RMSe with year inside input (start from 0, not 1955)
		result.append(result[-1] ** 2)
		result.append(result[-3] ** 2)
	return result

def addFeaturesComp(currentTrainingSample, selectedFeatureType = 1):
	''' feature type: 0: none; 1: x+x**2; 2: x2-x1; 3: log(x); 4: x1*x2 '''
	result = currentTrainingSample
	windowSize = len(currentTrainingSample)
	if selectedFeatureType == 1:
		result.append(result[0] * result[2])
		result.append(result[1] ** 2)
		result.append(result[1] * result[2])
		result.append(result[2] ** 2)
	elif selectedFeatureType == 2:
		for step in range(windowSize):
			result.append(result[step] ** 2)
		result.append(result[0] * result[1])
		result.append(result[0] * result[2])
		result.append(result[1] * result[2])
	elif selectedFeatureType == 3: # Note: log indicates that if the score is <= 1, the final score will be <= 0, and 0 will gives none value
		for step in range(windowSize):
			result.append(result[step] ** 2)
	elif selectedFeatureType == 4: # Note: currently only implemented the consecutive terms' cross product
		result.append(result[0] * result[1])
		result.append(result[2] ** 2)
	elif selectedFeatureType == 5:
		for windowStep in range(windowSize):
			result.append(result[windowStep] ** 2)
		for windowStep in range(windowSize - 1):
			result.append(result[windowStep] * result[windowStep + 1])
	elif selectedFeatureType == 6:
		result.append(result[0] * result[1])
		result.append(result[0] * result[2])
		result.append(result[2] ** 2)
	elif selectedFeatureType == 7:
		result.append(result[2] ** 2)
	elif selectedFeatureType == 8:
		result.append(result[0] ** 2)
		result.append(result[0] * result[2])
		result.append(result[2] ** 2)
	elif selectedFeatureType == 9:
		result.append(result[0] ** 2)
		result.append(result[0] * result[1])
		result.append(result[2] ** 2)
	elif selectedFeatureType == 10:
		result.append(result[0] ** 2)
		result.append(result[1] ** 2)
		result.append(result[1] * result[2])
		result.append(result[2] ** 2)
	return result

def addFeaturesKF(currentTrainingSample, selectedFeatureType = 1):
	''' feature type: 0: none; 1: x+x**2; 2: x2-x1; 3: log(x); 4: x1*x2 '''
	result = currentTrainingSample
	windowSize = len(currentTrainingSample)
	if selectedFeatureType == 1:
		result.append(result[0] ** 2)
	elif selectedFeatureType == 2:
		result.append(result[0] * result[1])
	elif selectedFeatureType == 3: # Note: log indicates that if the score is <= 1, the final score will be <= 0, and 0 will gives none value
		result.append(result[0] ** 2)
		result.append(result[0] * result[1])
	elif selectedFeatureType == 4: # Note: currently only implemented the consecutive terms' cross product
		for windowStep in range(windowSize):
			result.append(result[windowStep] ** 2)
		for windowStep in range(windowSize - 1):
			result.append(result[windowStep] * result[windowStep + 1])
		result.append(result[0] * result[2])
	elif selectedFeatureType == 5:
		for windowStep in range(windowSize):
			result.append(result[windowStep] ** 2)
		for windowStep in range(windowSize - 1):
			result.append(result[windowStep] * result[windowStep + 1])
	elif selectedFeatureType == 6:
		result.append(result[0] * result[2])
		result.append(result[1] ** 2)
		result.append(result[1] * result[2])
		result.append(result[2] ** 2)
	elif selectedFeatureType == 7:
		result.append(result[0] ** 2)
		result.append(result[1] ** 2)
		result.append(result[1] * result[2])
		result.append(result[2] ** 2)
	elif selectedFeatureType == 8:
		result.append(result[0] * result[1])
		result.append(result[1] ** 2)
		result.append(result[1] * result[2])
		result.append(result[2] ** 2)
	elif selectedFeatureType == 9:
		result.append(result[0] * result[1])
		result.append(result[0] * result[2])
		result.append(result[1] ** 2)
		result.append(result[1] * result[2])
		result.append(result[2] ** 2)
	elif selectedFeatureType == 10:
		result.append(result[0] ** 2)
		result.append(result[1] * result[2])
		result.append(result[2] ** 2)
	return result

def combination(currentTrainingSample):
	''' The power set iterates through all the possible combination of additional features '''
	powerset = []
	windowSize = len(currentTrainingSample)
	for i in range(windowSize):
		for j in range(i, windowSize):
			powerset.append(currentTrainingSample[i] * currentTrainingSample[j])
	return powerset

def generatePowerset(s):
	resultList = []
	x = len(s)
	for i in range(1<<x):
		resultList.append([s[j] for j in range(x) if (i & (1 << j))])

	return resultList

def mapPhraseListUsingIndex(phraseIndexList, phraseList):
	# result = [phraseList[ele] for ele in phraseIndexList]
	result = []
	for index in range(len(phraseList)):
		if index in phraseIndexList:
			result.append(phraseList[index])
	return result

def writePhraseListTotal(phraseList, fileName):
	tf = open(fileName, "w")

	for phraseListThisYear in phraseList:
		for phraseListSub in phraseListThisYear:
			tf.write("%s\n" % str(phraseListSub))
		tf.write("\n")

	tf.close()
	return

def isSubarray(small, big):
	windowSize = len(small)
	steps = len(big) - windowSize + 1
	for startingIndex in range(steps):
		testArray = big[startingIndex:startingIndex + windowSize]
		if np.array_equal(testArray, small):
			return True
	return False

def checkPrecisionRecall(Xdata, Ydata, Yprediction):
	Xdata = np.asarray(Xdata)
	Ydata = np.asarray(Ydata)
	Yprediction = np.asarray(Yprediction)
	maxXdata = np.amax(Xdata, axis = 1)
	actualDist = maxXdata < Ydata
	predictDist = maxXdata < Yprediction

	TP = np.count_nonzero(actualDist * predictDist)
	TN = np.count_nonzero(actualDist) - TP
	precision, recall, f1 = 0, 0, 0
	if np.count_nonzero(predictDist) != 0:
		precision = np.float(TP) / np.count_nonzero(predictDist)
	else:
		precision = 'NA'

	if np.count_nonzero(actualDist) != 0:
		recall = np.float(TP) / np.count_nonzero(actualDist)
	else:
		recall = 'NA'

	if precision != 'NA' and recall != 'NA' and (precision + recall) > 0:
		f1 = 2 * (precision * recall) / (precision + recall)
	return precision, recall, f1

def retrieveTrendingIndices(Xdata, Ydata, Yprediction):
	Xdata = np.asarray(Xdata)
	Ydata = np.asarray(Ydata)
	Yprediction = np.asarray(Yprediction)
	maxXdata = np.amax(Xdata, axis = 1)
	actualDist = maxXdata < Ydata
	predictDist = maxXdata < Yprediction

	actualDist = Ydata - maxXdata
	predictDist = Yprediction - maxXdata

	def filterNonZero(x): 
		if x <= 0:
			return 0
		return x
	actualDist = np.asarray([filterNonZero(x) for x in actualDist])
	predictDist = np.asarray([filterNonZero(x) for x in predictDist])

	TPdist = np.asarray(actualDist * predictDist)

	actualDistIndices = np.argpartition(actualDist, -20)[-20:]
	predictDistIndices = np.argpartition(predictDist, -20)[-20:]

	TPDistIndices = np.argpartition(TPdist, -20)[-20:]

	return actualDistIndices, predictDistIndices, TPDistIndices

def calcMRRMAPNDCG(actualIndices, predictIndices):
	# predictIndices = actualIndices[:]
	# print actualIndices
	# print predictIndices
	scores = np.asarray([float(1) / (i + 1) for i in range(len(actualIndices))])
	predictScores = np.asarray([0 for n in range(len(actualIndices))],dtype=float)
	num = len(actualIndices)
	DCG_GT = scores[0]

	for index in range(1, num):
		DCG_GT += (scores[index] / math.log((index + 1), 2))

	mask = actualIndices == predictIndices
	predictScores[mask] = scores[mask]
	# for actualIndex in actualIndices:
	# 	if actualIndex in predictIndices:
	# 		i, = np.where(predictIndices == actualIndex)
	# 		i = i[0]
	# 		j, = np.where(actualIndices == actualIndex)
	# 		j = j[0]
	# 		predictScores[i] = scores[j]
	# 		predictScores[i] = 1
	# 		print "actual score: %f" % scores[j]
	# 		print "predict score: %f" % predictScores[i]

	DCG_Pred = predictScores[0]
	for index in range(1, num):
		DCG_Pred += (predictScores[index] / math.log((index + 1), 2))

	nDCG = DCG_Pred / DCG_GT
	print "GT: %f" % DCG_GT
	print "Pr: %f" % DCG_Pred

	return nDCG

def scale(originalData):
	npData = np.array(originalData)
	currMin = npData.min()
	currMax = npData.max()
	result = (npData - currMin) / (currMax - currMin)
	return result

def scale2DArray(originalData):
	npData = np.array(originalData)
	result = []
	for currRow in npData:
		currMin = currRow.min()
		currMax = currRow.max()
		result.append((currRow - currMin) / (currMax - currMin))
	result = np.array(result)
	return result

if (__name__ == "__main__"):
	# fileName = "textrank040linear/TextrankSeries-040.txt"
	# fileName = "OccurrenceSeries.txt"
	# fileName = "MMR/data/TFIDFSeries.txt"
	# fileName = "MMR/data/TextrankSeriesNew-window4.txt"
	fileName = "data/soft-abs-nolast-ori-if-or-no.txt"
	aggregateCoefficient = 0.2
	phraseList, timeSeries = readTimeSeriesData(fileName)
	# timeSeries = aggregatePhraseScore(timeSeries, aggregateCoefficient)
	windowSize = 3
	testSize = 20
	numOfTopics = 10
	checkAllowance = 0

	# timeSeries = scale2DArray(np.array(timeSeries)).tolist()

	newXList, newYList, powerTerms, yearCover = splitData3(timeSeries, windowSize)
	newXList = np.asarray(newXList)
	newYList = np.asarray(newYList)

	print "finished splitting data"

	regression = linear_model.LinearRegression()
	phraseCount = len(phraseList)
	# meanSquareError = {}
	# varianceScore = {}
	# coefRegression = {}

	# RMSErrorTermListMap = {phrase:[0 for n in range(yearCover)] for phrase in phraseList}
	# RMSErrorTermList = [[0 for n in range(yearCover)] for i in range(phraseCount)]
	RMSErrorTerm = []
	DIFFTermListActual = [[0 for n in range(yearCover)] for i in range(phraseCount)]
	DIFFTermListPrediction = [[0 for n in range(yearCover)] for i in range(phraseCount)]
	precisionRecallList = [(0, 0, 0) for n in range(yearCover)]



	newXTrainWithIndices, newXTestWithIndices, newYTrain, newYTest = train_test_split(newXList, newYList, test_size = 0.2, random_state = 42)

	newXTrain = np.delete(newXTrainWithIndices, np.s_[0:2], axis = 1) # format of data: [year index, phrase index, scores]
	newXTest = np.delete(newXTestWithIndices, np.s_[0:2], axis = 1)

	newXTrainYear = newXTrainWithIndices[:, 0].astype(int)
	newXTestYear = newXTestWithIndices[:, 0].astype(int)
	newXTrainIndices = newXTrainWithIndices[:, 1].astype(int)
	newXTestIndices = newXTestWithIndices[:, 1].astype(int)

	newXTrainBkp = np.copy(newXTrain)
	newYTrainBkp = np.copy(newYTrain)

	newXListWithoutIndices = np.delete(newXList, np.s_[0:2], axis = 1)

	# print "length of training"
	# print len(newXTrain)
	# print "before statsmodel"



	# statsRegression = smapi.ols("data ~ x", data=dict(data=newYTrain, x=newXTrain)).fit()
	# print "after smapi"
	# influence = statsRegression.get_influence()
	# print "after influence"
	# resid_student = influence.resid_studentized_external
	# print "after resid student"
	# (cooks, p) = influence.cooks_distance
	# print "after cooks"
	# (dffits, p) = influence.dffits
	# print "after dffits"
	# leverage = influence.hat_matrix_diag
	# print "after leverages"

	# outlierIndices, = np.where(abs(resid_student) > 2)
	# outlierIndices = tuple(outlierIndices.tolist())



	# # sampleSize = len(newYTrain)
	# # featureSize = len(newXTrain[0])

	# # allCooksPoints, = np.where(abs(cooks) > (4 / sampleSize))
	# # allDFFPoints, = np.where(abs(dffits) > (2 * np.sqrt(float(featureSize) / sampleSize)))

	# # test = statsRegression.outlier_test()
	# # indicesToBeRemoved = tuple(test[test.icol(2) < 0.05].index.tolist())

	# # removingIndices = tuple(set(outlierIndices) | set(allDFFPoints))



	# removingIndices = tuple(set(outlierIndices))

	# newXTrainAfterDelete = np.delete(newXTrain, removingIndices, axis = 0)
	# newYTrainAfterDelete = np.delete(newYTrain, removingIndices, axis = 0)



	# newXTrainIndices = np.delete(newXTrainIndices, removingIndices, axis = 0)
	# newXTrainYear = np.delete(newXTrainYear, removingIndices, axis = 0)
	# newXTrainWithIndices = np.delete(newXTrainWithIndices, removingIndices, axis = 0)

	# print "finished removing outliers"



	"""
	This is where the LSTM comes in;
	the current implementation assumes single feature LSTM and all 3 time steps are consolidated into one feature

	"""
	lookback = len(newXTrain[0])
	XTrainLSTM = np.reshape(newXTrain, (newXTrain.shape[0], newXTrain.shape[1], 1))
	XTestLSTM = np.reshape(newXTest, (newXTest.shape[0], newXTest.shape[1], 1))
	LSTMModel = Sequential()

	LSTMModel.add(LSTM(4, input_shape=(lookback, 1)))
	LSTMModel.add(Dense(1))
	LSTMModel.compile(loss="mean_squared_error", optimizer = 'adam')
	LSTMModel.fit(XTrainLSTM, newYTrain, epochs = 70, batch_size = 10, verbose = 2)

	LSTMTrainPredict = LSTMModel.predict(XTrainLSTM)
	LSTMTestPredict = LSTMModel.predict(XTestLSTM)

	"""
	This is the end of LSTM computation
	"""

	newPredictedYTrain = LSTMTrainPredict.flatten()
	newPredictedYTest = LSTMTestPredict.flatten()


	"""
	This is the original regression part
	"""
	# regression.fit(newXTrain, newYTrain)
	# newPredictedYTrain = regression.predict(newXTrain)
	# newPredictedYTest = regression.predict(newXTest)

	"""
	End of regression part
	"""

	# print "regression coefficients:"
	# print regression.coef_

	newTrainingSize = len(newXTrain)
	newTestSize = len(newXTest)

	# DIFFTermListActualTrain = [[0 for n in range(yearCover)] for i in range(phraseCount)]
	# DIFFTermListActualTest = [[0 for n in range(yearCover)] for i in range(phraseCount)]
	# DIFFTermListPredictionTrain = [[0 for n in range(yearCover)] for i in range(phraseCount)]
	# DIFFTermListPredictionTest = [[0 for n in range(yearCover)] for i in range(phraseCount)]

	comparisonTermListActual = [[0 for n in range(yearCover)] for i in range(phraseCount)]
	comparisonTermListPredict = [[0 for n in range(yearCover)] for i in range(phraseCount)]
	maxTermListActual = [[0 for n in range(yearCover)] for i in range(phraseCount)]


	CVDIFFList = [0 for n in range(len(newYTrain))]

	for sampleIndex in range(newTrainingSize):
		trainingSampleYear = newXTrainYear[sampleIndex]
		trainingSamplePhraseIndex = newXTrainIndices[sampleIndex]
		trainingSamplePhrase = phraseList[int(trainingSamplePhraseIndex)]
		# RMSErrorTermListMap[trainingSamplePhrase][trainingSampleYear] = (newYTrain - newPredictedYTrain)[sampleIndex]
		# RMSErrorTermList[trainingSamplePhraseIndex][trainingSampleYear] = (newYTrain - newPredictedYTrain)[sampleIndex]
		DIFFTermListActual[trainingSamplePhraseIndex][trainingSampleYear] = newYTrain[sampleIndex] - np.max(newXTrain[sampleIndex])
		DIFFTermListPrediction[trainingSamplePhraseIndex][trainingSampleYear] = newPredictedYTrain[sampleIndex] - np.max(newXTrain[sampleIndex])
		maxTermListActual[trainingSamplePhraseIndex][trainingSampleYear] = np.max(newXTrain[sampleIndex])
		# DIFFTermListActualTrain[trainingSamplePhraseIndex][trainingSampleYear] = newYTrain[sampleIndex] - np.max(newXTrain[sampleIndex])
		# DIFFTermListPredictionTrain[trainingSamplePhraseIndex][trainingSampleYear] = newPredictedYTrain[sampleIndex] - np.max(newXTrain[sampleIndex])

	for sampleIndex in range(newTestSize):
		testSampleYear = newXTestYear[sampleIndex]
		testSamplePhraseIndex = newXTestIndices[sampleIndex]
		testSamplePhrase = phraseList[int(testSamplePhraseIndex)]
		# RMSErrorTermListMap[testSamplePhrase][testSampleYear] = (newYTest - newPredictedYTest)[sampleIndex]
		# RMSErrorTermList[testSamplePhraseIndex][testSampleYear] = (newYTest - newPredictedYTest)[sampleIndex]
		DIFFTermListActual[testSamplePhraseIndex][testSampleYear] = newYTest[sampleIndex] - np.max(newXTest[sampleIndex])
		DIFFTermListPrediction[testSamplePhraseIndex][testSampleYear] = newPredictedYTest[sampleIndex] - np.max(newXTest[sampleIndex])
		maxTermListActual[testSamplePhraseIndex][testSampleYear] = np.max(newXTest[sampleIndex])
		# DIFFTermListActualTest[testSamplePhraseIndex][testSampleYear] = newYTest[sampleIndex] - np.max(newXTest[sampleIndex])
		# DIFFTermListPredictionTest[testSamplePhraseIndex][testSampleYear] = newPredictedYTest[sampleIndex] - np.max(newXTest[sampleIndex])

	# RMSErrorTermMap = {phrase:np.sqrt(np.sum((np.asarray(RMSErrorTermListMap[phrase])**2)) / yearCover) for phrase in phraseList}
	# RMSErrorTerm = [np.sqrt(np.sum(np.array(a) ** 2) / yearCover) for a in RMSErrorTermList]
	DIFFTermListActual = np.asarray(DIFFTermListActual)
	DIFFTermListPrediction = np.asarray(DIFFTermListPrediction)
	maxTermListActual = np.asarray(maxTermListActual)
	# maxTermListActual = np.reshape(maxTermListActual, (maxTermListActual.shape[0], maxTermListActual.shape[1]))
	# DIFFTermListActualTrain = np.asarray(DIFFTermListActualTrain)
	# DIFFTermListActualTest = np.asarray(DIFFTermListActualTest)
	# DIFFTermListPredictionTrain = np.asarray(DIFFTermListPredictionTrain)
	# DIFFTermListPredictionTest = np.asarray(DIFFTermListPredictionTest)
	phraseList = np.asarray(phraseList)

	trendingPhrasesList = []
	trendingPhrasesListWithTrainTest = []
	newTrendingPhraseList = []
	nDCGList = []
	numberOfTrendingWords = 20

	weightDIFF = 0.8
	weightOrigin = 1 - weightDIFF

	comparisonTermListActual = weightOrigin * maxTermListActual + weightDIFF * DIFFTermListActual
	comparisonTermListPredict = weightOrigin * maxTermListActual + weightDIFF * DIFFTermListPrediction



	# kf = KFold(n_splits=5)
	# CVErrorsList = []
	# kfIterator = []
	# for kfTrain, kfTest in kf.split(newXListWithoutIndices):
	# 	kfIterator.append((kfTrain, kfTest))
	# for featureIndex in range(10):
	# 	KFoldSetBp = newXListWithoutIndices[:]
	# 	KFoldSet = []
	# 	currError = 0
	# 	for setIndex in range(len(KFoldSetBp)):
	# 		KFoldSet.append(addFeaturesKF(KFoldSetBp[setIndex].tolist(), featureIndex + 1))

	# 	KFoldSet = np.asarray(KFoldSet)
	# 	for (currTrain, currTest) in kfIterator:
	# 		kfXTrain, kfXTest, kfYTrain, kfYTest = KFoldSet[currTrain], KFoldSet[currTest], newYList[currTrain], newYList[currTest]
	# 		regression.fit(kfXTrain, kfYTrain)
	# 		kfYPredicted = regression.predict(kfXTest)
	# 		currdiff = kfYPredicted - kfYTest
	# 		currError += np.sum((currdiff ** 2))

	# 	CVErrorsList.append(np.sqrt(currError / float(len(newYList))))

	# print "Using KF for CV results:"
	# print CVErrorsList



	# CVPredicted = cross_val_predict(regression, newXListWithoutIndices, newYList, cv=5)
	# CVDIFFList = CVPredicted - newYList
	# CVRMSError = np.sqrt(np.sum(np.array([a ** 2 for a in CVDIFFList])) / float(len(newYList)))

	# print "Current CV RMS error:"
	# print CVRMSError



	# the TP, FP, TN should all come from the top 20 words (in the slide confirm this)
	for year in range(yearCover):
		currentYearActualDiff = DIFFTermListActual[:, year]
		currentYearPredictDiff = DIFFTermListPrediction[:, year]

		currentYearActualDiff = comparisonTermListActual[:, year]
		currentYearPredictDiff = comparisonTermListPredict[:, year]

		# TPdist = np.asarray(currentYearActualDiff + currentYearPredictDiff)

		actualDistIndices = np.argpartition(currentYearActualDiff, -numberOfTrendingWords)[-numberOfTrendingWords:]
		predictDistIndices = np.argpartition(currentYearPredictDiff, -numberOfTrendingWords)[-numberOfTrendingWords:]

		actualDistIndices = actualDistIndices[np.argsort(currentYearActualDiff[actualDistIndices])]
		predictDistIndices = predictDistIndices[np.argsort(currentYearPredictDiff[predictDistIndices])]

		actualDistIndices = actualDistIndices[::-1]
		predictDistIndices = predictDistIndices[::-1]


		# TPDistIndices = np.intersect1d(actualDistIndices, predictDistIndices)
		# TNDistIndices = np.setdiff1d(actualDistIndices, TPDistIndices)
		# FPDistIndices = np.setdiff1d(predictDistIndices, TPDistIndices)

		actualTrendingPhrases = phraseList[actualDistIndices]
		predictTrendingPhrases = phraseList[predictDistIndices]

		# TPTrendingPhrases = phraseList[TPDistIndices]
		# TNTrendingPhrases = phraseList[TNDistIndices]
		# FPTrendingPhrases = phraseList[FPDistIndices]

		totalList = []

		# predefine the longest word will be shorter than 20 characters

		for index in range(numberOfTrendingWords):
			currentActualPhrase = actualTrendingPhrases[index]
			currentPredictPhrase = predictTrendingPhrases[index]
			if currentActualPhrase in predictTrendingPhrases:
				numOfSpaces = 20 - len(currentActualPhrase)
				for i in range(numOfSpaces):
					currentActualPhrase += " "
				currentActualPhrase += "+ +"
			else:
				numOfSpaces = 20 - len(currentActualPhrase)
				for i in range(numOfSpaces):
					currentActualPhrase += " "
				currentActualPhrase += "+ -"

			if currentPredictPhrase in actualTrendingPhrases:
				numOfSpaces = 20 - len(currentPredictPhrase)
				for i in range(numOfSpaces):
					currentPredictPhrase += " "
				currentPredictPhrase += "+ +"
			else:
				numOfSpaces = 20 - len(currentPredictPhrase)
				for i in range(numOfSpaces):
					currentPredictPhrase += " "
				currentPredictPhrase += "- +"

			totalList.append(currentActualPhrase)
			totalList.append(currentPredictPhrase)

		currentNDCG = calcMRRMAPNDCG(actualDistIndices, predictDistIndices)
		currentNDCGWrite = str(2015 - yearCover - windowSize + year) + ": " + str(currentNDCG)
		nDCGList.append(currentNDCGWrite)

		newTrendingPhraseList.append("----Trending words " + str(2015 - yearCover - windowSize + year) + "--actual--predict--")
		newTrendingPhraseList += totalList
		newTrendingPhraseList.append("\n")

		# precision = float(len(TPTrendingPhrases)) / numberOfTrendingWords
		# precisionRecallList[year] = (precision, precision, precision)

		# maxLength = max(len(TPTrendingPhrases), len(TNTrendingPhrases), len(FPTrendingPhrases))
		# TPTrendingPhrases = np.append(TPTrendingPhrases, np.asarray(["     " for n in range(maxLength - len(TPTrendingPhrases))]))
		# TNTrendingPhrases = np.append(TNTrendingPhrases, np.asarray(["     " for n in range(maxLength - len(TNTrendingPhrases))]))
		# FPTrendingPhrases = np.append(FPTrendingPhrases, np.asarray(["     " for n in range(maxLength - len(FPTrendingPhrases))]))

		# trendingPhrases = np.array([TPTrendingPhrases, TNTrendingPhrases, FPTrendingPhrases]).T.tolist()
		# trendingPhrasesList.append("----Trending words " + str(2015 - yearCover - windowSize + year) + "-----")
		# trendingPhrasesList.append("-- TP (+ +) -- TN (+ -) -- FP (- +) --")
		# trendingPhrasesList += trendingPhrases
		# trendingPhrasesList.append("\n")

		# currActualDiffTrain = DIFFTermListActualTrain[:, year]
		# currActualDiffTest = DIFFTermListActualTest[:, year]
		# currPredDiffTrain = DIFFTermListPredictionTrain[:, year]
		# currPredDiffTest = DIFFTermListPredictionTest[:, year]
		# actualDistIndicesTrain = np.argpartition(currActualDiffTrain, -numberOfTrendingWords)[-numberOfTrendingWords:]
		# actualDistIndicesTest = np.argpartition(currActualDiffTest, -numberOfTrendingWords)[-numberOfTrendingWords:]
		# predDistIndicesTrain = np.argpartition(currPredDiffTrain, -numberOfTrendingWords)[-numberOfTrendingWords:]
		# predDistIndicesTest = np.argpartition(currPredDiffTest, -numberOfTrendingWords)[-numberOfTrendingWords:]
		# TPDistIndicesTrain = np.intersect1d(actualDistIndicesTrain, predDistIndicesTrain)
		# TNDistIndicesTrain = np.setdiff1d(actualDistIndicesTrain, TPDistIndicesTrain)
		# FPDistIndicesTrain = np.setdiff1d(predDistIndicesTrain, TPDistIndicesTrain)
		# TPDistIndicesTest = np.intersect1d(actualDistIndicesTest, predDistIndicesTest)
		# TNDistIndicesTest = np.setdiff1d(actualDistIndicesTest, TPDistIndicesTest)
		# FPDistIndicesTest = np.setdiff1d(predDistIndicesTest, TPDistIndicesTest)
		# TPPhraseTrain = phraseList[TPDistIndicesTrain]
		# TNPhraseTrain = phraseList[TNDistIndicesTrain]
		# FPPhraseTrain = phraseList[FPDistIndicesTrain]
		# TPPhraseTest = phraseList[TPDistIndicesTest]
		# TNPhraseTest = phraseList[TNDistIndicesTest]
		# FPPhraseTest = phraseList[FPDistIndicesTest]
		# maxLengthTrain = max(len(TPPhraseTrain), len(TNPhraseTrain), len(FPPhraseTrain))
		# maxLengthTest = max(len(TPPhraseTest), len(TNPhraseTest), len(FPPhraseTest))

		# TPPhraseTrain = np.append(TPPhraseTrain, np.asarray(["     " for n in range(maxLengthTrain - len(TPPhraseTrain))]))
		# TNPhraseTrain = np.append(TNPhraseTrain, np.asarray(["     " for n in range(maxLengthTrain - len(TNPhraseTrain))]))
		# FPPhraseTrain = np.append(FPPhraseTrain, np.asarray(["     " for n in range(maxLengthTrain - len(FPPhraseTrain))]))

		# TPPhraseTest = np.append(TPPhraseTest, np.asarray(["     " for n in range(maxLengthTest - len(TPPhraseTest))]))
		# TNPhraseTest = np.append(TNPhraseTest, np.asarray(["     " for n in range(maxLengthTest - len(TNPhraseTest))]))
		# FPPhraseTest = np.append(FPPhraseTest, np.asarray(["     " for n in range(maxLengthTest - len(FPPhraseTest))]))

		# trendingPhraseTrain = np.array([TPPhraseTrain, TNPhraseTrain, FPPhraseTrain]).T.tolist()
		# trendingPhrasesListWithTrainTest.append("----Train data trending words " + str(2015 - yearCover - windowSize + year) + "-----")
		# trendingPhrasesListWithTrainTest.append("-- TP (+ +) -- TN (+ -) -- FP (- +) --")
		# trendingPhrasesListWithTrainTest += trendingPhraseTrain
		# trendingPhraseTest = np.array([TPPhraseTest, TNPhraseTest, FPPhraseTest]).T.tolist()
		# trendingPhrasesListWithTrainTest.append("----Test data trending words " + str(2015 - yearCover - windowSize + year) + "-----")
		# trendingPhrasesListWithTrainTest.append("-- TP (+ +) -- TN (+ -) -- FP (- +) --")
		# trendingPhrasesListWithTrainTest += trendingPhraseTest
		# trendingPhrasesListWithTrainTest.append("\n")




	# RMSErrorTermMap = {phrase:np.sqrt(np.sum((np.asarray(RMSErrorTermListMap[phrase])**2)) / yearCover) for phrase in phraseList}
	writeList(nDCGList, "result/nDCG-soft-nolast-ori.txt")
	writeList(newTrendingPhraseList, "result/phrase-soft-abs-nolast-ori-lstm-if-or-no.txt")
	# writeList(RMSErrorTerm, "textrank040linear/RMSErrorListForm.txt")
	# writeList(trendingPhrasesList, "textrank040linear/trendingPhraseList.txt")
	# writeList(trendingPhrasesListWithTrainTest, "textrank040linear/trendingPhraseListWithTrainTest.txt")
	# writeList(precisionRecallList, "textrank040linear/precisionRecallList.txt")

	# writeScore(meanSquareError, "textrank040linear/meanLinear-" + str(100 - testSize) + "-" + str(windowSize) + ".txt")
	# writeScore(varianceScore, "textrank040linear/varianceLinear-" + str(100 - testSize) + "-" + str(windowSize) + ".txt")
	# writeScore(coefRegression, "textrank040linear/coef-" + str(100 - testSize) + "-" + str(windowSize) + ".txt")

	pass