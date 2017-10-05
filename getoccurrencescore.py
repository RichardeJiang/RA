from xml.dom.minidom import parse
import xml.dom.minidom
import operator
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from itertools import takewhile, tee, izip, chain
import os
import math
import numpy as np

import networkx
import string
import sys

#added for solving the headache ascii encode/decode problem
reload(sys)  
sys.setdefaultencoding('utf8')

def writeMapToFile(listFile, fileName):
	theFile = open(fileName, 'w')
	for item in listFile:
		theFile.write("%s:" % str(item))
		for inner in listFile[item]:
			theFile.write("%s " % str(inner))
		theFile.seek(-1, os.SEEK_CUR)
		theFile.write("\n")
	theFile.close()
	return

def writeListToFile(listFile, fileName):
	theFile = open(fileName, 'w')
	for item in listFile:
		theFile.write("%s: " % str(item[0]))
		for inner in item[1]:
			theFile.write("%s, " % str(inner))
		theFile.seek(-2, os.SEEK_CUR)
		theFile.write("\n")
	theFile.close()
	return

def writeFreqToFile(listFile, fileName):
	theFile = open(fileName, "w")
	for item in listFile:
		theFile.write("%s: " % str(item[0]))
		theFile.write("%s\n" % str(item[1]))
	return

def XmlParsing(targetFile, targetTag):
	try:
		DOMTree = xml.dom.minidom.parse(targetFile)
	except xml.parsers.expat.ExpatError, e:
		print "The file causing the error is: ", fileName
		print "The detailed error is: %s" %e
	else:
		collection = DOMTree.documentElement

		resultList = collection.getElementsByTagName(targetTag)
		return resultList

	return "ERROR"

def tagNPFilter(sentence):
	tokens = nltk.word_tokenize(sentence)
	tagged = nltk.pos_tag(tokens)
	# NPgrammar = r"""NP:{<DT>?<JJ|NN|NNS>*<NN|NNS>}
	# ND:{<DT>?<NN|NNS><IN><DT>?<JJ|NN|NNS>*}"""
	NPgrammar = "NP:{<DT>?<JJ|NN|NNS>*<NN|NNS>}"
	#Problem: "a powerful computer with strong support from university" 
	#1, nested; 2, 'computer' is the keywords? or 'computer with support' is the keywords?
	cp = nltk.RegexpParser(NPgrammar)
	resultTree = cp.parse(tagged)   #result is of type nltk.tree.Tree
	result = ""
	stemmer = SnowballStemmer("english")
	for node in resultTree:
		if (type(node) == nltk.tree.Tree):
			#result += ''.join(item[0] for item in node.leaves()) #connect every words
			#result += stemmer.stem(node.leaves()[len(node.leaves()) - 1][0]) #use just the last NN

			if node[0][1] == 'DT':
				node.remove(node[0])  #remove the determiners
			currNounPhrase = ''.join(stemmer.stem(item[0]) for item in node.leaves())
			result += currNounPhrase

			multiplyTimes = len(node.leaves())
			for i in range(multiplyTimes - 1):
				result += " "
				result += currNounPhrase

			# if len(node.leaves()) == 1:
			# 	pass
			# else:
			# 	multiplyTimes = len(node)
			# 	result += ' '
			# 	result += currNounPhrase #double noun phrases to increase the weight

			### The following part assumes nested grammar can be supported ###
			### which turns out to be false, so use the previous selction instead ###
			# if (node.label() == 'NP'):   # NN phrases
			# 	result += node.leaves()[len(node.leaves()) - 1][0]
			# else:    # IN phrases
			# 	if (node[0][1] == 'NN' or node[0][1] == 'NNS'):    # the first element is NN
			# 		result += node[0][0]
			# 	else:    # the first element is DT
			# 		result += node[1][0]
			### End of wasted part ###

		else:
			result += stemmer.stem(node[0])
		result += " "
	return result

def generateNPTextList(text):

	stop_words = set(nltk.corpus.stopwords.words('english'))

	result = []
	tokens = nltk.word_tokenize(text)
	tagged = nltk.pos_tag(tokens)
	# NPGrammar = "NP:{<DT>?<JJ|NN|NNS>*<NN|NNS>}"
	NPGrammar = "NP:{<JJ|NN|NNS>*<NN|NNS>}"
	cp = nltk.RegexpParser(NPGrammar)
	resultTree = cp.parse(tagged)
	stemmer = SnowballStemmer("english")
	for node in resultTree:
		if (type(node) == nltk.tree.Tree):
			# if node[0][1] == 'DT':
			# 	node.remove(node[0])
			if node.leaves()[-1][0].lower() not in stop_words or len(node.leaves()) >= 2:
				currNNs = [stemmer.stem(item[0]) for item in node.leaves()]
				currNPs = []
				for index, NN in enumerate(currNNs):
					currNPs.append(''.join(item for item in currNNs[index:]))
					for replicateTimes in range(len(currNNs) - index - 1):
						currNPs.append(''.join(item for item in currNNs[index:]))

				result.append(currNPs)
		
		#Remove the following else clause to ignore all single terms
		# else:
		# 	if node[0].lower() not in stop_words:
		# 		result.append([stemmer.stem(node[0])])

	return result

def getRippleScores(wordRanks, numOfWordsWanted, graph):
	'''the ripple smooths in 3 folds: 0.2, 0.6, 0.8'''
	# word_ranks = {word_rank[0]: word_rank[1]
	# 	for word_rank in sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[:n_keywords]}
	# keywords = set(word_ranks.keys())
	result = {}
	keywordList = []
	ripple = [0.36, 0.6]
	for index in range(numOfWordsWanted):
		if len(wordRanks) > 0:
			currKeyword = sorted(wordRanks.iteritems(), key=lambda x: x[1], reverse=True)[0][0]
			currKeywordRank = sorted(wordRanks.iteritems(), key=lambda x: x[1], reverse=True)[0][1]
			keywordList.append(currKeyword)
			result[currKeyword] = currKeywordRank

			currOldValue = wordRanks[currKeyword]
			wordRanks[currKeyword] = (0.2 * currOldValue)
			openSet = [currKeyword]
			closeSet = []
			for iterIndex in range(2):
				currRipple = ripple[iterIndex]
				temp = []
				for ele in openSet:
					affectedEdges = graph.edges(ele)
					for affectedEdge in affectedEdges:
						node = affectedEdge[1]
						if node not in closeSet:
							oldValue = wordRanks[node]
							wordRanks[node] = (oldValue * currRipple)
							temp.append(node)

				closeSet += openSet
				openSet = list(temp)

	return result

def keyWordFilter(article, keywords, filterTags):
	# the idea is to use citation, reference, keyword list to find software engineering related articles
	for filterTag in filterTags:
		tagContents = article.getElementsByTagName(filterTag)

		for tagContent in tagContents:
			for keyword in keywords:
				keywordDash = '-'.join(keyword.split(' '))
				if ((keyword in tagContent.childNodes[0].data.lower()) 
					or (keywordDash in tagContent.childNodes[0].data.lower())):
					return True
	
	return False

def extract_candidate_words(text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
	stop_words = set(nltk.corpus.stopwords.words('english'))
	tagged_words = nltk.pos_tag(nltk.word_tokenize(text))
	candidates = [word.lower() for word, tag in tagged_words
		if tag in good_tags and word.lower() not in stop_words and len(word) > 1]
	return candidates

def getKeyphraseByTextRankFromNP(textList, n_keywords=0.7, n_windowSize=3):
	graph = networkx.Graph()
	graph.add_nodes_from(set([NN for NNList in textList for NN in NNList]))
	listLength = len(textList)
	for i in range(0, n_windowSize - 1):
		for textIndex, texts in enumerate(textList):
			if textIndex + i + 1 < listLength:
				neighboringTexts = textList[textIndex + i + 1]
				# If one of the words is a single term, link only it with its neighbor's full NP
				if len(texts) == 1 or len(neighboringTexts) == 1:
					# graph.add_edge(texts[0], neighboringTexts[0], weight = 1)
					pass
				else:
					temptexts = list(texts)
					tempneighboringtexts = list(neighboringTexts)
					temptexts.pop()
					tempneighboringtexts.pop()
					# graph.add_edges_from([(texta, textb) for texta in temptexts for textb in tempneighboringtexts])
					edges = [(texta, textb) for texta in temptexts for textb in tempneighboringtexts]
					for edge in edges:
						if graph.has_edge(edge[0], edge[1]):
							graph[edge[0]][edge[1]]['weight'] += 1
						else:
							graph.add_edge(edge[0], edge[1], weight = 1)
						# if edge in graph.edges() or edge[::-1] in graph.edges():

					# graph.add_edges_from([(texta, textb) for texta in texts for textb in neighboringTexts])
					print 'original texts: ', temptexts
					print 'neighboring texts: ', tempneighboringtexts

	ranks = networkx.pagerank(graph)
	if 0 < n_keywords < 1:
		n_keywords = int(round(listLength * n_keywords))


	# MMRKeywordList = []
	# word_ranks = {}
	# for index in range(n_keywords):
	# 	if len(sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)) > 0:
	# 		currKeyword = sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[0][0]
	# 		currKeywordRank = sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[0][1]
	# 		MMRKeywordList.append(currKeyword)
	# 		word_ranks[currKeyword] = currKeywordRank
	# 		graph.remove_node(currKeyword)
	# 		ranks = networkx.pagerank(graph)



	# word_ranks = getRippleScores(ranks, n_keywords, graph)
	# keywords = set(word_ranks.keys())

	# print "In TextRank: ", keywords

	word_ranks = {word_rank[0]: word_rank[1]
		for word_rank in sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[:n_keywords]}
	keywords = set(word_ranks.keys())


	# keywords = set(MMRKeywordList)

	return word_ranks

def getKeyphraseByTextRank(text, n_keywords=0.4, n_windowSize=4, n_cooccurSize=3):
	words = [word.lower()
		for word in nltk.word_tokenize(text)
		if len(word) > 1]
	
	candidates = extract_candidate_words(text)
	# print candidates
	graph = networkx.Graph()
	graph.add_nodes_from(set(candidates))
	
	for i in range(0, n_windowSize-1):
		def pairwise(iterable):
			a, b = tee(iterable)
			next(b, None)
			for j in range(0, i):
				next(b, None)
			return izip(a, b)
		for w1, w2 in pairwise(candidates):
			if w2:
				graph.add_edge(*sorted([w1, w2]))

	ranks = networkx.pagerank(graph)
	if 0 < n_keywords < 1:
		n_keywords = int(round(len(candidates) * n_keywords))


	MMRKeywordList = []
	word_ranks = {}
	for index in range(n_keywords):
		if len(sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)) > 0:
			currKeyword = sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[0][0]
			currKeywordRank = sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[0][1]
			MMRKeywordList.append(currKeyword)
			word_ranks[currKeyword] = currKeywordRank
			graph.remove_node(currKeyword)
			ranks = networkx.pagerank(graph)



	# word_ranks = getRippleScores(ranks, n_keywords, graph)
	# keywords = set(word_ranks.keys())



	# word_ranks = {word_rank[0]: word_rank[1]
	# 	for word_rank in sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[:n_keywords]}
	# keywords = set(word_ranks.keys())


	keywords = set(MMRKeywordList)

	keyphrases = {}
	j = 0
	for i, word in enumerate(words):
		if i<j:
			continue
		if word in keywords:
			kp_words = list(takewhile(lambda x: x in keywords, words[i:i+n_cooccurSize]))
			avg_pagerank = sum(word_ranks[w] for w in kp_words) / float(len(kp_words))
			keyphrases[' '.join(kp_words)] = avg_pagerank
			if len(kp_words) > 1:
				for kpWord in kp_words:
					keyphrases[kpWord] = word_ranks[kpWord]

			j = i + len(kp_words)

	results = [(ele[0], ele[1]) for ele in sorted(keyphrases.iteritems(), key=lambda x: x[1], reverse=True)]
	# results = self.duplicateHigherRankingTerms(results)
	# targetTagSet = ['NN', 'NNS', 'NNP', 'NNPS']
	# for result in results:
	# 	tempSet = self.removeDuplicates(result.split())
	# 	if nltk.pos_tag(nltk.word_tokenize(tempSet[-1]))[0][1] not in targetTagSet:
	# 		results.remove(result)
	# 	else:
	# 		newPhrase = ''.join(stemmer.stem(wordEle) for wordEle in tempSet)
	# 		results[results.index(result)] = newPhrase
	# results = self.removeDuplicates(results)
	# if (len(results) > 200):
	# 	results = results[:len(results) * 0.25]
	# return ' '.join(results)
	#results = {item[0]:item[1] for item in results}
	return duplicateHigherRankingTerms(results)

def duplicateHigherRankingTerms(rawList): # This function actually is stemming the word in each phrase
										  # Nothing to do with the duplciation of higher ranking terms
	rawList = removeDuplicates(rawList)
	if len(rawList) < 1:
		return ""
	baseFreq = float(rawList[-1][1]) # Unused var
	result = []

	phraseScoreMap = {}
	targetTagSet = ['NN', 'NNS', 'NNP', 'NNPS']
	stemmer = SnowballStemmer("english")
	for ele in rawList:
		tempSet = removeDuplicates(ele[0].split())
		# Only consider the noun phrases
		if nltk.pos_tag(nltk.word_tokenize(tempSet[-1]))[0][1] not in targetTagSet:
			pass
		else:
			newPhrase = ''.join(stemmer.stem(wordEle) for wordEle in tempSet)
			result.append((newPhrase, ele[1]))

	# if (len(result) > 300):
	# 	result = result[:len(result) / 2]
	

	# result = result[:150]

	phraseScoreMap = {item[0]:item[1] for item in result}

	return phraseScoreMap

def removeDuplicates(seq):
	seen = set()
	seen_add = seen.add
	return [x for x in seq if not (x in seen or seen_add(x))]

def getInfluentialPhraseScoreSeries(phraseScoreMapSeries):
	upperBound = 0.5
	lowerBound = 0.05
	yearCover = len(phraseScoreMapSeries)
	startingTemp = phraseScoreMapSeries[0]
	result = {key:[0 for n in range(yearCover)] for key in startingTemp}
	for index in range(yearCover):
		phraseScoreMap = phraseScoreMapSeries[index]
		for phrase in phraseScoreMap:
			if not result.has_key(phrase):
				result[phrase] = [0 for n in range(yearCover)]
			result[phrase][index] = phraseScoreMap[phrase]

	finalResult = {}
	for phrase in result:
		checkTimeSeries = result[phrase]
		zeroYears = np.asarray(checkTimeSeries)
		# if (zeroYears == 0).sum() >= (lowerBound * yearCover) and (zeroYears == 0).sum() <= (upperBound * yearCover):
		if (zeroYears == 0).sum() <= (upperBound * yearCover):
			finalResult[phrase] = checkTimeSeries
	return finalResult

if (__name__ == '__main__'):
	fileList = os.listdir('.')
	targetList = []
	for fileName in fileList:
		if fileName.endswith('.xml'):
			targetList.append(fileName)

	scoreList = {}
	authorIdMap = {}
	count = 0
	flag9000_1 = False
	flag9000_2 = False

	predefinedNumberOfVIPAuthors = 300
	predefinedNumberOfVIPPhrases = 300
	commonPhrasesRecognitionCriteria = int(0.25 * predefinedNumberOfVIPAuthors)
	keywords = ["software engineering", "software and its engineering"]
	# keywords = ["database", "storage"]
	# keywords = ["www", "search engine", "world wide web", "information retrieval", "internet"]
	filterList = ["kw", "ref_text", "cited_by_text", "concept_desc", "subtitle"]


	vocabOccurrenceMapSeries = [{} for n in range(70)]
	vocabTextRankMapSeries = [{} for n in range(70)]
	baseYear = 1950

	count = 0
	for target in targetList:
		articleList = XmlParsing(target, "article_rec")
		if articleList == "ERROR":
			continue

		# This part is to check whether all files have been iterated through
		count += 1
		if count > 9000:
			flag9000_2 = True

		print 'Currently processing: %d/9719' % count

		for article in articleList:

			if keyWordFilter(article, keywords, filterList):

				timeStamp = int(article.getElementsByTagName("article_publication_date").item(0).childNodes[0].data.split("-")[2])
				offset = timeStamp - baseYear
				vocabOccurrenceMap = vocabOccurrenceMapSeries[offset]
				vocabTextRankMap = vocabTextRankMapSeries[offset]


				abstract = article.getElementsByTagName("par")
				# abstract = article.getElementsByTagName("ft_body")
				
				if len(abstract) > 0 and len(abstract.item(0).childNodes) > 0:
					abstract = abstract.item(0).childNodes[0].data
					abstract = re.sub(r'<.*?>', "", abstract)
					abstract = re.sub(r'\"', "", abstract)
					abstract = str(abstract.encode('utf-8')).translate(None, string.punctuation)
					abstract = ''.join([i for i in abstract if not i.isdigit()])

					# paragraph = tagNPFilter(abstract)
					# phraseList = paragraph.split()
					# for identifiedPhrase in phraseList:
					# 	if not vocabOccurrenceMap.has_key(identifiedPhrase):
					# 		vocabOccurrenceMap[identifiedPhrase] = 1
					# 	else:
					# 		vocabOccurrenceMap[identifiedPhrase] += 1

					currTextList = generateNPTextList(abstract)
					currPhraseScoreMap = getKeyphraseByTextRankFromNP(currTextList)
					for currPhrase in currPhraseScoreMap:
						if not vocabTextRankMap.has_key(currPhrase):
							vocabTextRankMap[currPhrase] = currPhraseScoreMap[currPhrase]
						else:
							vocabTextRankMap[currPhrase] += currPhraseScoreMap[currPhrase]
					
	
	# finalVocabOccurrenceMapSeriesMap = getInfluentialPhraseScoreSeries([ele for ele in vocabOccurrenceMapSeries if bool(ele)])
	finalVocabTextrankMapSeriesMap = getInfluentialPhraseScoreSeries([ele for ele in vocabTextRankMapSeries if bool(ele)])

	writeMapToFile(finalVocabTextrankMapSeriesMap, 'c/soft-abs-nolast-ori.txt')
	# writeMapToFile(finalVocabOccurrenceMapSeriesMap, 'b/OccurrenceSeriesNew.txt')

	# if flag9000_1:
	# 	print "1st 9000 has been reached"

	# if flag9000_2:
	# 	print "2nd 9000 has been reached"

	print 'End processing!'

	pass