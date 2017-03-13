
# coding: utf-8

# In[16]:

import pyspark
import string
import itertools

from html import unescape
from pyspark import SparkContext
from pyspark.mllib.feature import *
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import MulticlassMetrics
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *


# In[2]:

sc = SparkContext()


# In[17]:

training_file = 'data/train.csv'
test_file = 'data/test.csv'

hashingTF = HashingTF()
idf = IDF()


# In[18]:

def getStopWords(fileName):
    stopword_file = open(fileName, 'r')
    stopeord_string = stopword_file.read()
    stopword_file.close()

    stopword_list = stopeord_string.split('\n')
    stopword_dict = {}

    for word in stopword_list:
        stopword_dict[word] = ''

    return stopword_dict


# In[19]:

stopword_file = 'data/stopwords.txt'
stopwordDict = getStopWords(stopword_file)


# In[20]:

stem_porter = PorterStemmer()
#exclude_char = set(string.punctuation)
exclude_char = ['.', '?', '!', ',', ':' ]
APOSTROPHES = {"'s" : " is", "'re" : " are", "'ll" : "will", "n't" : "not"}


# In[21]:

def getRootWord(word):
    return stem_porter.stem(word)


# In[22]:

def removeSpecialChar(word):
    return ''.join([ch for ch in word if ch not in exclude_char])


# In[72]:

def cleanTweetText(original_tweet):
    tweet_text = []
    
    # Remove html escape characters and replace with their meaning
    tweet_html_str = unescape(original_tweet)

#     # Decode tweet to utf-8 format
    tweet_decoded = tweet_html_str.encode("ascii", "ignore").decode("utf8")

#     # Remove multiple repetition of a character in word
    tweet_str = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet_decoded))

#     # Convert all the apostrophes to standard lexicons
    words = []
    tokens = word_tokenize(tweet_str)
    words = [APOSTROPHES[word] if word in APOSTROPHES else word for word in tokens]
    
    for word in tweet_html_str.split(" "):
        if word in stopwordDict:
            continue
        elif word[0] == '@':
            tweet_text.append("AT_USER")
        elif word[0] == '#':
            tweet_text.append(str.lower(word[1:]))
        elif word[0:7] == 'http://' or word[0:8] == 'https://' or word[0:4] == 'www.':
            tweet_text.append("URL")
        elif str.isalpha(word):
            root_word = getRootWord((str.lower(word)))
            tweet_text.append(root_word[0])
        else:
            continue
    
    return tweet_text


# In[73]:

def getTweetLabelPoint(record):
    tweet_polarity = record.split(",", 1)[0].strip('"')
    tweet_str = record.split(",",5)[-1].strip("\n" " " '"')
    
    tweet_text_vector = hashingTF.transform(cleanTweetText(tweet_str))


    return LabeledPoint(tweet_polarity, tweet_text_vector)


# In[74]:

train_docs = sc.textFile(training_file).map(getTweetLabelPoint)


# In[75]:

test_docs = sc.textFile(test_file).map(getTweetLabelPoint)


# In[76]:

def computeAccuracy(predictionAndLabel, total_count):
    return 1.0 * predictionAndLabel.filter(lambda t: t[0] == t[1]).count() / total_count


# In[77]:

total_accuracy = 0
kfold = 1
for i in range(0,kfold):    
    # Split data aproximately into training (60%) and test (40%)
    training, test = train_docs.randomSplit([0.8, 0.2], seed=0)

    model = LogisticRegressionWithLBFGS.train(training,iterations=150, regType='l2')

    # Make prediction and test accuracy.
    predictionAndLabel = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))
    
    # Compute accuracy
    accuracy = computeAccuracy(predictionAndLabel, test.count()) 
    total_accuracy += accuracy
    
print(total_accuracy / kfold)


# In[78]:

total_accuracy = 0
kfold = 1
for i in range(0,kfold):    
    
    model = LogisticRegressionWithLBFGS.train(train_docs,iterations=150, regType='l2')

    # Make prediction and test accuracy.
    predictionAndLabel = test_docs.map(lambda lp: (float(model.predict(lp.features)), lp.label))

    # Evaluating the model on training data
#     labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
#     trainErr = labelsAndPreds.filter(lambda t: t[0] != t[1]).count() / float(parsedData.count())
#     print("Training Error = " + str(trainErr))

    # Compute accuracy
    accuracy = computeAccuracy(predictionAndLabel, test_docs.count()) 
    total_accuracy += accuracy
    
print(total_accuracy / kfold)


# In[79]:

# Make prediction and test accuracy.
fullTrainpredictionAndLabel = train_docs.map(lambda lp: (float(model.predict(lp.features)), lp.label))

# Compute accuracy
accuracy = computeAccuracy(fullTrainpredictionAndLabel, train_docs.count()) 
    
print(accuracy)


# In[37]:

# Instantiate metrics object
metrics = BinaryClassificationMetrics(predictionAndLabel)


# In[51]:

# Area under precision-recall curve
print("Area under PR = %s" % metrics.areaUnderPR)

# Area under ROC curve
print("Area under ROC = %s" % metrics.areaUnderROC)


# In[23]:

# Instantiate metrics object
metrics = MulticlassMetrics(predictionAndLabel)


# In[24]:

# Overall statistics
precision = metrics.precision()
recall = metrics.recall()
f1Score = metrics.fMeasure()
print("Summary Stats")
print("Precision = %s" % precision)
print("Recall = %s" % recall)
print("F1 Score = %s" % f1Score)


# In[54]:

# Statistics by class
labels = train_docs.map(lambda lp: lp.label).distinct().collect()
for label in sorted(labels):
    print("Class %s precision = %s" % (label, metrics.precision(label)))
    print("Class %s recall = %s" % (label, metrics.recall(label)))
    print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))


# In[55]:

# Weighted stats
print("Weighted recall = %s" % metrics.weightedRecall)
print("Weighted precision = %s" % metrics.weightedPrecision)
print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)


# In[ ]:



