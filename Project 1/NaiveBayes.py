
# coding: utf-8

# In[1]:

import nltk
import pyspark
import string

from pyspark import SparkContext
from pyspark.mllib.feature import *
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from nltk.stem.porter import *


# In[2]:

sc = SparkContext()


# In[3]:

training_file = 'data/train.csv'
test_file = 'data/test.csv'

hashingTF = HashingTF()
idf = IDF()


# In[4]:

def getStopWords(fileName):
    stopword_file = open(fileName, 'r')
    stopeord_string = stopword_file.read()
    stopword_file.close()

    stopword_list = stopeord_string.split('\n')
    stopword_dict = {}

    for word in stopword_list:
        stopword_dict[word] = ''

    return stopword_dict


# In[5]:

stopword_file = 'data/stopwords.txt'
stopwordDict = getStopWords(stopword_file)


# In[6]:

stem_porter = PorterStemmer()
#exclude_char = set(string.punctuation)
exclude_char = ['.', '?', '!', ',', ]


# In[7]:

def getRootWord(word):
    return stem_porter.stem(word)


# In[8]:

def removeSpecialChar(word):
    return ''.join([ch for ch in word if ch not in exclude_char])


# In[16]:

def cleanTweetText(tweet_str):
    tweet_text = []
    for word in tweet_str.split(" "):
        if word in stopwordDict:
            continue
        elif word[0] == '@':
            tweet_text.append("AT_USER")
        elif word[0] == '#':
            tweet_text.append(str.lower(word[1:]))
        elif word[0:7] == 'http://' or word[0:8] == 'https://' or word[0:4] == 'www.':
            tweet_text.append("URL")
        elif str.isalpha(word[0]):
            root_word = getRootWord((str.lower(word)))
            tweet_text.append(root_word)
        else:
            continue
    
    return tweet_text


# In[17]:

def getTweetLabelPoint(record):
    tweet_polarity = record.split(",", 1)[0].strip('"')
    tweet_str = record.split(",",5)[-1].strip("\n" " " '"')
    
    tweet_text_vector = hashingTF.transform(cleanTweetText(tweet_str))
#     tweet_text_vector.cache()
#     model = idf.fit(tweet_text_vector)
#     tfidf = model.transform(tweet_text_vector)

    return LabeledPoint(tweet_polarity, tweet_text_vector)


# In[18]:

train_docs = sc.textFile(training_file).map(getTweetLabelPoint)


# In[19]:

test_docs = sc.textFile(test_file).map(getTweetLabelPoint)


# In[20]:

def computeAccuracy(predictionAndLabel, total_count):
    accuracy_count = 0
    for x in predictionAndLabel.collect():
        if x[0] == x[1]: 
            accuracy_count += 1
    
    accuracy = 100.0 * accuracy_count / total_count
    return accuracy


# ###### Randamize split func

# In[42]:

total_accuracy = 0
kfold = 1
for i in range(0,kfold):    
    # Split data aproximately into training (60%) and test (40%)
    training, test = train_docs.randomSplit([0.8, 0.2], seed=0)

    model = NaiveBayes.train(training, 1.0)

    # Make prediction and test accuracy.
    predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
    
    accuracy = computeAccuracy(predictionAndLabel, test.count()) 
    total_accuracy += accuracy

print(total_accuracy / kfold)


# ###### Full Train-Test data

# In[ ]:

total_accuracy = 0
kfold = 1
for i in range(0,kfold):    
    # Train documents
    model = NaiveBayes.train(train_docs, 1.0)

    # Make prediction and test accuracy.
    predictionAndLabel = test_docs.map(lambda p: (model.predict(p.features), p.label))
    
    accuracy = computeAccuracy(predictionAndLabel, test_docs.count()) 
    total_accuracy += accuracy

print(total_accuracy / kfold)


# In[53]:

# Instantiate metrics object
metrics = BinaryClassificationMetrics(predictionAndLabel)


# In[56]:

# Area under precision-recall curve
print("Area under PR = %s" % metrics.areaUnderPR)

# Area under ROC curve
print("Area under ROC = %s" % metrics.areaUnderROC)


# In[60]:

# Make prediction and test accuracy.
predictionAndLabel = test_docs.map(lambda p: (model.predict(p.features), p.label))


# In[56]:

def computeAccuracy(predictionAndLabel, total_count):
    accuracy_count = 0
    for x in predictionAndLabel.collect():
        if x[0] == x[1]: 
            accuracy_count += 1
    
    accuracy = 100.0 * accuracy_count / total_count
    return accuracy


# In[16]:

accuracy = computeAccuracy(predictionAndLabel, test.count()) 
print(accuracy)


# In[25]:

accuracy_count = 0


# In[26]:

def computeAccuracyByMap(x):
    if x[0] == x[1]: 
        accuracy_count += 1


# In[27]:

predictionAndLabel.map(computeAccuracyByMap)
print(accuracy_count)


# In[14]:

print(train_docs.take(5))


# In[ ]:



