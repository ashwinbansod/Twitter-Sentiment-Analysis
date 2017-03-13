
# coding: utf-8

# ###### Import statements

# In[102]:

import sys
import csv
import nltk
import pyspark

from pyspark import SparkContext
from pyspark.mllib.feature import *
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel

from nltk.metrics import edit_distance
from nltk.stem.porter import *


# ###### Get spark contex variable

# In[7]:

sc = SparkContext()


# ###### Function to read stopwords file and return stopword dict

# In[2]:

def getStopWords(fileName):
    stopword_file = open(fileName, 'r')
    stopeord_string = stopword_file.read()
    stopword_file.close()

    stopword_list = stopeord_string.split('\n')
    stopword_dict = {}

    for word in stopword_list:
        stopword_dict[word] = ''

    return stopword_dict


# ###### Read stopwords file

# In[4]:

stopword_file = 'data/stopwords.txt'
stopword_dict = getStopWords(stopword_file)


# Use porter stemer to get root word of the tweet word

# In[17]:

stem_porter = PorterStemmer()


# In[18]:

def getRootWord(word):
    return stem_porter.stem(word)


# In[ ]:




# In[19]:

def getRootWordList(tweet_text):
    tweet_text_list = []
    for word in tweet_text.split():
        
        tweet_text_list.append(stem_porter.stem(word))
    
    return tweet_text_list


# File name of train.csv file to read

# In[78]:

csv_filename = 'data/train_sample.csv'


# In[17]:

csv_file = open(csv_filename, 'r')
csv_file_reader = csv.reader(csv_file)

doc_tweet_text_list = []

for tweet in csv_file_reader:
    tweet_text = tweet[5:]
    tweet_polarity = tweet[1]
    
    doc_tweet_text_list.append(getRootWordList(tweet_text))

doc_tweet_text_list
         


# In[ ]:




# In[97]:

def getTweetLabelPoint(record):
    tweet_text = record.split(",")[5:].strip('"')
    tweet_polarity = record.split(",")[0].strip('"')
    tweet_text_vector = doc_hashingTF.transform(tweet_text.split(" "))
    
    return LabeledPoint(tweet_polarity, tweet_text_vector)


# In[69]:

def getTweetPolarity(record):
    tweet_polarity = record.split(",")[0].strip('"')
    return tweet_polarity


# In[118]:

documents = sc.textFile(csv_filename).map(lambda line: line.strip("\n").split(","))
# polarity = sc.textFile(csv_filename).map(getTweetPolarity)

tweet_data = []
doc_hashingTF = HashingTF()

document_data = documents.collect()

for doc in document_data:
    # print(doc[1])
    tweet_polarity = doc[0].strip('"')
    tweet_text_vector = doc_hashingTF.transform(doc[5].strip('"').split(" "))
    tweet_data.append(LabeledPoint(tweet_polarity, tweet_text_vector))



# Split data aproximately into training (60%) and test (40%)
training, test = sc.parallelize(tweet_data).randomSplit([0.6, 0.4], seed=0)


# doc_tf = hashingTF.transform(documents)
# doc_tf.cache()
# document_idf = IDF().fit(doc_tf)
# document_tfidf = document_idf.transform(doc_tf)

# polarity_hashingTF = HashingTF()
# polarity_tf = polarity_hashingTF.transform(polarity)
# polarity_tf.cache()
# polarity_idf = IDF().fit(polarity_tf)
# polarity_tfidf = polarity_idf.transform(polarity_tf)



# In[119]:

model = NaiveBayes.train(training, 1.0)


# In[121]:

# Make prediction and test accuracy.
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))


# In[130]:

accuracy = 1.0 * predictionAndLabel.map(lambda x, v: x == v).count() / test.count()


# In[128]:

predictionAndLabel.collect()


# In[ ]:



