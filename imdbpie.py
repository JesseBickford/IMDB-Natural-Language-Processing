from imdbpie import Imdb
import os
import subprocess
import collections
import re
import csv
import json

import pandas as pd
import numpy as np
import scipy

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import psycopg2
import requests
import nltk

import urllib
from bs4 import BeautifulSoup
import nltk

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


#1. Connect to the imdbpie API
imdb = Imdb()
imdb = Imdb(anonymize = True)

#2. Query the top 250 rated movies in the database
imdb.top_250()

#3. Put the information into a dataframe, then keep only relevant columns
data = pd.DataFrame(imdb.top_250())
data.head()

data.drop('can_rate', axis=1, inplace=True)
data.drop('image', axis=1, inplace=True)
data.drop('type', axis=1, inplace=True)

#4. Select only the top 100 movies
data = data.iloc[0:100]

#change the column name tconst to movie_id
data.rename(columns={'tconst': 'movie_id'}, inplace=True)
data.head()


#5. Get the genres and runtime for each movie and add to the dataframe

idlist = data.movie_id.values #setup a movie id list to cycle through

rungenre = []
def get_genre_runtime(movie):
    for i in idlist:
        biglist = []
        title = imdb.get_title_by_id(i)
        runtime = title.runtime
        genre = title.genres
        biglist.append(runtime)  #here we append the two variables
        biglist.append(genre)
        rungenre.append(biglist) #here we append those lists to make a list of lists


get_genre_runtime(idlist)

#set the list of lists to be a dataframe
df = pd.DataFrame(rungenre)
df.rename(columns={0: 'runtime', 1: 'genre'}, inplace=True)
df

data = pd.concat([data,df],axis=1)
data.head()

#export the data to csv
data.to_csv('moviedata.csv', encoding='utf-8')

#now we will import the same data so that we don't need to continually query the api in case we have to restart our kernel
data = pd.read_csv('~/DSI/week-7/Project-6/moviedata.csv')
data.head()

#####Part 2
#1. Get the reviews of the top 100 movies

#this is our list of movies according to their id
idlist

#2. Get ratings and reviews
#create an empty list that will take all of the reviews
allreviews = []
#this function will loop through each movie and get its reviews, then store them in the allreviews list
def get_reviews(movie):
    for movienum in idlist:          #the first for loop, for each movie in the list, reviews are set for each movie and max results to 10,000 incase there are that many reviews
        reviews = imdb.get_title_reviews(movienum, max_results=10000)
        for review in reviews:       #the second for loop, for each review in the above loop, create and append to a list the movienumber, rating, and review text.
            this_review = []
            this_review.append(movienum)
            this_review.append(review.rating)
            this_review.append(review.text)
            allreviews.append(this_review)   #send this list back to the allreviews list, which will be a list of lists

#call the function now to go through the idlist, aka each movie
get_reviews(idlist)

#set the allreviews list to be a dataframe called reviewsdf
reviewsdf = pd.DataFrame(allreviews)
#save it to a csv with encoding to get rid of errors
reviewsdf.to_csv('reviewsdf.csv', encoding= 'utf-8')
#now we will import the same data so that we don't need to continually query the api in case we have to restart our kernel
reviewsdf = pd.read_csv('/Users/Starshine/DSI/Week-7/Project-6/reviewsdf.csv')

reviewsdf.head(3)

#change the column names to something more identifiable
reviewsdf.columns = ['review_number', 'movie_id', 'rating', 'review']

#3. Remove non-alpha numeric characters from the reviews
#use regex with the pattern which means not alphanumeric/whitespace
reviewsdf['review'] = reviewsdf['review'].str.replace('[^\w\s]', '')

#replace the '\n' that appears in some reviews too for a better reading experience
reviewsdf['review'] = reviewsdf['review'].str.replace('\n', ' ')
reviewsdf.review

#check to see if there is missing data in our reviews dataframe
reviewsdf.isnull().sum()
#drop the 17,000 reviews that have missing score data; this line of code drops all rows that have any NaN's
reviewsdf.dropna(axis=0, how='any', inplace = True)

#4. Calculate the top 200 ngrams from the user reviews
#initialize the TfidfVectorizer method with given parameters, we want the top 200 features
vectorizer = TfidfVectorizer(ngram_range = (1,2), stop_words = 'english', binary = False, max_features = 200)
#fit the vectorizer on our reviews from the reviewsdf and saved it as 'vect'
vect = vectorizer.fit_transform(reviewsdf.review)


vect1 = vectorizer.fit(reviewsdf.review.values)

reviews_vect = pd.DataFrame(vectorizer.transform(reviewsdf.review.values).todense(),
                    columns=vectorizer.get_feature_names(),
                    index=reviewsdf.index.values)
reviews_vect.head(3)

reviewsdf.head(3)

bigreviewsdf = pd.concat([reviewsdf, reviews_vect], axis=1)
bigreviewsdf.head(3)





#define the inverse document frequency
idf = vectorizer.idf_
#make a dicitonary for easy reading of the feature names and their idf scores
vectdict = dict(zip(vectorizer.get_feature_names(), idf))
#sort that dictionary to see smallest to largest idf
sorted([(key,value) for (value,key) in vectdict.items()])

#####Part 4
#1. Rename any columns you think should be renamed for clarity
reviewsdf.head(3)
data.head()


plt.scatter(data.num_votes, data.rating)
plt.xlabel('Number of votes')
plt.ylabel('Rating')

plt.scatter(data.year, data.rating)
plt.xlabel('Year')
plt.ylabel('Rating')




#####Part 5
#1. Setup your X and y variables
# X will be your TfidfVectorizer'ed reviews, y will be the ratings
X = bigreviewsdf.iloc[:, 4:205]
y = bigreviewsdf.rating

#2. Prepare the X and y matrices and preprocess
X_train, X_test, y_train, y_test = train_test_split(X, y)
dct = DecisionTreeClassifier()
dct.fit(X_train, y_train)

y_pred = dct.predict(X_test)
print metrics.accuracy_score(y_test, y_pred)

#now try with DecisionTreeRegressor
dcr = DecisionTreeRegressor()
dcr.fit(X_train, y_train)
y_pred_dcr = dcr.predict(X_test)
print metrics.r2_score(y_test, y_pred_dcr)


#####Part 7
#Bagging and Boosting: Random Forests, Extra Trees, and AdaBoost
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor

rfr = RandomForestRegressor()
etr = ExtraTreesRegressor()
abr = AdaBoostRegressor()

rfr.fit(X_train, y_train)
y_pred_rfr = rfr.predict(X_test)
print metrics.r2_score(y_test, y_pred_rfr)


etr.fit(X_train, y_train)
y_pred_etr = etr.predict(X_test)
print metrics.r2_score(y_test, y_pred_etr)

abr.fit(X_train, y_train)
y_pred_abr = abr.predict(X_test)
print metrics.r2_score(y_test, y_pred_abr)

#The adaboost regressor peforms best
