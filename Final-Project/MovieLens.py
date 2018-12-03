
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import operator
from scipy import spatial
from math import log
from random import shuffle
from sklearn.model_selection import train_test_split
import string
from sklearn.svm import LinearSVC,SVC
from sklearn.multiclass import OneVsRestClassifier


# In[2]:


movie_data = pd.read_csv('data/movies.csv')
movies = movie_data['movieId'].unique().tolist()
print('Number of unique movies in the dataset: {}\n'.format(len(movies)))

genres = movie_data['genres'].unique().tolist()
unique_genres = set()
for genre_list in genres:
    sp = genre_list.split('|')
    for gen in sp:
        unique_genres.add(gen)
print('List of possible genres in the dataset:')
for genre in sorted(unique_genres)[:len(unique_genres)-1]:
    print(genre, end=', ')
print(sorted(unique_genres)[len(unique_genres)-1])
    
rating_data = pd.read_csv('data/ratings.csv')
unique_users = rating_data['userId'].unique().tolist()
print('\n\nNumber of users in the dataset: {}'.format(len(unique_users)))
print('Number of ratings in the dataset: {}'.format(len(rating_data['userId'].tolist())))


# In[3]:


movie_categories = dict()

id_genres = movie_data[['movieId','genres']].values
for pair in id_genres:
    movie_categories[pair[0]] = pair[1].split('|')

rating_movies = rating_data[['movieId']].values
category_counts = defaultdict(int)
for movie in rating_movies:
    for cat in movie_categories[movie[0]]:
        category_counts[cat] += 1
category_counts_list = list(category_counts.items())
category_counts_list.sort(key=operator.itemgetter(1))
category_counts_list.reverse()
print('Top 5 most watched movie categories are:')
top_categories = []
for i in range(5):
    top_categories.append(category_counts_list[i][0])
    print('{}'.format(category_counts_list[i][0]))


# In[4]:


movie_tags = pd.read_csv('data/tags.csv')
movie_tags.drop(columns=['timestamp'],inplace=True)
movie_tags.drop_duplicates(inplace=True)


# In[ ]:


movie_tag_counts = defaultdict(list)
movie_tags_list = list(movie_tags[['movieId','tag']].values)
translator = str.maketrans('','',string.punctuation)
for pair in movie_tags_list:
    if isinstance(pair[1],str):
        for tag in (pair[1].translate(translator)).lower().split():
            movie_tag_counts[pair[0]].append(tag)
def get_defaultdict_int():
    return defaultdict(int)


# In[ ]:


def get_defaultdict_float():
    return defaultdict(float)
category_tags = defaultdict(get_defaultdict_int)
for movie in movie_tag_counts:
    for category in movie_categories[movie]:
        for tag in movie_tag_counts[movie]:
            category_tags[category][tag] += 1
category_totals = dict()
for category in category_tags:
    total = 0
    for tag in category_tags[category]:
        total += category_tags[category][tag]
    category_totals[category] = total

category_tf_scores = defaultdict(get_defaultdict_float)
for category in category_tags:
    for tag in category_tags[category]:
        category_tf_scores[category][tag] = category_tags[category][tag] / category_totals[category]
unique_tags = set()

for category in category_tf_scores:
    for tag in category_tf_scores[category]:
        unique_tags.add(tag)
tag_idf_scores = defaultdict(float)

for tag in list(unique_tags):
    doc_count = 0
    for category in category_tf_scores:
        if tag in category_tf_scores[category]:
            doc_count += 1
    tag_idf_scores[tag] = log(len(category_tf_scores)/doc_count)
    
tag_tfidf_scores = defaultdict(get_defaultdict_float)
for category in category_tags:
    for tag in list(unique_tags):
        tag_tfidf_scores[category][tag] = category_tf_scores[category][tag] * tag_idf_scores[tag]

for cat in top_categories:
    tags = list(tag_tfidf_scores[cat].items())
    tags.sort(key=operator.itemgetter(1))
    tags.reverse()
    print(cat)
    print(tags[:10])


# In[ ]:


all_categories = [category[0] for category in category_counts_list]



check_words = []
for cat in top_categories:
    tags = list(tag_tfidf_scores[cat].items())
    tags.sort(key=operator.itemgetter(1))
    tags.reverse()
    for i in range(50):
        check_words.append(tags[i][0])

movie_tag_pairs = movie_tags[['movieId','tag']].values
train,test = train_test_split(movie_tag_pairs)
        
def get_features(data,train=True):
    X = []
    y = []
    genre_list = []
    for datum in data:
        feature = []
        feature.append(1)
        movie_id = datum[0]
        tags = (str(datum[1]).translate(translator)).lower()
        for word in check_words:
            feature.append(word in tags)
        genres = movie_categories[movie_id]
        if not train:
            X.append(feature)
            genre_list.append(genres)
        for genre in genres:
            if train:
                X.append(feature)
            y.append(all_categories.index(genre))
    if train:
        return X,y
    else:
        return X,genre_list

X_train, y_train = get_features(train)
X_test, genres = get_features(test,False)


# In[ ]:

print('Starting training')
ovr = OneVsRestClassifier(SVC(gamma='auto'))
ovr.fit(X_train,y_train)


# In[ ]:


correct = 0
predictions = ovr.predict(X_test)
for pred,genre_list in zip(predictions,genres):
    if all_categories[pred] in genre_list:
        correct += 1
print('Accuracy of tag SVM: {}'.format(correct/len(predictions)))


# In[ ]:


correct = 0
predictions = dict()
for pair in movie_tag_pairs:
    predictions[pair[0]] = top_categories[0]
for movie in predictions:
    if predictions[movie] in movie_categories[movie]:
        correct += 1
print('Accuracy of naive classifier: {}'.format(correct/len(predictions)))


# In[ ]:



if 'timestamp' in rating_data.columns:
    rating_data.drop(columns=['timestamp'],inplace=True)
ratings = rating_data.values

