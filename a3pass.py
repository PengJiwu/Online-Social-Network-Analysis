# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    t_list = []
    for i in range(0,len(movies.index)):
        s = movies.iloc[i]['genres']
        ts = tokenize_string(s)
        t_list.append(ts)
    movies['tokens'] = t_list
    return movies
    pass


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, which has been modified to include a column named 'features'.

    >>> movies = pd.DataFrame([[123,'Horror|Romance'], [456,'Sci-Fi|Horror']], columns = ['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies = featurize(movies)
    >>> movies.iloc[0]['features'].toarray()
    array([[ 0.     ,  0.30103,  0.     ]])
    >>> movies.iloc[1]['features'].toarray()
    array([[ 0.     ,  0.     ,  0.30103]])
    """
    ###TODO
    # store every the tokens column as one list. 
    #then transfer it to  dict, the key is the name of one term
    # value is the number of documents containing term i.
    
    global_dic = {}
    global_list = []
    for j in range(0,len(movies.index)):
        f_set = set(movies.iloc[j]['tokens'])
        global_list.extend(f_set)
    c0 = Counter(global_list)
    global_dic = dict(c0)
    
    #construct a dict, key is the name of one term, value is the index.(like vocab)
    keys = global_dic.keys()
    keys = sorted(keys)
    index = 0
    vocab = {}
    for k in keys:
        vocab[k] = index
        index += 1
    # for now, we know df(i), then cal tf(i)
    feat_list = []
    N = len(movies.index)
    for i in range(0,len(movies.index)):
        feat = movies.iloc[i]['tokens']
        c = Counter(feat)
        d_dic = dict(c)
        length = len(feat)
        max_term = c.most_common(1)
        max_k = max_term[0][1] / length
        col = []
        data = []
        scanner = d_dic.keys()
        scanner = sorted(scanner)
        for items in scanner:
            tf =  d_dic[items] / length
            data.append( tf / max_k * np.log10(N/global_dic[items]))
            col.append(vocab[items])
        data = np.array(data, dtype = 'float64')
        row = np.array([0 for x in range(len(col))], dtype = 'int64')
        cols = np.array(col,dtype = 'int64')
        matrix = csr_matrix((data,(row,cols)), shape = (1,len(vocab.keys())))
        feat_list.append(matrix)
        #movies[i]['features'] = matrix
    """
    for ele in feat_list:
        print(ele.toarray())
    """
    movies['features'] = feat_list
    return movies, vocab
    pass


def train_test_split(ratings):
    """
    DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similadrity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    >>> d1 = np.array([4,5,1])
    >>> r1 = np.array([0,0,0])
    >>> c1 = np.array([0,3,4])
    >>> d2 = np.array([2,4,5])
    >>> r2 = np.array([0,0,0])
    >>> c2 = np.array([3,4,5])
    >>> m1 = csr_matrix((d1,(r1,c1)), shape = (1,7))
    >>> m2 = csr_matrix((d2,(r2,c2)), shape = (1,7))
    >>> cosine_sim(m1,m2)
    """
    ###TODO
    ay = a.toarray()
    by = b.toarray()
    norm_a = 0
    norm_b = 0
    dot = 0
    for i in range(0, len(ay[0])):
        dot += (ay[0][i]) * (by[0][i])
        norm_a += (ay[0][i]) **2
        norm_b += (by[0][i]) **2
    norm_a = np.sqrt(norm_a)
    norm_b = np.sqrt(norm_b)

    if norm_a != 0 and norm_b != 0:
        cos_sim = dot / (norm_b * norm_a)
    else:
        cos_sim = 0
    return cos_sim
    pass


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    #get the userId and movieId in the test. store as two list.
    user_test = []
    movie_test = []
    for index, row in ratings_test.iterrows():
        user_test.append(row.userId)
        movie_test.append(row.movieId)
    result = []
    for j in range(0,len(user_test)):
        u = user_test[j]
        i = movie_test[j]
        #rated_movies store all the rating information of user u.(all the rating user u has made)
        rated_movies = ratings_train[ratings_train.userId == u]
        #for every test movie i, cal its similarity movies in rated_movies.
        test_feature = movies[movies.movieId == i].iloc[0]['features']
        predict = 0
        summ = 0
        rate = []
        for index,row in rated_movies.iterrows():
            if row.movieId != i:
                train_feature = movies[movies.movieId == row.movieId].iloc[0]['features']
                cosim = cosine_sim(test_feature, train_feature)
                r = row['rating']
                if cosim > 0:
                    #r = row['rating']
                    predict += cosim * r
                    summ += cosim
                else:
                    rate.append(r)

        if summ != 0:
            predict = predict / summ
        elif summ == 0:
            predict = np.mean(rate)
        result.append(predict)
    #print(result)
    return np.array(result)

    pass


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    file = open('train','w+')
    file.write(str(ratings_train))
    file.close()
    f2 = open('test','w+')
    f2.write(str(ratings_test))
    f2.close()
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    #print(ratings_test)
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
