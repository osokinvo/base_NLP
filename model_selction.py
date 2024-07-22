import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def model_selection_word_exist(metod_df, unic_words):
    
    w2v = Word2Vec(sentences=[unic_words], min_count=1)

    X = np.zeros((len(metod_df.word), w2v.wv.vector_size))
    for i, word in enumerate(metod_df.word):
        X[i,:w2v.wv.vector_size] = w2v.wv[word]
    Y = pd.cut((-metod_df['Negative'] + metod_df['Positive']) * 0.5 **metod_df['Neutral'], bins=[-2, -0.33, 0.33,2], labels=[-1, 0, 1])
    train_X, test_X, train_Y, test_Y = train_test_split(
        X, Y, test_size=0.2, random_state=42)
    clf = LogisticRegression(solver='saga')
    param_grid = {
        'C': np.arange(1, 5)
    }

    search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=5, refit=True, scoring='accuracy')
    search.fit(train_X, train_Y)
    clf = LogisticRegression(C=search.best_params_['C'])
    clf.fit(train_X, train_Y)
    pred_Y = clf.predict(test_X)
    return accuracy_score(test_Y, pred_Y)

def model_selection_word_count(metod_df, unic_words):
    
    w2v = Word2Vec(sentences=[unic_words], min_count=1)

    X = np.zeros((len(metod_df.word), w2v.wv.vector_size + 3))
    for i, word in enumerate(metod_df.word):
        X[i,:w2v.wv.vector_size] = w2v.wv[word]
    X[:,w2v.wv.vector_size] = metod_df['Negative counts']
    X[:,w2v.wv.vector_size + 1] = metod_df['Neutral counts']
    X[:,w2v.wv.vector_size + 2] = metod_df['Positive counts']
    Y = pd.cut((-metod_df['Negative'] + metod_df['Positive']) * 0.5 **metod_df['Neutral'], bins=[-2, -0.33, 0.33,2], labels=[-1, 0, 1])
    train_X, test_X, train_Y, test_Y = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    clf = LogisticRegression(solver='saga')
    param_grid = {
        'C': np.arange(1, 5)
    }

    search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=5, refit=True, scoring='accuracy')
    search.fit(train_X, train_Y)
    clf = LogisticRegression(C=search.best_params_['C'])
    clf.fit(train_X, train_Y)
    pred_Y = clf.predict(test_X)
    return accuracy_score(test_Y, pred_Y)

def model_selection_tfidf(metod_df, unic_words):
    
    w2v = Word2Vec(sentences=[unic_words], min_count=1)

    X = np.zeros((len(metod_df.word), w2v.wv.vector_size))
    for i, word in enumerate(metod_df.word):
        X[i,:w2v.wv.vector_size] = w2v.wv[word]
    Y = pd.cut((-metod_df['Negative TFIDF'] + metod_df['Positive TFIDF']) * 0.5 **metod_df['Neutral TFIDF'], bins=[-2, -0.33, 0.33,2], labels=[-1, 0, 1])
    train_X, test_X, train_Y, test_Y = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    clf = LogisticRegression(solver='saga')
    param_grid = {
        'C': np.arange(1, 5)
    }

    search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=5, refit=True, scoring='accuracy')
    search.fit(train_X, train_Y)
    clf = LogisticRegression(C=search.best_params_['C'])
    clf.fit(train_X, train_Y)
    pred_Y = clf.predict(test_X)
    return accuracy_score(test_Y, pred_Y)