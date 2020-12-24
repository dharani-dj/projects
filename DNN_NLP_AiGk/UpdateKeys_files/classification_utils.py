import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from cleantext import clean
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.layers import Dropout
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier 
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
import pickle


import pymongo
from bs4 import BeautifulSoup
import requests
import logging
import pymongo
import multiprocessing
import dateparser
import pandas as pd
import datetime
import numpy as np
import re
from tqdm import tnrange,tqdm_notebook
import tqdm
from bson import ObjectId
import fasttext
from keras.models import model_from_json




def data_processing(df_classi): #preprocessing 'text' column
    
    df_classi= shuffle(df_classi)


    text = [] #some articles have text in paragraphs which are in list format.So append all the list items into a single text
    for i in range(len(df_classi)):
        if df_classi['label'].values[i] ==0:
            total_string = ''
            for j in df_classi['text'].values[i]:
                total_string = total_string +' '+ j
            text.append(total_string)        
        else:
            text.append(df_classi['text'].values[i])
    df_classi['text'] = text

    text = []
    for i in range(len(df_classi['text'].values)):
        if df_classi['label'].values[i] ==1:
            a = df_classi['text'].values[i]
            if ';' in a: 
                text.append(a.split(';')[1])
            else:
                text.append(a)
        else:
            text.append(df_classi['text'].values[i])
    df_classi['text']  = text
    return(df_classi)


def split_data(df): 
    train, test = train_test_split(df, test_size=0.2)
    return(train, test)


def split_train_data(df):
    X_train, X_val, Y_train, Y_val = train_test_split(df['text'].values, df['label'].values, test_size = 0.15, random_state=5)
    return(X_train, X_val, Y_train, Y_val)

def clean_doc(text): #clean text with punctuations and stop words
    tokens = clean(text,no_urls=True,no_numbers=True,no_digits=True,no_punct=True).split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english')) 
    tokens = [w for w in tokens if not w in stop_words] #convert all the text into tokens(words)
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def add_doc_to_vocab(text, vocab):
    tokens = clean_doc(text)
    vocab.update(tokens)  #update the vocab file with tokens obtained from clean doc function

def process_docs_vocab(docs, vocab):
    '''
    process docs for creating vocabulary 
    '''
    for i in docs:  #update vocab for all docs
        add_doc_to_vocab(i, vocab)  
        
def save_list_vocab(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
    

        
def create_vocab(docs,save_to_text):
    '''
    Create vocabulary for given list of documents and saving it to text file
    '''
    vocab = Counter() 
    process_docs_vocab(docs, vocab)
    min_occurance = 2
    tokens = [k for k,c in vocab.items() if c >= min_occurance]
    if save_to_text:
        save_list_vocab(tokens, 'vocab.txt')
    return(vocab)



def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
    
def doc_to_line(text, vocab): #all the individual tokens are converted as a corpus from voacb file
    tokens = clean_doc(text)
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)
 
def process_docs(X, vocab):
    lines = list()
    for i in X:
        line = doc_to_line(i, vocab)
        lines.append(line)
    return lines




def pipeline_tokenizer_matrix(X_train,X_val):
    tokenizer = Tokenizer() #It converts input text to streams of tokens, where each token is a separate word, punctuation sign, number/amount, date, e-mail, URL/URI, etc
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_matrix(X_train, mode='freq') #text_to_matrix used to create one vector per document provided per input. The length of the vectors is the total size of the vocabulary.mode used here is frequency i.e., the number in matrix is the frequency of word
    X_val = tokenizer.texts_to_matrix(X_val, mode='freq')
    return(X_train,X_val,tokenizer)


def pipeline_tokenizer_sequence(X_train,X_val):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train,)
    X_train = tokenizer.texts_to_sequences(X_train,) #split text into a list of words.
    max_length = max([len(s) for s in X_train])
    X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
    X_val = tokenizer.texts_to_sequences(X_val)
    X_val = pad_sequences(X_val, maxlen=max_length, padding='post')
    return(X_train,X_val,max_length,tokenizer)




def pipeline_pca(X_train,X_val,N): #PCA on the text abotained after tokenizing
    pca = PCA(n_components=N)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    return(X_train,X_val,pca)



def load_vocab(vocab_filename):
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)
    return(vocab)


def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
def testing_data_bow(model,df_test,tokenizer_object,pca_object,vocab):  #model=model.h5 file  has weights  and where else are we using vocab?
    X_test = df_test['text'].values
    Y_test = df_test['label'].values
    X_test = process_docs(X_test, vocab)
    X_test = tokenizer_object.texts_to_matrix(X_test, mode='freq')
    X_test = pca_object.transform(X_test)
    Y_pred_test = model.predict(X_test)
    Y_pred_test_b = []
    for i in Y_pred_test:
        if i>0.5:
            Y_pred_test_b.append(1)
        else:
            Y_pred_test_b.append(0)
    acc = accuracy_score(Y_test,Y_pred_test_b)
            
    return(Y_pred_test_b,Y_test,acc)
    

def prediction_bow(model,text,tokenizer_object,pca_object,vocab): #this is called to update keys og probability nd label in mongo
    X_test = [text]
    X_test = process_docs(X_test, vocab)
    X_test = tokenizer_object.texts_to_matrix(X_test, mode='freq')
    X_test = pca_object.transform(X_test)
    Y_pred_test = model.predict(X_test) 
    Y_pred_test_b = []
    for i in Y_pred_test:
        if i>0.5:
            Y_pred_test_b.append(1)
        else:
            Y_pred_test_b.append(0)
    return(Y_pred_test[0],Y_pred_test_b[0])

def pipeline_modelling_rf(X_train,X_val,Y_train,Y_val): #random forest (gradient boosting)
    model = GradientBoostingClassifier()
    model.fit(X_train, Y_train)
    Y_pred_train = model.predict(X_train)
    insample_acc = accuracy_score(Y_pred_train,Y_train)
    Y_pred_val =  model.predict(X_val)
    outsample_accuracy = accuracy_score(Y_pred_val,Y_val)
    return(insample_acc,outsample_accuracy,model)

def testing_data_cnn(df_test,tokenizer_object,max_length,vocab): #cnn? 
    X_test = df_test['text'].values
    Y_test = df_test['label'].values
    
    X_test = process_docs(X_test, vocab)
    X_test = tokenizer_object.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, maxlen=max_length, padding='post')
    
    Y_pred_test = model.predict(X_test)   #model not defined
    Y_pred_test_b = []
    for i in Y_pred_test:
        if i>0.5:
            Y_pred_test_b.append(1)
        else:
            Y_pred_test_b.append(0)
    acc = accuracy_score(Y_test,Y_pred_test_b)
            
    return(Y_pred_test_b,Y_test,acc)



def load_embedding(filename):
    file = open(filename,'r')
    lines = file.readlines()
    file.close()
    embedding = dict()
    for line in lines:
        parts = line.split()
        embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
    return embedding

def get_weight_matrix(embedding, vocab): #where is this used (checked in training_process and update_keys files didnt find)
    vocab_size = len(vocab) + 1
    weight_matrix = np.zeros((vocab_size, 100))
    for word, i in vocab.items():
        vector = embedding.get(word)
        if vector is not None:
            weight_matrix[i] = vector
    return weight_matrix



def pipeline_modelling_cnn_embedding(tokenizer,embedding_vectors, X_train,X_val,Y_train,Y_val,v):
    vocab_size = len(tokenizer.word_index) + 1
    embedding_layer = keras.layers.Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False) #converting embedding vector file to be add as a layer in CNN
    model = keras.Sequential()
    model.add(embedding_layer)# add this layer to network
    model.add(keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu')) #cnn with activation relu and kernal size 5
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1, activation='sigmoid')) 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=20, verbose=v,validation_data=(X_val,Y_val))
    loss, acc = model.evaluate(X_val, Y_val, verbose=0)
    return(history,acc,model)



def pipeline_modelling_cnn(tokenizer,X_train,X_val,Y_train,Y_val,v): #CNN model with embedding layer having random weights
    vocab_size = len(tokenizer.word_index) + 1
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 100, input_length=max_length))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='relu')) #hidden layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))  #output layer

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=20, verbose=v,validation_data=(X_val,Y_val))
    loss, acc = model.evaluate(X_val, Y_val, verbose=0)
    return(history,acc,model)

    
def pipeline_modelling_bow(X_train,X_val,Y_train,Y_val,v): #bag of words with DNN
    feature_shape = X_train.shape[1]
    model = keras.Sequential()
    model.add(keras.layers.Dense(150, input_shape=(feature_shape,), activation='relu')) #DNN with 150 neurons
    model.add(keras.layers.Dense(1, activation='sigmoid')) #output layer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history=model.fit(X_train, Y_train, epochs=100,verbose=v,validation_data=(X_val,Y_val))
    loss, acc = model.evaluate(X_val, Y_val)
    return(history,acc,model)
    

def pipeline_modelling_embeddings(tokenizer,embedding_vectors, X_train,X_val,Y_train,Y_val,v): #DNN with embeddings
    vocab_size = len(tokenizer.word_index) + 1
    embedding_layer = keras.layers.Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)#embedding vectors as input layer
    model = keras.Sequential()
    model.add(embedding_layer)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=20, verbose=v,validation_data=(X_val,Y_val))
    loss, acc = model.evaluate(X_val, Y_val, verbose=0)
    return(history,acc,model)



def pipeline_modelling_lstm(tokenizer,X_train,X_val,Y_train,Y_val,v):
    vocab_size = len(tokenizer.word_index) + 1
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 100, input_length=max_length))
    model.add(keras.layers.LSTM(10))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=20, verbose=v,validation_data=(X_val,Y_val))
    loss, acc = model.evaluate(X_val, Y_val, verbose=0)
    return(history,acc,model)