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
from classification_utils import *
from text_rank_utils import *
from gensim.summarization import keywords


def create_keys(text,word_count,summary_limit,pca_object,tokenizer_object,vocab,loaded_model):
    article = clean(text,no_urls=True,replace_with_url="",lower=False)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(article)
    summary = CentroidWordEmbeddingsSummarizer(BaseSummarizer).summarize(text=article,limit_type = 'word',limit=int(len(tokens)*summary_limit))
    probability,label =  prediction_bow(loaded_model,text,tokenizer_object,pca_object,vocab)
    keywords_text = keywords(text,words =word_count).split('\n')
    return(summary,float(probability[0]),float(label),keywords_text)




words_count = 10
summary_shrink = 0.25


with open('model_objects.pkl', 'rb') as f:
    model_objects_list = pickle.load(f)
pca_object       = model_objects_list[0]
tokenizer_object = model_objects_list[1]
    
vocab =  load_vocab('vocab.txt')
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
    
    
def create_keys_article_mongo(text,words_count,summary_shrink,pca_object,tokenizer_object,vocab,loaded_model):
    try:
        summary,probability_label,label,keywords_list = create_keys(text,words_count,summary_shrink,pca_object,tokenizer_object,vocab,loaded_model)
        
    except:
        summary='None'
        probability_label = 'None' 
        label= 'None' 
        keywords_list='None'  
    return(summary,probability_label,label,keywords_list)
        
        
        
        
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["aigk"]
mycol = mydb["articles"]
articles_collection = mycol.find({"summary":{'$exists': False}})

count = 0
for i in articles_collection:
    if i['source_id'] != 'forum_ias':
        count = count+1
        print(count)
        if i['source_id'] == 'businessline':
            total_string = ''
            for j in i['text']:
                total_string = total_string +' '+ j
            text_article = total_string
            
        elif i['source_id'] == 'gktoday':
            a = i['text']
            if ';' in a:
                text_article = a.split(';')[1]
            else:
                text_article = a
        else:
            text_article=i['text']
                
        
        try:
            summary,probability_label,label,keywords_list = create_keys_article_mongo(text_article,words_count,summary_shrink,pca_object,tokenizer_object,vocab,loaded_model)
        
        except:
            summary='None_2'
            probability_label = 'None_2' 
            label= 'None_2' 
            keywords_list='None_2'  
        
        if (summary == 'None') or (summary == 'None_2'):
            print("none noted")
        new_keys={'$set':{'summary':summary,'probability_label':probability_label,'label':label,'keywords_list':keywords_list}}
        if i['source_id']=='PIB':
            update_id = {'_id':i['_id']}
        else:
            update_id={'_id': ObjectId(i['_id'])}
        mycol.update_one(update_id,new_keys)
