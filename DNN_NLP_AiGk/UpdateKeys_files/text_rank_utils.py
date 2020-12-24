import re
import string
import unidecode
import numpy as np
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
from nltk.tokenize import word_tokenize as nltk_word_tokenize
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
from gensim.summarization.textcleaner import split_sentences as gensim_sent_tokenize
import flashtext
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
from cleantext import clean
from nltk.tokenize import RegexpTokenizer






def similarity(v1, v2):
    score = 0.0
    if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
        score = ((1 - cosine(v1, v2)) + 1) / 2
    return score


class BaseSummarizer:

    extra_stopwords = ["''", "``", "'s"]

    def __init__(self,
                 language='english',
                 preprocess_type='nltk',
                 stopwords_remove=True,
                 length_limit=10,
                 debug=False):
        self.language = language
        self.preprocess_type = preprocess_type
        self.stopwords_remove = stopwords_remove
        self.length_limit = length_limit
        self.debug = debug
        if stopwords_remove:
            stopword_remover = flashtext.KeywordProcessor()
            for stopword in stopwords.words('english'):
                stopword_remover.add_keyword(stopword, '')
            self.stopword_remover = stopword_remover
        return

    def sent_tokenize(self, text):
        if self.preprocess_type == 'nltk':
            sents = nltk_sent_tokenize(text, 'english')
        else:
            sents = gensim_sent_tokenize(text)
        sents_filtered = []
        for s in sents:
            if s[-1] != ':' and len(s) > self.length_limit:
                sents_filtered.append(s)
            # else:
            #   print("REMOVED!!!!" + s)
        return sents_filtered

    def preprocess_text_nltk(self, text):
        sentences = self.sent_tokenize(text)
        sentences_cleaned = []
        for sent in sentences:
            if self.stopwords_remove:
                self.stopword_remover.replace_keywords(sent)
            words = nltk_word_tokenize(sent, 'english')
            words = [w for w in words if w not in string.punctuation]
            words = [w for w in words if w not in self.extra_stopwords]
            words = [w.lower() for w in words]
            sentences_cleaned.append(" ".join(words))
        return sentences_cleaned

    def preprocess_text_regexp(self, text):
        sentences = self.sent_tokenize(text)
        sentences_cleaned = []
        for sent in sentences:
            sent_ascii = unidecode.unidecode(sent)
            cleaned_text = re.sub("[^a-zA-Z0-9]", " ", sent_ascii)
            if self.stopwords_remove:
                cleaned_text = self.stopword_remover.replace_keywords(cleaned_text)
            words = cleaned_text.lower().split()
            sentences_cleaned.append(" ".join(words))
        return sentences_cleaned

    def preprocess_text(self, text):
        if self.preprocess_type == 'nltk':
            return self.preprocess_text_nltk(text)
        else:
            return self.preprocess_text_regexp(text)

    def summarize(self, text, limit_type='word', limit=100):
        raise NotImplementedError("Abstract method")
        
        
#from text_summarizer import base
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class CentroidBOWSummarizer(BaseSummarizer):

    def __init__(self,
                 language='english',
                 preprocess_type='nltk',
                 stopwords_remove=True,
                 length_limit=10,
                 debug=False,
                 topic_threshold=0.3,
                 sim_threshold=0.95):
        super().__init__(language, preprocess_type, stopwords_remove, length_limit, debug)
        self.topic_threshold = topic_threshold
        self.sim_threshold = sim_threshold
        return

    def summarize(self, text, limit_type='word', limit=100):
        raw_sentences = self.sent_tokenize(text)
        clean_sentences = self.preprocess_text(text)

        vectorizer = CountVectorizer()
        sent_word_matrix = vectorizer.fit_transform(clean_sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())
        for i in range(centroid_vector.shape[0]):
            if centroid_vector[i] <= self.topic_threshold:
                centroid_vector[i] = 0

        sentences_scores = []
        for i in range(tfidf.shape[0]):
            score = similarity(tfidf[i, :], centroid_vector)
            sentences_scores.append((i, raw_sentences[i], score, tfidf[i, :]))

        sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)

        count = 0
        sentences_summary = []
        for s in sentence_scores_sort:
            if count > limit:
                break
            include_flag = True
            for ps in sentences_summary:
                sim = similarity(s[3], ps[3])
                # print(s[0], ps[0], sim)
                if sim > self.sim_threshold:
                    include_flag = False
            if include_flag:
                # print(s[0], s[1])
                sentences_summary.append(s)
                if limit_type == 'word':
                    count += len(s[1].split())
                else:
                    count += len(s[1])

        summary = "\n".join([s[1] for s in sentences_summary])
        return summary
    
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.models import KeyedVectors
import gensim.downloader as gensim_data_downloader


def average_score(scores):
    score = 0
    count = 0
    for s in scores:
        if s > 0:
            score += s
            count += 1
    if count > 0:
        score /= count
        return score
    else:
        return 0


def stanford_cerainty_factor(scores):
    score = 0
    minim = 100000
    for s in scores:
        score += s
        if s < minim & s > 0:
            minim = s
    score /= (1 - minim)
    return score


def get_max_length(sentences):
    max_length = 0
    for s in sentences:
        l = len(s.split())
        if l > max_length:
            max_length = l
    return max_length


def load_gensim_embedding_model(model_name):
    available_models = gensim_data_downloader.info()['models'].keys()
    assert model_name in available_models, 'Invalid model_name: {}. Choose one from {}'.format(model_name, ', '.join(available_models))
    model_path = gensim_data_downloader.load(model_name, return_path=True)
    return KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')


class CentroidWordEmbeddingsSummarizer(BaseSummarizer):
    def __init__(self,
                 embedding_model,
                 language='english',
                 preprocess_type='nltk',
                 stopwords_remove=True,
                 length_limit=10,
                 debug=False,
                 topic_threshold=0.3,
                 sim_threshold=0.95,
                 reordering=True,
                 zero_center_embeddings=False,
                 keep_first=False,
                 bow_param=0,
                 length_param=0,
                 position_param=0):
        super().__init__(language, preprocess_type, stopwords_remove, length_limit, debug)

        self.embedding_model = embedding_model

        self.word_vectors = dict()

        self.topic_threshold = topic_threshold
        self.sim_threshold = sim_threshold
        self.reordering = reordering

        self.keep_first = keep_first
        self.bow_param = bow_param
        self.length_param = length_param
        self.position_param = position_param

        self.zero_center_embeddings = zero_center_embeddings

        if zero_center_embeddings:
            self._zero_center_embedding_coordinates()
        return

    def get_bow(self, sentences):
        vectorizer = CountVectorizer()
        sent_word_matrix = vectorizer.fit_transform(sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())
        for i in range(centroid_vector.shape[0]):
            if centroid_vector[i] <= self.topic_threshold:
                centroid_vector[i] = 0
        return tfidf, centroid_vector

    def get_topic_idf(self, sentences):
        vectorizer = CountVectorizer()
        sent_word_matrix = vectorizer.fit_transform(sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())
        feature_names = vectorizer.get_feature_names()

        relevant_vector_indices = np.where(centroid_vector > self.topic_threshold)[0]

        word_list = list(np.array(feature_names)[relevant_vector_indices])
        return word_list

    def word_vectors_cache(self, sentences):
        self.word_vectors = dict()
        for s in sentences:
            words = s.split()
            for w in words:
                if self.word_vectors.get(w) is not None:
                    if self.zero_center_embeddings:
                        self.word_vectors[w] = (self.embedding_model[w] - self.centroid_space)
                    else:
                        self.word_vectors[w] = self.embedding_model[w]
        return
    
    def compose_vectors(self, words):
        composed_vector = np.zeros(model.vector_size, dtype="float32")
        word_vectors_keys = set(self.word_vectors.keys())
        count = 0
        for w in words:
            if w in word_vectors_keys:
                composed_vector = composed_vector + self.word_vectors[w]
                count += 1
        if count != 0:
            composed_vector = np.divide(composed_vector, count)
        return composed_vector

    def summarize(self, text, limit_type='word', limit=100):
        raw_sentences = self.sent_tokenize(text)
        clean_sentences = self.preprocess_text(text)

        if self.debug:
            print("ORIGINAL TEXT STATS = {0} chars, {1} words, {2} sentences".format(len(text),
                                                                                     len(text.split(' ')),
                                                                                     len(raw_sentences)))
            print("*** RAW SENTENCES ***")
            for i, s in enumerate(raw_sentences):
                print(i, s)
            print("*** CLEAN SENTENCES ***")
            for i, s in enumerate(clean_sentences):
                print(i, s)

        centroid_words = self.get_topic_idf(clean_sentences)

        if self.debug:
            print("*** CENTROID WORDS ***")
            print(len(centroid_words), centroid_words)

        self.word_vectors_cache(clean_sentences)
        centroid_vector = self.compose_vectors(centroid_words)

        tfidf, centroid_bow = self.get_bow(clean_sentences)
        max_length = get_max_length(clean_sentences)

        sentences_scores = []
        for i in range(len(clean_sentences)):
            scores = []
            words = clean_sentences[i].split()
            sentence_vector = self.compose_vectors(words)

            scores.append(similarity(sentence_vector, centroid_vector))
            scores.append(self.bow_param * similarity(tfidf[i, :], centroid_bow))
            scores.append(self.length_param * (1 - (len(words) / max_length)))
            scores.append(self.position_param * (1 / (i + 1)))
            score = average_score(scores)
            sentences_scores.append((i, raw_sentences[i], score, sentence_vector))

            if self.debug:
                print(i, scores, score)

        sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)
        if self.debug:
            print("*** SENTENCE SCORES ***")
            for s in sentence_scores_sort:
                print(s[0], s[1], s[2])

        count = 0
        sentences_summary = []

        if self.keep_first:
            for s in sentence_scores_sort:
                if s[0] == 0:
                    sentences_summary.append(s)
                    if limit_type == 'word':
                        count += len(s[1].split())
                    else:
                        count += len(s[1])
                    sentence_scores_sort.remove(s)
                    break

        for s in sentence_scores_sort:
            if count > limit:
                break
            include_flag = True
            for ps in sentences_summary:
                sim = similarity(s[3], ps[3])
                if sim > self.sim_threshold:
                    include_flag = False
            if include_flag:
                sentences_summary.append(s)
                if limit_type == 'word':
                    count += len(s[1].split())
                else:
                    count += len(s[1])

        if self.reordering:
            sentences_summary = sorted(sentences_summary, key=lambda el: el[0], reverse=False)

        summary = "\n".join([s[1] for s in sentences_summary])

        if self.debug:
            print("SUMMARY TEXT STATS = {0} chars, {1} words, {2} sentences".format(len(summary),
                                                                                    len(summary.split(' ')),
                                                                                    len(sentences_summary)))

            print("*** SUMMARY ***")
            print(summary)

        return summary

    def _zero_center_embedding_coordinates(self):
        count = 0
        self.centroid_space = np.zeros(model.vector_size, dtype="float32")
        self.index2word_set = set(self.embedding_model.wv.index2word)
        for w in self.index2word_set:
            self.centroid_space = self.centroid_space + self.embedding_model[w]
            count += 1
        if count != 0:
            self.centroid_space = np.divide(self.centroid_space, count)
            
import gensim

model =  gensim.models.KeyedVectors.load_word2vec_format('glove.6B.100d.txt', binary=False)

