#This code is a Python script for a web application. The script implements 3 functions: home, preprocess, and summarizenow.

from django.shortcuts import render
from django.http import HttpResponse
import numpy as np            #used for mathematical operations and numerical computations
import nltk                   #python library for working with human language data
from nltk.stem import PorterStemmer 
from nltk.tokenize import sent_tokenize, word_tokenize #it is used to tokenize text
from nltk.corpus import stopwords
nltk.download('stopwords')                  #reduces noise from text
stop_words = set(stopwords.words("english"))                     
from sklearn.feature_extraction.text import TfidfVectorizer

#The home function is a simple view that returns the home.html template.
def home(request):
    return render(request, "home.html")

#This preprocessing step is useful to clean the input text and make it ready for further processing like summary generation.
def preprocess(document):
    return document

#performs tf-idf operation
def tf_idf(documents):
    #documents = [preprocess(document) for document in documents] # preprocess the document
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    feature_names = tfidf_vectorizer.vocabulary_.keys()


    tfidf_scores = zip(tfidf_matrix.nonzero()[0], tfidf_matrix.data)
    tfidf_scores_tuples = [(index, score) for index, score in tfidf_scores]

    sentences = documents[0].split('. ')
    sentences_with_scores = [(sentence, tfidf_scores_tuples[index][1]) for index, sentence in enumerate(sentences)]

    top_sentences = sorted(sentences_with_scores, key=lambda x: x[1], reverse=True)[:10]
    summary = '. '.join([sentence[0] for sentence in top_sentences])
    return summary


# the summarizenow function returns the home.html template
# with the input text and summary as context variables. 
def summarizenow(request):
    input_text = request.POST.get('text', '')
    summary = tf_idf([input_text]) # Use AI algorithm to generate summary
    return render(request, 'home.html', {'input_text': input_text, 'summary':summary})
