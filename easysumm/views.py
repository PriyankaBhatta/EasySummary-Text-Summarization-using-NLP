
#Made by: Priyanka Bhatta
#This code is a Python script for a web application.  
<<<<<<< HEAD
import requests
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from bs4 import BeautifulSoup
import numpy as np 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline

nltk.download('stopwords')
nltk.download('punkt')

=======
import re
import string
from typing import List
import io
import docx
import PyPDF2
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import requests
from bs4 import BeautifulSoup
import numpy as np                                     #used for mathematical operations and numerical computations
import nltk                                            #python library for working with human language data
from nltk.stem import PorterStemmer 
from nltk.tokenize import sent_tokenize #it is used to tokenize text
from nltk.corpus import stopwords                      #reduces noise from text
nltk.download('stopwords')  
nltk.download('punkt')                      
stop_words = set(stopwords.words("english"))                     
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import docx
from sklearn.metrics.pairwise import cosine_similarity
from django.core.files.uploadedfile import UploadedFile  #for handling doc files
from docx import Document

'''


#performs tf-idf operation
def tf_idf(documents, length=0.15):

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    feature_names = tfidf_vectorizer.vocabulary_.keys()


    tfidf_scores = zip(tfidf_matrix.nonzero()[0], tfidf_matrix.data)
    tfidf_scores_tuples = [(index, score) for index, score in tfidf_scores]

    sentences = documents[0].split('. ')
    sentences_with_scores = [(sentence, tfidf_scores_tuples[index][1]) for index, sentence in enumerate(sentences)]

    
    sorted_sentences = sorted(sentences_with_scores, key=lambda x: x[1], reverse=True)
    top_sentences = sorted_sentences[:int(length * len(sentences))]
    summary = '. '.join([sentence[0] for sentence in top_sentences])
    return summary


# the summarizenow function returns the home.html template with the input text and summary as context variables. 
def summarizenow(request):
    input_text = ""
    summary = ""
    if request.method == 'POST':
        
        #for url
        if request.POST.get('urlInput'):
            url = request.POST.get('urlInput')
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            input_text = soup.get_text()
            
            
        #for text input    
        elif request.POST.get('text'):
            input_text = request.POST.get('text', '')

        if input_text:

            #get the selected summary length value
            summary_length = int(request.POST.get("summary_length", 2))
        
            # Use AI algorithm to generate summary
            if summary_length == 1:
                
                summary = tf_idf([input_text], length=0.15)

            elif summary_length == 2:
                
                summary = tf_idf([input_text], length=0.5)

            else:
                
                summary = tf_idf([input_text], length=0.8)

            
           
            return render(request, 'home.html',{'input_text': input_text, 'summary':summary}) 
        else:
            return render(request, 'home.html',{'input_text': input_text, 'summary':summary})
    else:  
        return render(request, 'home.html',{'input_text': input_text, 'summary':summary})

'''
>>>>>>> origin/master
#The home function is the view for your home page
def home(request):
    return render(request, "home.html")

#This is the processing function used to summarize text
def preprocess(input_text):
    stop_words = set(stopwords.words("english"))
    words = nltk.word_tokenize(input_text)
    words = [word.lower() for word in words if word.isalpha() and word not in stop_words]
    return " ".join(words)

<<<<<<< HEAD
tfidf_vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
=======
tfidf_vectorizer = TfidfVectorizer()
>>>>>>> origin/master


def tf_idf(input_text, length=0.15):
   
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
    tfidf_matrix = tfidf_vectorizer.fit_transform(input_text)
    feature_names = tfidf_vectorizer.vocabulary_.keys()


    tfidf_scores = zip(tfidf_matrix.nonzero()[0], tfidf_matrix.data)
    tfidf_scores_tuples = [(index, score) for index, score in tfidf_scores]

    
    sentences = [sentence for sentence in input_text[0].split('. ') if not sentence.isdigit() and len(sentence) > 1]
    sentences_with_scores = [(sentence, tfidf_scores_tuples[index][1]) for index, sentence in enumerate(sentences)]

    
    sorted_sentences = sorted(sentences_with_scores, key=lambda x: x[1], reverse=True)
    top_sentences = sorted_sentences[:int(length * len(sentences))]
    summary = '. '.join([sentence[0] for sentence in top_sentences])
    return summary 


def text_rank(input_text, length=0.5):
    sentences = " ".join(input_text).split('.')
    graph = nx.Graph()
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            if i != j:
                graph.add_edge(i, j, weight=text_similarity(sentences[i], sentences[j]))
    ranks = nx.pagerank(graph)
    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    top_sentences = [sentences[index] for index, score in sorted_ranks[:int(length * len(sentences))]]
    summary = '. '.join(top_sentences)
    return summary
    

def text_similarity(sentence1, sentence2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([sentence1, sentence2]).toarray()
    vector1 = vectors[0]
    vector2 = vectors[1]
    similarity = cosine_similarity(vector1.reshape(1,-1), vector2.reshape(1,-1))
    return similarity[0][0]

def transformers_summarizer(input_text, length=0.8):
    summarizer = pipeline("summarization")
    summary = summarizer(input_text, max_length=int(length*len(input_text)), min_length=int(length*len(input_text)/2), do_sample=False)[0]['summary_text']
    return summary

# summarizenow is the main function that is exposed as a view in the Django web framework and handles processing user inputs,calling the other functions as necessary, and rendering the output in a template.
def summarizenow(request):
    input_text = ""
    summary = ""
<<<<<<< HEAD

    if request.method == 'POST':    
        if request.POST.get('urlInput'):
            url = request.POST.get('urlInput')
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            input_text = soup.get_text()
        elif request.FILES.get('fileInput'):
            file = request.FILES['fileInput']
            input_text = file.read().decode('utf-8')
        elif request.POST.get('text'):
            input_text = request.POST.get('text', '')

        if input_text:
            # get the selected summary length value
            summary_length = int(request.POST.get("summary_length", 2))
            
=======
    sentences = [] #define the variable with an empty list

    if request.method == 'POST':
            
        if request.POST.get('urlInput'):
            url = request.POST.get('urlInput')
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            input_text = soup.get_text()

         #for text input    
        elif request.POST.get('text'):
            input_text = request.POST.get('text', '')  

        if input_text:
            # get the selected summary length value
            summary_length = int(request.POST.get("summary_length", 2))
            
>>>>>>> origin/master
            # Use AI algorithm to generate summary
            if summary_length == 1:
                summary = text_rank([input_text], length=0.15)
            elif summary_length == 2:
                summary = tf_idf([input_text], length=0.5)
            else:
<<<<<<< HEAD
                summary = transformers_summarizer([input_text], length=0.8)
=======
                summary = text_rank([input_text], length=0.8)
>>>>>>> origin/master

            return render(request, 'home.html',{'input_text': input_text, 'summary':summary}) 
        else:
            return render(request, 'home.html',{'input_text': input_text, 'error_message':"Please provide some input."})
    else:  
        return render(request, 'home.html',{'input_text': input_text, 'summary':summary})
<<<<<<< HEAD
 
=======
 
>>>>>>> origin/master
