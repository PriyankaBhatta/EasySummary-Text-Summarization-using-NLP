#Made by: Priyanka Bhatta
#This code is a Python script for a web application.  


from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.utils.translation import gettext_lazy as _
import requests
from bs4 import BeautifulSoup
import numpy as np                                     #used for mathematical operations and numerical computations
import nltk                                            #python library for working with human language data
from nltk.stem import PorterStemmer 
from nltk.tokenize import sent_tokenize , word_tokenize #it is used to tokenize text
from nltk.corpus import stopwords                      #reduces noise from text
nltk.download('stopwords')                        
stop_words = set(stopwords.words("english"))                     
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

'''
#The home function is a simple view that returns the home.html template.
def home(request):
    return render(request, "home.html")

#This preprocessing step is useful to clean the input text and make it ready for further processing like summary generation.
def preprocess(document):
    return document

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
#The home function is the view for your home page
def home(request):
    return render(request, "home.html")

#This is the processing function used to summarize tex
def preprocess(input_text):
    stop_words = set(stopwords.words("english"))
    words = nltk.word_tokenize(input_text)
    words = [word.lower() for word in words if word.isalpha() and word not in stop_words]
    return " ".join(words)


tfidf_vectorizer = TfidfVectorizer()


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


def text_rank(input_text, length=0.15):
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


# summarizenow is the main function that is exposed as a view in the Django web framework and handles processing user inputs,calling the other functions as necessary, and rendering the output in a template.
def summarizenow(request):
    input_text = ""
    summary = ""
    if request.method == 'POST' or request.method == 'GET':
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
                    summary = text_rank([input_text], length=0.15)

                elif summary_length == 2:
                    summary = text_rank([input_text], length=0.5)

                else:   
                    summary = text_rank([input_text], length=0.8)

        else:
            summary_length = int(request.GET.get("summary_length", 2))
            # Use AI algorithm to generate summary
            if summary_length == 1:

                summary = text_rank([input_text], length=0.15)

            elif summary_length == 2:

                summary = text_rank([input_text], length=0.5)

            else:

                summary = text_rank([input_text], length=0.8)

        if request.method == 'GET':
            summary_length = request.GET.get("summary_length","medium")

            #retrive the summary based on the desired length
            return JsonResponse({"summary": summary})

        else:
            return render(request, 'home.html',{'input_text': input_text, 'summary':summary}) 
    else:  
        return render(request, 'home.html',{'input_text': input_text, 'summary':summary})

