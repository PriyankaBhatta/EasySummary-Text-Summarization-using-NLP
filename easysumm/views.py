
#Made by: Priyanka Bhatta
#This code is a Python script for a web application.  
'''
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

#The home function is the view for your home page
def home(request):
    return render(request, "home.html")

#This is the processing function used to summarize text
def preprocess(input_text):
    stop_words = set(stopwords.words("english"))
    words = nltk.word_tokenize(input_text)
    words = [word.lower() for word in words if word.isalpha() and word not in stop_words]
    return " ".join(words)

tfidf_vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')


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
            
            # Use AI algorithm to generate summary
            if summary_length == 1:
                summary = text_rank([input_text], length=0.15)
            elif summary_length == 2:
                summary = tf_idf([input_text], length=0.5)
            else:
                summary = text_rank([input_text], length=0.8)

            return render(request, 'home.html',{'input_text': input_text, 'summary':summary}) 
        else:
            return render(request, 'home.html',{'input_text': input_text, 'error_message':"Please provide some input."})
    else:  
        return render(request, 'home.html',{'input_text': input_text, 'summary':summary})
 
'''
from django.shortcuts import render
from django.http import HttpResponse
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize

nltk.download('stopwords')
stop_words = stopwords.words('english')

def home(request):
    return render(request, "home.html")

def summarize(input_text, summary_length):
    sentences = sent_tokenize(input_text)
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(sentences)
    scores = X.sum(axis=1)
    scores = scores / scores.max(axis=0)
    ranked_sentences = []
    for i, score in enumerate(scores):
        ranked_sentences.append((score, i))
    ranked_sentences.sort(reverse=True)
    top_sentence_indices = [ranked_sentences[i][1] for i in range(summary_length)]
    top_sentence_indices.sort()
    summary = [sentences[i] for i in top_sentence_indices]
    return ' '.join(summary)

def get_paragraphs(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    paragraphs = soup.find_all('p')
    return '\n'.join([p.get_text() for p in paragraphs])

def summarizenow(request):
    output_text = ''
    error_message = ''
    input_text = ''
    summary = ''
    
    if request.method == 'POST':
        try:
            url = request.POST['urlInput']
            input_text = get_paragraphs(url)
        except:
            input_text = request.POST['text']

        if len(input_text.strip()) > 0:
            summary_length = request.POST.get('summary_length','small')
            if summary_length == 'small':
                summary_length = 3
            elif summary_length == 'medium':
                summary_length = 5
            else:
                summary_length = 7
            
            summary = summarize(input_text, summary_length)
            output_text = summary

        else:
            error_message = 'Please enter some text or provide a valid URL.'

    return render(request, 'home.html', {'output_text': output_text,
                                        'error_message': error_message,
                                        'input_text': input_text,
                                        'summary': summary})

