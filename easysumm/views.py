
#Made by: Priyanka Bhatta
#This code is a Python script for a web application.  

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
    clean_paragraphs = []
    for p in paragraphs:
        text = p.get_text()
        clean_text = re.sub(r'\[\d+\]', '', text) # remove numbers like [17], [71], etc.
        clean_paragraphs.append(clean_text)
    return '\n'.join(clean_paragraphs)

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

