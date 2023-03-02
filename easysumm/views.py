
#Made by: Priyanka Bhatta
#This code is a Python script for a web application.  

from django.shortcuts import render
from django.http import HttpResponse
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import PyPDF2
import io
from PyPDF2 import PdfFileReader
import docx2txt
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
    
'''
#this two functions work
def summarize_file(file, summary_length):
    # Extract paragraphs
    paragraphs = extract_paragraphs(file)

    # Join paragraphs into single string
    input_text = '\n'.join(paragraphs)

    # Summarize the text
    return summarize(input_text, summary_length)


def extract_paragraphs(file):
    if file.name.endswith('.docx'):
        text = docx2txt.process(file)
        paragraphs = [p for p in text.split('\n') if len(p.strip()) > 0]
    elif file.name.endswith('.pdf'):
        pdf_reader = PdfFileReader(file)
        pages = [pdf_reader.getPage(i).extractText() for i in range(pdf_reader.getNumPages())]
        text = '\n'.join(pages)
        paragraphs = [p for p in text.split('\n') if len(p.strip()) > 0]
    else:
        paragraphs = []
    return paragraphs
'''
def get_paragraphs_from_file(file):
    paragraphs = []
    # Handle PDF file
    if file.name.endswith('.pdf'):
        pdf_reader = PdfFileReader(file)
        for i in range(pdf_reader.getNumPages()):
            page = pdf_reader.getPage(i)
            text = page.extractText()
            paragraphs += [p.strip() for p in text.split('\n\n') if p.strip()]
    # Handle DOCX file
    elif file.name.endswith('.docx'):
        text = docx2txt.process(file)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    # Handle other file types
    else:
        text = file.read().decode('utf-8')
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs

def summarizenow(request):
    output_text = ''
    error_message = ''
    input_text = ''
    summary = ''
    
    if request.method == 'POST':
        try:
            file = request.FILES['file']
            paragraphs = get_paragraphs_from_file(file)
            summary_length = request.POST.get('summary_length','small')
            if summary_length == 'small':
                summary_length = 3
            elif summary_length == 'medium':
                summary_length = 5
            else:
                summary_length = 7

            summary = summarize('.\n'.join(paragraphs), summary_length)
            output_text = summary

        except:
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

'''

def summarize_file(file, summary_length):
    # Handle PDF file
    if file.name.endswith('.pdf'):
        pdf_reader = PdfFileReader(file)
        pages = [pdf_reader.getPage(i).extractText() for i in range(pdf_reader.getNumPages())]
        input_text = '\n'.join(pages)
    # Handle DOCX file
    elif file.name.endswith('.docx'):
        input_text = docx2txt.process(file)
    # Handle other file types
    else:
        input_text = file.read().decode('utf-8')

    return summarize(input_text, summary_length)


def extract_paragraphs(file):
    if file.name.endswith('.docx'):
        text = docx2txt.process(file)
        paragraphs = [p for p in text.split('\n') if len(p.strip()) > 0]
    else:
        paragraphs = []
    return paragraphs
'''

'''
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

'''
 