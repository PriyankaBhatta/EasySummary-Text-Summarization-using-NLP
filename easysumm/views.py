
#Made by: Priyanka Bhatta
#This code is a Python script for a web application.  

from django.shortcuts import render 
from django.shortcuts import redirect
from django.http import HttpResponse
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import PyPDF2
from PyPDF2 import PdfFileReader
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

#this function is for PDF files only
def get_summary_from_PDF(file, summary_length):
    pdf_reader = PdfFileReader(file)
    summary = ''

    #extract abstract, introduction and conclusion parts
    for page_num in range(min(3, pdf_reader.getNumPages())):
        page = pdf_reader.getPage(page_num)
        text = page.extractText().replace('\n', '')
        if 'abstract' in text.lower():
            summary += text
        elif 'introduction' in text.lower():
            summary += text
        elif 'conclusion' in text.lower():
            summary += text

    #if no abstarct, introduction or conclusion is found, use first page
    if not summary:
        page = pdf_reader.getPage(0)
        summary = page.extractText().replace('\n', '')

    #summarize the extracted text
    return summarizenow(summary, summary_length)

#this function carries out the summary using summarizenow tag in html.
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
                summary_length = 5
            elif summary_length == 'medium':
                summary_length = 9
            else:
                summary_length = 11

            
            # get summary of abstract, introduction, and conclusion
            abstract_intro_conclusion_summary = get_summary_from_PDF(paragraphs)
            summary = summarize(abstract_intro_conclusion_summary, summary_length)
            
            # add remaining content to the summary
            summary += summarize('.\n'.join(paragraphs), summary_length)

            output_text = summary

            #summary = summarize('.\n'.join(paragraphs), summary_length)
            #output_text = summary

        except:
            try:
                url = request.POST['urlInput']
                input_text = get_paragraphs(url)
            except:
                input_text = request.POST['text']

            if len(input_text.strip()) > 0:
                summary_length = request.POST.get('summary_length','small')
                if summary_length == 'small':
                    summary_length = 5
                elif summary_length == 'medium':
                    summary_length = 9
                else:
                    summary_length = 11

                summary = summarize(input_text, summary_length)
                output_text = summary

            else:
                error_message = 'Please enter some text or provide a valid URL.'

    return render(request, 'home.html', {'output_text': output_text,
                                        'error_message': error_message,
                                        'input_text': input_text,
                                        'summary': summary})

