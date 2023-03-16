
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
import docx2txt
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

#this is the tf-idf function
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

#this function will only extract <p> tags from URL's
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
def get_summary_from_PDF(file):
    pdf_reader = PdfFileReader(file)
    summary = ''

    # extract abstract, introduction and conclusion parts
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
    return summary
'''

def extract_text(file_path, file_format):
    if file_format == 'docx':
        text = docx2txt.process(file_path)
    elif file_format == 'pdf':
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfFileReader(f)
            text = ''
            for i in range(reader.getNumPages()):
                text += reader.getPage(i).extractText()
    else:
        raise ValueError('Unsupported file format.')
    
    abstract_start = text.find('Abstract')
    intro_start = text.find('Introduction')
    conclusion_start = text.find('Conclusion')

    if abstract_start != -1 and intro_start != -1 and conclusion_start != -1:
        abstract = text[abstract_start: intro_start]
        intro = text[intro_start: conclusion_start]
        conclusion = text[conclusion_start:]
        
        return abstract + intro + conclusion
    else:
        raise ValueError('Could not find required sections in the document.')
    
#this function carries out the summary using summarizenow tag in html.
def summarizenow(request):
    output_text = ''
    error_message = ''
    input_text = ''
    summary = ''
    
    if request.method == 'POST':
        try:
            file = request.FILES['file']
            if file.name.endswith('.pdf'):
                input_text = extract_text(file, file.name.split('.')[-1])
            elif file.name.endswith('.docx'):
                input_text = extract_text(file, file.name.split('.')[-1])

            if len(input_text.strip()) > 0:
                summary_length = request.POST.get('summary_length', 'small')
                if summary_length == 'small':
                    summary_length = 5
                elif summary_length == 'medium':
                    summary_length = 9
                else:
                    summary_length = 11

                summary = summarize(input_text, summary_length)
                output_text = summary
                
            else:
                error_message = 'The file could not be processed. Please upload a valid file.'

                
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

