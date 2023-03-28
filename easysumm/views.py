
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
import docx
import io
import docx2txt
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from builtins import PendingDeprecationWarning



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

#this function is used to extract text from files 

def extract_text(file_path, file_format, summary_length):
    if file_format == 'docx':
        doc = docx.Document(file_path)
        abstract = []
        intro = []
        conclusion = []
        in_summary = False
        
        for para in doc.paragraphs:
            # Check for start of summary
            if not in_summary and re.search(r'^Abstract', para.text):
                in_summary = True
            elif not in_summary and re.search(r'^Introduction', para.text):
                in_summary = summary_length in ('medium', 'long')
            elif not in_summary and re.search(r'^Conclusion', para.text):
                in_summary = summary_length == 'long'
            
            # Add paragraphs to appropriate sections
            if in_summary:
                if para.text:
                    if re.search(r'^Abstract', para.text):
                        current_section = abstract
                    elif re.search(r'^Introduction', para.text):
                        current_section = intro
                    elif re.search(r'^Conclusion', para.text):
                        current_section = conclusion
                    else:
                        # Exclude headings and figures numbers
                        if not re.search(r'^\s*\d+\.\s', para.text) and not re.search(r'^\s*\d+\s', para.text):
                            current_section.append(para.text)
        
        # Combine paragraphs from each section
        summary = []
        if summary_length == 'short':
            summary = abstract
        elif summary_length == 'medium':
            summary = abstract + intro
        else:
            summary = abstract + intro + conclusion
        
        # Join paragraphs into a single string
        return '\n'.join(summary)
    
    elif file_format == 'pdf':
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfFileReader(f)
            text = ''
            for i in range(reader.getNumPages()):
                text += reader.getPage(i).extractText()
        
        abstract_start = re.search(r'^Abstract', text, flags=re.MULTILINE)
        intro_start = re.search(r'^Introduction', text, flags=re.MULTILINE)
        conclusion_start = re.search(r'^Conclusion', text, flags=re.MULTILINE)
        
        if abstract_start and intro_start and conclusion_start:
            abstract = text[abstract_start.end(): intro_start.start()]
            intro = text[intro_start.end(): conclusion_start.start()]
            conclusion = text[conclusion_start.end():]
            
            summary = []
            if summary_length == 'short':
                summary = abstract.split('\n\n')
            elif summary_length == 'medium':
                summary = (abstract + intro).split('\n\n')
            else:
                summary = (abstract + intro + conclusion).split('\n\n')
            
            # Exclude headings and figures numbers
            summary = [para for para in summary if not re.search(r'^\s*\d+\.\s', para) and not re.search(r'^\s*\d+\s', para)]
            
            # Join paragraphs into a single string
            return '\n\n'.join(summary)
        else:
            raise ValueError('Could not find required sections in the document.')
    
    else:
        raise ValueError('Unsupported file format.')



#this function carries out the summary using 
# summarizenow tag in html.
def summarizenow(request):
    output_text = ''
    error_message = ''
    input_text = ''
    summary = ''
    current_summary_length = 'small'
    
    if request.method == 'POST':
        try:
            file = request.FILES['file']

            if file.name.endswith('.pdf'):
                input_text = extract_text(file, file.name.split('.')[-1], request.POST.get('summary_length', 'short'))
                current_summary_length = request.POST.get('summary_length', 'small')
            elif file.name.endswith('.docx'):
                input_text = extract_text(file, file.name.split('.')[-1], request.POST.get('summary_length', 'short'))
                current_summary_length = request.POST.get('summary_length','small')

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
                current_summary_length = summary_length
                
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
                current_summary_length = summary_length

            else:
                error_message = 'Please enter some text or provide a valid URL.'

    return render(request, 'home.html', {'output_text': output_text,
                                        'error_message': error_message,
                                        'input_text': input_text,
                                        'summary': summary,
                                        'current_summary_length': current_summary_length})

