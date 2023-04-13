
#Made by: Priyanka Bhatta
#This code is a Python script for a web application.  

from django.shortcuts import render 
from django.shortcuts import redirect
from django.http import HttpResponse
import requests
import numpy as np
from bs4 import BeautifulSoup
import PyPDF2
import docx2txt
import docx
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
    input_text = re.sub(r'\[\d+\]', '', input_text)  # remove numbers like [17], [71], etc
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
def summarize_file(file):
    file_type = file.name.split('.')[-1]
    if file_type == 'pdf':
        pdf_reader = PyPDF2.PdfFileReader(file)
        input_text = ' '
        for page_num in range(pdf_reader.getNumPages()):
            page = pdf_reader.getPage(page_num)
            input_text += page.extractText()
    elif file_type == 'docx':
        input_text = docx2txt.process(file)
    else:
        return ['Invaid file format.']
    
    summary = summarize(input_text, summary_length=9)
    summary_paragraphs = summary.split('\n')
    return summary_paragraphs
    #return '\n'.join(input_text)


def summarizenow(request):
    output_text = ''
    input_text = ''
    summary = ''
    
    if request.method == 'POST':
        try:
            file = request.FILES['file']
            input_text = summarize_file(file)
            summary_length = request.POST.get('summary_length','small')
            if summary_length == 'small':
                summary_length = 9
            elif summary_length == 'medium':
                summary_length = 15
            else:
                summary_length = 19
            summary = summarize(input_text, summary_length)
            output_text = summary

        except:
            try:
                url = request.POST['urlInput']
                input_text = get_paragraphs(url)
                summary_length = request.POST.get('summary_length','small')
                if summary_length == 'small':
                    summary_length = 9
                elif summary_length == 'medium':
                    summary_length = 15
                else:
                    summary_length = 19
                summary = summarize(input_text, summary_length)
                output_text = summary
                
            except:
                input_text = request.POST['text']
                if len(input_text.strip()) >0:
                    summary_length = request.POST.get('summary_length', 'small')
                    if summary_length == 'small':
                        summary_length = 9
                    elif summary_length == 'medium':
                        summary_length = 15
                    else:
                        summary_length = 19
                    summary = summarize(input_text, summary_length)
                    output_text = summary

    else:
        output_text = 'The file or URL doesnt have valid text to be summarized.'

    return render(request, 'home.html', {'output_text':output_text,
                                         'input_text': input_text, 
                                         'summary':summary,
                                         })


    