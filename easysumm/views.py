from django.shortcuts import render
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
import io
import PyPDF2
from docx2python import docx2python
from sklearn.metrics.pairwise import cosine_similarity

# Create your views here.
def home(request):
    return render(request, "home.html")

def extract_text_from_pdf(file):
    # create a pdf reader object
    pdf_reader = PyPDF2.PdfFileReader(file)
    # iterate over each page
    text = ""
    for page in range(pdf_reader.getNumPages()):
        text += pdf_reader.getPage(page).extractText()
    return text

def generate_summary(text, num_sentences):
    # preprocess the text
    processed_text = preprocess_text(text)
    # create the tf-idf matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_text.splitlines())
    feature_names = vectorizer.get_feature_names()
    # calculate the similarity score between each sentence and the entire text
    sentence_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
    # sort the sentences based on their scores
    sorted_sentences = np.argsort(sentence_scores.flatten())[::-1]
    # return the top-n scored sentences as the summary
    return " ".join([processed_text.splitlines()[i] for i in sorted_sentences[:num_sentences]])

def summarizenow(request):
    if request.method == 'POST':
        input_file = request.FILES.get('input_file')
        if input_file:
            if input_file.content_type == 'application/pdf':
                # handle PDF file input
                pdf_file = PyPDF2.PdfFileReader(input_file.file)
                text = ""
                for page in range(pdf_file.getNumPages()):
                    text += pdf_file.getPage(page).extractText()
                    summary = generate_summary(text, 5)
                    return render(request, 'summarize.html', {'summary': summary})
            elif input_file.content_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                # handle doc file input
                doc_file = docx2python(input_file)
                text = doc_file.get_text()
                summary = generate_summary(text, 5)
                return render(request, 'home.html', {'summary': summary})
            else:
                return render(request, 'home.html', {'error': 'Invalid file type. Please upload a PDF or DOC file.'})
        else:
            return render(request, 'home.html', {'error': 'Please provide a valid file.'})
    else:
        return render(request, 'home.html')
