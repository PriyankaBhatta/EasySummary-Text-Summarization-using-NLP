from django.shortcuts import render
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import PyPDF2
from PyPDF2 import PdfFileReader

# Create your views here.
def home(request):
    return render(request, "home.html")

#def summarizenow(request):
    #return render(request, "summarizenow.html", {})

def summarizenow(request):
    text = ""
    summary = ""
    if request.method == 'POST':
        # check if a text is submitted
        if 'text' in request.POST:
            text = request.POST.get('text')
        # check if a file is uploaded
        elif 'file' in request.FILES:
            file = request.FILES['file']
            # check if the file is a pdf
            if file.content_type == 'application/pdf':
                pdf_reader = PyPDF2.PdfFileReader(file)
                text = " ".join([pdf_reader.getPage(i).extractText() for i in range(pdf_reader.numPages)])
            # check if the file is a doc
            elif file.content_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                text = file.read().decode()
        # perform text pre-processing and summarization here
        tokens = nltk.word_tokenize(text)
        stopwords = nltk.corpus.stopwords.words("english")
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
        filtered_text = " ".join(filtered_tokens)

        if filtered_text:
            tfidf = TfidfVectorizer()
            scores = tfidf.fit_transform([filtered_text])

            sentence_scores = {}
            for i, sentence in enumerate(nltk.sent_tokenize(text)):
                try:
                    sentence_scores[sentence] = sum(scores[i])
                except IndexError:
                    pass
            N = 10
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:N]
            summary = " ".join([s[0] for s in top_sentences])
        else:
            summary = "The file contains only stopwords and no meaningful text."

    return render(request, 'home.html', {'summary': summary, 'input_text': text})
