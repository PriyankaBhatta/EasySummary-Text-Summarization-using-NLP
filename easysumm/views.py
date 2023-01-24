from django.shortcuts import render
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Create your views here.
def home(request):
    return render(request, "home.html")

#def summarizenow(request):
    #return render(request, "summarizenow.html", {})

def summarizenow(request):
    text = ""
    summary = ""
    if request.method == 'POST':
        text = request.POST.get('text')
        tokens = nltk.word_tokenize(text)
        stopwords = nltk.corpus.stopwords.words("english")
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
        filtered_text = " ".join(filtered_tokens)

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
    return render(request, 'home.html', {'summary': summary, 'input_text': text})
