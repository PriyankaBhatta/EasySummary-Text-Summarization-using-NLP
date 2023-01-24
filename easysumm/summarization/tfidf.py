import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Text input from user
text = "This is a sample text for demonstration purposes. It can be replaced with the actual text input from the user in the HTML template."

# Tokenize the text
tokens = nltk.word_tokenize(text)

# Remove stop words
stopwords = nltk.corpus.stopwords.words("english")
filtered_tokens = [token for token in tokens if token.lower() not in stopwords]

# Perform Tf-Idf
tfidf = TfidfVectorizer()
scores = tfidf.fit_transform(filtered_tokens)

# Sentence scoring
sentence_scores = {}
for i, sentence in enumerate(nltk.sent_tokenize(text)):
    sentence_scores[sentence] = sum(scores[i])

# Get top N sentences
N = 3
top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:N]

# Print summary
summary = " ".join([s[0] for s in top_sentences])
print(summary)