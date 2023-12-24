# views.py

from django.shortcuts import render
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
import re
import string
import joblib
from django.http import JsonResponse
import nltk
import pickle
import numpy as np
import os

from django.shortcuts import render
from django.http import HttpResponse
import re
import pickle
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
from keras.models import model_from_json
import numpy as np










def index(request):
    return render(request, 'pages/index.html')



def sentiment(request):
    return render(request, "includes/sentiment.html")


def sentimentDL(request):
    return render(request, "includes/sentiment_DL.html")

def SentimentBi(request):
    return render(request, "includes/sentiment_BI.html")




def sentiment(request):

    comment = request.GET.get('comment', '')
    
    # Load the pre-trained SVM model
    model = joblib.load('decision_tree_model.sav')
    tfidf = joblib.load('tfidf_vectorizer.sav')
    preprocessor = TextPreprocessor()

    comment_preprocessed = preprocessor.transform([comment])[0]
    print(comment_preprocessed)

    comment_vectorized = tfidf.transform([comment_preprocessed])
    print(comment_vectorized)

    prediction = model.predict(comment_vectorized)
    print(prediction)

    # Pass the mapped prediction value to the template context
    context = {'prediction': prediction}
    
    # Pass the prediction value to the template context
    

    return render(request, "includes/sentiment.html", context)



"""
def sentimentDL(request):

    comment = request.GET.get('comment', '')
    
    # Load the pre-trained SVM model
    model = joblib.load('decision_tree_model.sav')
    tfidf = joblib.load('tfidf_vectorizer.sav')
    preprocessor = TextPreprocessor()

    comment_preprocessed = preprocessor.transform([comment])[0]
    print(comment_preprocessed)

    comment_vectorized = tfidf.transform([comment_preprocessed])
    print(comment_vectorized)

    prediction = model.predict(comment_vectorized)
    print(prediction)

    # Pass the mapped prediction value to the template context
    context = {'prediction': prediction}
    
    # Pass the prediction value to the template context
    

    return render(request, "includes/sentiment_DL.html", context)

"""



def sentimentDL(request):
    if request.method == 'POST':  # 'post' should be 'POST'
        # Load Model
        with open('model_architecture.json', 'r') as json_file:
            loaded_model_json = json_file.read()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('model_weights.h5')

        # Load Tokenizer
        with open('tokenizer.pkl', 'rb') as tokenizer_file:
            tk = pickle.load(tokenizer_file)

        # Get data from form or request
        comment = request.POST.get('comment', '')  # 'post' should be 'POST'

        def clean_text_single_string(text):
            text = text.lower()
            text = re.sub('\d+', '', text)  # Remove numbers
            text = re.sub(r'@\w+', '', text)  # Remove Twitter account names
            text = re.sub(r'http\S+', '', text)  # Remove website URLs
            text = re.sub(r"[^A-Za-z0-9']+", ' ', text)  # Remove special characters
            text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
            return text.strip()  # Strip leading/trailing whitespaces

        def stem_review_single_string(review):
            lem = WordNetLemmatizer()
            return ' '.join([lem.lemmatize(word) for word in review.split(' ')])

        def normalize_text_single_string(text):
            text_processed = clean_text_single_string(text)
            normalized_text = stem_review_single_string(text_processed.lower())
            return normalized_text

        # Process new single string
        normalized_text = normalize_text_single_string(comment)

        # Tokenize the single string
        normalized_text_seq = tk.texts_to_sequences([normalized_text])

        # Pad the sequence
        max_words = 100  # Same as used during model training
        normalized_text_pad = pad_sequences(normalized_text_seq, maxlen=max_words, padding='post')

        # Make predictions on the transformed data
        predictions = loaded_model.predict(normalized_text_pad)

        predicted_class = np.argmax(predictions)

        sentiment_classes = ['negative', 'neutral', 'positive']
        predicted_sentiment = sentiment_classes[predicted_class]
        return render(request, 'includes/sentiment_DL.html', {'predicted_sentiment': predicted_sentiment})

    return render(request, 'includes/sentiment_DL.html')  































class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
   
    def fit(self, X, y=None):
        return self
   
    def transform(self, X):
        X_transformed = []
        for text in X:
            text = self.remove_username(text)
            text = self.remove_url(text)
            text = self.remove_emoji(text)
            text = self.decontraction(text)
            text = self.separate_alphanumeric(text)
            text = self.unique_char(text)
            text = self.char(text)
            text = text.lower()
            text = self.remove_stopwords(text)
            X_transformed.append(text)
        return X_transformed

    def remove_stopwords(self, text):
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
        return text

    def remove_url(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', text)

    def remove_username(self, text):
        return re.sub('@[^\s]+', '', text)

    def remove_emoji(self, text):
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def decontraction(self, text):
        text = re.sub(r"won\'t", " will not", text)
        text = re.sub(r"won\'t've", " will not have", text)
        text = re.sub(r"can\'t", " can not", text)
        text = re.sub(r"don\'t", " do not", text)
       
        text = re.sub(r"can\'t've", " can not have", text)
        text = re.sub(r"ma\'am", " madam", text)
        text = re.sub(r"let\'s", " let us", text)
        text = re.sub(r"ain\'t", " am not", text)
        text = re.sub(r"shan\'t", " shall not", text)
        text = re.sub(r"sha\n't", " shall not", text)
        text = re.sub(r"o\'clock", " of the clock", text)
        text = re.sub(r"y\'all", " you all", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"n\'t've", " not have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'d've", " would have", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ll've", " will have", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = re.sub(r"\'re", " are", text)
        return text  

    def separate_alphanumeric(self, text):
        words = text
        words = re.findall(r"[^\W\d_]+|\d+", words)
        return " ".join(words)

    def cont_rep_char(self, text):
        tchr = text.group(0)
        if len(tchr) > 1:
            return tchr[0:2]

    def unique_char(self, text):
        substitute = re.sub(r'(\w)\1+', self.cont_rep_char, text)
        return substitute

    def char(self, text):
        substitute = re.sub(r'[^a-zA-Z]', ' ', text)
        return substitute
    









































