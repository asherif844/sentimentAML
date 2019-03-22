%%writefile score.py

import json
import numpy as np
import os
import pickle
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

from textblob import TextBlob

from azureml.core.model import Model

def init():
    global model
    # retrieve the path to the model file using the model name
    model_path = Model.get_model_path('textblob')
    model = joblib.load(model_path)

def run(raw_data):
    data = raw_data
    sentiment = TextBlob(data).sentiment.polarity
    return str(sentiment)
