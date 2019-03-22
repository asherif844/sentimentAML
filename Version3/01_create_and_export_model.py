import os

import dill
import pickle
import pandas as pd
from azureml.core import Run
# from textblob import TextBlob
from statistics import mean

# let user feed in 2 parameters, the location of the data files (from datastore), and the regularization rate of the logistic regression model

# load train and test set into numpy arrays
sample_data = [(1,'This is awesome'), (2, 'This is average'), (3, 'This is awful')]

df = pd.DataFrame(sample_data, columns = ['no', 'text'])

print('Test a textblob model')

############

def textblob_model(text):
    a = len(text)
    return a
############


# calculate accuracy on the prediction
scores = []
for i in df['text']:
    print(textblob_model(i))
    scores.append(textblob_model(i))

print(f'The average accuracy: {mean((scores))}')
folder = 'Version3/outputs/'
os.makedirs(folder, exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
filename = folder+'ta_model.pkl'
dill.dump(textblob_model, open(filename, 'wb'))
# pickle.dumps(textblob_model, open(filename, 'wb'))