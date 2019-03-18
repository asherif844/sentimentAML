import dill
import numpy as np
import pandas as pd
import json 

sample_data = [(1,'This is awesome'), (2, 'This is average'), (3, 'This is awful')]

df = pd.DataFrame(sample_data, columns = ['no', 'text'])


filename = 'outputs/ta_model.pkl'
model = dill.load(open(filename, 'rb'))

scores = []
for i in df['text']:
    a = model(i)
    scores.append(a)

# raw_data = df['text'][0]
raw_data  = '10'
data = model(raw_data)

data


