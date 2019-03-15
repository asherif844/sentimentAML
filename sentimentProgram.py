from statistics import mean

import dill
import pandas as pd
from textblob import TextBlob

sample_data = [(1,'This is awesome'), (2, 'This is average'), (3, 'This is awful')]

df = pd.DataFrame(sample_data, columns = ['no', 'text'])

df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)


def textblob_model(text):
    a = TextBlob(text).sentiment.polarity
    return a


# tbmodel = dill.dumps(textblob_model)

# serialized_dill_model = dill.loads(tbmodel)

# serialized_dill_model('This is ok')

scores = []
for i in df['text']:
    print(textblob_model(i))
    scores.append(textblob_model(i))

print(f'accuracy: {mean((scores))}')
print(scores)
