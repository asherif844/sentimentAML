import pandas as pd 
import dill


sample_data = [(1,'This is awesome'), (2, 'This is average'), (3, 'This is awful'), (4, 'This is spectacular')]

df = pd.DataFrame(sample_data, columns = ['no', 'text'])


# load the model from disk
filename = 'Version3/outputs/ta_model.pkl'
loaded_model = dill.load(open(filename, 'rb'))

for i in df['text']:
    print(loaded_model(i))