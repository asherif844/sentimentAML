############################################
#       Serialize and Test the model
############################################

experiment_name = 'sentimentClassification'
exp = Experiment(workspace=ws, name=experiment_name)

print(ws.compute_targets, ws.experiments)

data_folder = os.path.join(os.getcwd(), 'Version2/output')
data_folder

os.makedirs(data_folder, exist_ok=True)

filename = 'Version2/output/sentiment_model.pkl'
vectorizerName = 'Version2/output/vectorizer.pkl'
joblib.dump(lr, filename)
joblib.dump(features, vectorizerName)

tempModel = joblib.load(filename)

tempModel.predict(features[0])
# features[0].shape
# features[2]
data = 'No matter what I do this model gives me the same score'
split_data = data.split()
unique = set(split_data)

low = len(unique)
matrix_no = 15240-low
data_array = np.array([data])
type(data_array)

cv2 = joblib.load(vectorizerName)

vec_data = cv2(data_array)
type(vec_data)

from scipy import sparse


zeros = np.zeros([1,matrix_no], dtype=int)
zeros.shape
values_ = sparse.hstack((vec_data,zeros))



tempModel.predict(data_array)[0]