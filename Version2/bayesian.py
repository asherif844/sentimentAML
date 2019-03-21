#################################################
# import libraries for AML and Machine Learning
#################################################

import datetime
import os
import pickle
import sys

import azureml.core
import dill
import numpy as np
import pandas as pd
from azureml.core import Experiment, Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.image import ContainerImage
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice, Webservice
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob

print(sys.version)
print("You are currently using version",
      azureml.core.VERSION, "of the Azure ML SDK")

#################################################
# use this cell if you want to use an exisiting workspace
#################################################
subscription_id = 'ba3edd26-e2b1-4cc5-a19e-5bd21d7e9f5d'
resource_group = 'amlservicesbox'
workspace_name = 'bayesian'
workspace_region = 'eastus'

try:
    ws = Workspace(subscription_id=subscription_id,
                   resource_group=resource_group, workspace_name=workspace_name)

    ws.write_config()
    print("Workspace configuration succeeded. Skip the workspace creation steps below")

except:
    print("Workspace not accessible. A new workspace will be created now....")
    ws = Workspace.create(name=workspace_name,
                      subscription_id=subscription_id,
                      resource_group=resource_group,
                      location=workspace_region,
                      create_resource_group=True,
                      exist_ok=True)
    
    ws.write_config()

ws.get_details()    

cpu_cluster_name = "cpuclusterahmed"


# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print("Found existing cpucluster")
except ComputeTargetException:
    print("Creating new cpucluster")

    # Specify the configuration for the new cluster
    compute_config = AmlCompute.provisioning_configuration(vm_size="Standard_D4_v2",
                                                           min_nodes=0,
                                                           max_nodes=8)

    # Create the cluster with the specified name and configuration
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

    # Wait for the cluster to complete, show the output log
    cpu_cluster.wait_for_completion(show_output=True)

ws = Workspace.from_config()

print(ws.name, ws.location, ws.resource_group, sep = '\n')

# choose a name for the run history container in the workspace
experiment_name = 'bayesianClassification'
# project folder
project_folder = './bayesianClassification'    

output = {}
output['SDK version'] = azureml.core.VERSION
output['Subscription ID'] = ws.subscription_id
output['Workspace'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Project Directory'] = project_folder
pd.set_option('display.max_colwidth', -1)
outputDf = pd.DataFrame(data = output, index = [''])
outputDf.T
outputDf.to_csv('output_credentials.csv')

############################################
#       Train the Model
############################################

df = pd.read_csv('Version2/data/train.tsv', sep= '\t')
df.head()
df['Phrase'].apply(lambda x: Textblob(x).sentiment.polarity)


df.head()

cv = CountVectorizer(binary=True)

cv.fit(df['Phrase'])
features = cv.transform(df['Phrase'])
data_labels = df['Sentiment'].values
print(features.shape, data_labels.shape)


x_train,x_test, y_train, y_test = train_test_split(features, data_labels, test_size=0.2, random_state = 12345)
lr = LogisticRegression()

lr.fit(x_train, y_train)

y_predict = lr.predict(x_test)


acc = accuracy_score(y_predict, y_test)

print(f'accuracy score is {acc}')



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
joblib.dump(lr, filename)

tempModel = joblib.load(filename)

tempModel.predict(features[0])
# features[0].shape
# features[2]
data = 'gain the unconditional love she seeks'
data_array = np.array([data])
type(data_array)



vec_data = vectorizer.fit_transform(data_array)
type(vec_data)

from scipy import sparse

range(0,100)
zeros = np.zeros([1,15234], dtype=int)
zeros.shape
values_ = sparse.hstack((vec_data,zeros))



tempModel.predict(values_)[0]



############################################
# Incorporate metrics into Azure
############################################

run = exp.start_logging()                   
run.log("Experiment start time", str(datetime.datetime.now()))
run.log('accuracy score:', acc)

run.log("Experiment end time", str(datetime.datetime.now()))
run.complete()


print(run.get_portal_url())

############################
# Register the model in Azure
############################

model = Model.register(model_path = "Version2/output/sentiment_model.pkl",
                       model_name = "baysSentModel",
                       tags = {"key": "2"},
                       description = "Sentiment Prediction",
                       workspace = ws)

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={"data": "sentiment",  "method" : "bayesianM"}, 
                                               description='Predict Sentiment Score')


salenv = CondaDependencies()
salenv.add_conda_package("scikit-learn")

with open("salenv.yml","w") as f:
    f.write(salenv.serialize_to_string())
with open("salenv.yml","r") as f:
    print(f.read())


############################
# create azure scoring file score.py
############################

%%writefile score.py
import json
import numpy as np
import os
import pickle
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

from azureml.core.model import Model

def init():
    global model
    # retrieve the path to the model file using the model name
    model_path = Model.get_model_path('baysSentModel')
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    # make prediction
    y_hat = model.predict(data)
    return json.dumps(y_hat.tolist())

############################
# end scoring
############################

print(y_test.shape, y_predict.shape)
f1 = vectorizer.fit_transform(df['Phrase'])
f1.shape

f2 = f1.toarray()
f2[0].shape

tempModel.predict([f2[100]])

f2[0]

data = ['this is awesome']
data2 = pd.DataFrame((data))
f3 = vectorizer.fit_transform(np.array(data))
f3.reshape(16947)

pd.DataFrame(vectorizer.fit_transform(df[['Phrase']]))
