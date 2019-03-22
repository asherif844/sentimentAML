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
resource_group = 'amlservicesbox2'
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
project_folder = './Version2/bayesianClassification'    
os.makedirs(project_folder, exist_ok=True)

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
outputDf.to_csv(project_folder+'/output_credentials.csv')


############################################
# Incorporate metrics into Azure
############################################

# create an azure ml experiment
exp = Experiment(workspace = ws, name = experiment_name)

run = exp.start_logging()                   
run.log("Experiment start time", str(datetime.datetime.now()))

a = ['This is awesome', 'This is mediocre', 'This is awful']
df = pd.DataFrame(a, columns=['phrase'])
df['sentiment'] = df['phrase'].apply(lambda x: TextBlob(x).sentiment.polarity)

accuracy_collection = []
for i in df['sentiment']:
    accuracy_collection.append(i)

from statistics import mean
acc = mean(accuracy_collection)



run.log('accuracy score:', acc)

run.log("Experiment end time", str(datetime.datetime.now()))
run.complete()


print(run.get_portal_url())

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
lr = LogisticRegression(random_state=0, solver='lbfgs',
                          multi_class='multinomial').fit(X, y)
lr.predict(X[:2, :])

# freeze the model for registration
filename = '/Users/theahmedsherif/conda environments/Microsoft/sentimentAML/Version2/output/lrmodel.pkl'
joblib.dump(lr, filename)

loaded_model = joblib.load(filename)
loaded_model.predict(X[:2,:])

############################
# Register the model in Azure
############################

model = Model.register(model_path = filename,
                       model_name = "textblob",
                       tags = {"key": "2"},
                       description = "Sentiment Prediction",
                       workspace = ws)



aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={"data": "sentiment",  "method" : "textblob"}, 
                                               description='Predict Sentiment Score')


textblobenv = CondaDependencies()
textblobenv.add_conda_package("scikit-learn")
# textblobenv.add_conda_package("textblob")
# textblobenv.add_conda_package("pickle")
# textblobenv.add_conda_package("dill")

with open("textblobenv.yml","w") as f:
    f.write(textblobenv.serialize_to_string())
with open("textblobenv.yml","r") as f:
    print(f.read())

#############################
%%writefile score.py

import json
import numpy as np
import os
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

from textblob import TextBlob

from azureml.core.model import Model

def init():
    model = 1

def run(raw_data):
    data = raw_data
    sentiment = TextBlob(data).sentiment.polarity
    return str(sentiment)
#############################


%%time
image_config = ContainerImage.image_configuration(execution_script="score.py", 
                                                  runtime="python", 
                                                  conda_file="textblobenv.yml")


service = Webservice.deploy_from_model(workspace=ws,
                                       name='textblob-pred',
                                       deployment_config=aciconfig,
                                       models=[model],
                                       image_config=image_config)

service.wait_for_deployment(show_output=True)

