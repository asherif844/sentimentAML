import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
 
import azureml.core
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core import Experiment
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage
from azureml.core.webservice import AciWebservice
from azureml.core.conda_dependencies import CondaDependencies

# display the core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)



sub_id = 'ba3edd26-e2b1-4cc5-a19e-5bd21d7e9f5d'
# a = '72f988bf-86f1-41af-91ab-2d7cd011db47'
# sub_id = '024d8ccf-00d6-468b-9028-67355d955e25'


ws = Workspace.create(name='text-analytics2',
                      subscription_id=sub_id, 
                      resource_group='taResourceGroup',
                      create_resource_group=True,
                      location='eastus'
                     )

exp = Experiment(workspace=ws, name='text-analytics')
run = exp.start_logging()                   
run.log("Experiment start time", str(datetime.datetime.now()))


sample_data = [(1,'This is awesome'), (2, 'This is average'), (3, 'This is awful')]

df = pd.DataFrame(sample_data, columns = ['no', 'text'])

import dill
filename = '/Users/theahmedsherif/conda environments/Microsoft/sentimentAML/outputs/ta_model.pkl'
model = dill.load(open(filename, 'rb'))

scores = []
for i in df['text']:
    a = model(i)
    scores.append(a)

scores

from statistics import mean
average = mean(scores)
average

run.log('averge: ', average)
run.log("Experiment end time", str(datetime.datetime.now()))
run.complete()


model = Model.register(model_path = filename,
                       model_name = "ta_model",
                       tags = {"key": "1"},
                       description = "TextBlob Prediction",
                       workspace = ws)



ta_env = CondaDependencies()
ta_env.add_conda_package("scikit-learn")
ta_env.add_conda_package("dill")
 
with open("ta_env.yml","w") as f:
    f.write(ta_env.serialize_to_string())
with open("ta_env.yml","r") as f:
    print(f.read())                       


%%time
 
image_config = ContainerImage.image_configuration(execution_script="/Users/theahmedsherif/conda environments/Microsoft/sentimentAML/score.py", 
                                                  runtime="python", 
                                                  conda_file="ta_env.yml")    


aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={"data": "textanalytics",  "method" : "textblob"}, 
                                               description='Predict Sentiment of Sentences')
 
service = Webservice.deploy_from_model(workspace=ws,
                                       name='ta-textblob',
                                       deployment_config=aciconfig,
                                       models=[model],
                                       image_config=image_config)
 
service.wait_for_deployment(show_output=True)



service.scoring_uri


