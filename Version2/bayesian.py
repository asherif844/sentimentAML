# import libraries for AML and Machine Learning

import datetime
import pickle

import azureml.core
import dill
import numpy as np
import pandas as pd
from azureml.core import Experiment, Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.image import ContainerImage
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice, Webservice
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
