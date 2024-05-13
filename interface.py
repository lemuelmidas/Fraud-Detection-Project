from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
#from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
from pywebio import start_server
from pywebio.exceptions import SessionClosedException
import pandas as pd
import pickle
import warnings
import argparse

#app= Flask(__name__)

warnings.filterwarnings("ignore")

with open('best_model.pkl', 'rb') as f:
    model= pickle.load(f)

with open('columns.pkl', 'rb') as f:
    model_columns= pickle.load(f)

def prediction(prediction_df):
    model = pickle.load(open('best_model.pkl', 'rb'))
    query= pd.DataFrame(prediction_df, index= [0])
    result=list(model.predict(query))
    final_result= round(result[0],3)
