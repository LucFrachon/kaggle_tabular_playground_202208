"""
Deploy the model using a REST API
"""
import os, joblib
import pandas as pd
from flask import Flask

import config.config as cfg
import preprocessors as pp
import predict as pr
import hp_optimization as hpo


app = Flask(__name__)

@app.route("/test")
def test():
    return "<p>App running</p>"


@app.route("/predict")
def predict():
    pass