import os
import flwr as fl
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def get_model_params(model):
    params = []
    for k,v in model.get_params().items():
        params.append(np.asarray(v))
    return params

def set_model_params(model: DecisionTreeClassifier, params) -> DecisionTreeClassifier:
    for key, v in zip(model.get_params().keys(), params):
        model.set_params(key=v)
    return model