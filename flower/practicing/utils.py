import os
import flwr as fl
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score


def get_model_params(model):
    params = []
    for k,v in model.get_params().items():
        params.append(np.asarray(v))
    return params

def set_model_params(model: DecisionTreeClassifier, params) -> DecisionTreeClassifier:
    names = ['ccp_alpha', 'class_weight', 'criterion', 'max_depth', 
     'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 
     'min_samples_leaf', 'min_samples_split', 'min_weight_fraction_leaf', 
     'random_state', 'splitter']
   
    model.set_params(ccp_alpha=params[0])
    model.set_params(class_weight=params[1])
    model.set_params(criterion=params[2])
    model.set_params(max_depth=params[3])
    model.set_params(max_features=params[4])
    model.set_params(max_leaf_nodes=params[5])
    model.set_params(min_impurity_decrease=params[6])
    model.set_params(min_samples_leaf=params[7])
    model.set_params(min_samples_split=params[8])
    model.set_params(min_weight_fraction_leaf=params[9])
    model.set_params(random_state=params[10])
    model.set_params(splitter=params[11])

    return model

def get_accuracy(model,x_treino,y_treino,x_teste,y_teste):
    model.fit(x_treino,y_treino)
    prediction = model.predict(x_teste).round()
    
    acc  = accuracy_score(y_teste,prediction)
    return acc