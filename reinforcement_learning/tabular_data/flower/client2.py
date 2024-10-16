import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
import numpy as np
from flwr.client import NumPyClient
import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import argparse
import utils

warnings.filterwarnings("ignore")


NUM_UNIQUE_LABELS = 2  # Number of unique labels in your dataset
NUM_FEATURES = 15  # Number of features in your dataset

# Load your dataset
df = pd.read_csv('x_one_complete.csv')

# Partitioning the dataset into parts for each client
def partition_data(data, num_partitions):
    return np.array_split(data, num_partitions)

num_clients = 2  # Number of clients

model = LogisticRegression()

# print(np.arange(NUM_UNIQUE_LABELS))
# print(np.zeros((NUM_UNIQUE_LABELS, NUM_FEATURES)))

model.classes_ = np.array([i for i in range(NUM_UNIQUE_LABELS)])
model.coef_ = np.zeros((NUM_UNIQUE_LABELS, NUM_FEATURES))
if model.fit_intercept:
    model.intercept_ = np.zeros((NUM_UNIQUE_LABELS,))
        

class SimpleClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__()
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

    def get_parameters(self,config):
        return utils.get_model_parameters(model)
    
    def set_parameters(self, parameters):
        model.coef_ = parameters[0]
        if model.fit_intercept:
            model.intercept_ = parameters[1]
        

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.fit(self.X_train, self.y_train)

        return utils.get_model_parameters(model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = log_loss(self.y_test, model.predict_proba(self.X_test))
        accuracy = model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}

def create_client(cid: str):
    partitions = partition_data(df, num_clients)
    partition = partitions[int(cid)]  # cid will be the client index
    X = partition.drop('label', axis=1).values
    y = partition['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return SimpleClient(X_train, y_train, X_test, y_test)

if __name__ == "__main__":

    # Parse command line arguments for the partition ID
    # parser = argparse.ArgumentParser(description="Flower client using a specific data partition")
    # parser.add_argument("--partition-id", type=int, required=True, help="Data partition ID")
    # args = parser.parse_args()

    # Assuming the partitioner is already set up elsewhere and loaded here
    fl.client.start_client(server_address="0.0.0.0:8080", client=create_client(1).to_client())