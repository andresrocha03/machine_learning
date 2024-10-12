import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from flwr.client import NumPyClient
import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Load your dataset
df = pd.read_csv('x_one_complete.csv')

# Partitioning the dataset into parts for each client
def partition_data(data, num_partitions):
    return np.array_split(data, num_partitions)

num_clients = 2  # Number of clients


class SimpleClient(NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.model = LogisticRegression()
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

    def get_parameters(self):
        return self.model.coef_.flatten().tolist()

    def fit(self, parameters, config):
        self.model.coef_ = np.array(parameters).reshape(1, -1)
        self.model.fit(self.X_train, self.y_train)
        return self.model.coef_.flatten().tolist(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.coef_ = np.array(parameters).reshape(1, -1)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}

def client_fn(cid: str):
    # This function is called by the server to create a client
    partition = partitions[int(cid)]  # cid will be the client index
    X = partition.drop('target', axis=1).values
    y = partition['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return SimpleClient(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    # Load and partition data here or retrieve partitions created elsewhere
    partitions = partition_data(df, num_clients)
    for i in range(len(partitions)):
        fl.client.start_client(server_address="localhost:8080", client=(partitions[i]))