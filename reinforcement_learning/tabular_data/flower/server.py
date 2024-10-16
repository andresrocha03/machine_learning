import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression


fl.server.start_server(
    server_address="0.0.0.0:8080",  # Listening on all interfaces, port 8080
    config=fl.server.ServerConfig(num_rounds=3),  # Number of training rounds
)
