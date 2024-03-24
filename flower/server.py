import flwr as fl
import utils
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":

    #utils.set_initial_params(model, n_classes=3, n_features=4)
    fl.server.start_server(
        server_address="127.8.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
    )