import flwr as fl

fl.server.start_server(
    server_address="127.8.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=3),
)