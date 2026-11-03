import flwr as fl

def main():
    # Define FedAvg strategy with 2 minimum client nodes
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    print("Starting Flower Federated Learning Server on port 8080...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()