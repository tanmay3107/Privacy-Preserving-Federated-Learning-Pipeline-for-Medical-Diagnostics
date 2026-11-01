import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from collections import OrderedDict
from src.model import MedicalCNN, load_partition_data

class MedicalClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, device, use_dp=True):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.use_dp = use_dp

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # Attach Opacus Differential Privacy Engine if enabled
        if self.use_dp:
            privacy_engine = PrivacyEngine()
            # Wrap model, optimizer, and dataloader for DP-SGD training
            self.model, optimizer, self.trainloader = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=optimizer,
                data_loader=self.trainloader,
                epochs=1,
                target_epsilon=3.0,
                target_delta=1e-5,
                max_grad_norm=1.0,
            )

        # Local training loop
        for images, labels in self.trainloader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        return float(loss / len(self.testloader)), total, {"accuracy": float(accuracy)}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedicalCNN().to(device)
    trainloader, testloader = load_partition_data(partition_id=0, num_partitions=2)
    
    client = MedicalClient(model, trainloader, testloader, device, use_dp=True)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)