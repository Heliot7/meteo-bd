from typing import List

import torch
import numpy as np
from decimal import Decimal
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader


def get_mean_and_std_per_variable(
    train_data: np.ndarray, num_days: int
) -> [np.ndarray, List[float], List[float]]:
    num_values = int(train_data.shape[1] / num_days)
    clustered_values = [[] for _ in range(num_values)]
    for sample in train_data:
        for idx_value in range(num_values):
            clustered_values[idx_value].extend(
                sample[idx_value * num_days : (idx_value + 1) * num_days].tolist()
            )
    mean = []
    std = []
    for cluster in clustered_values:
        mean.append(np.mean(cluster))
        std.append(np.std(cluster))
    return mean, std


def normalize_input_data_per_channel(
    data: np.ndarray, num_days: int, mean: List[float], std: List[float]
) -> np.ndarray:
    num_values = int(data.shape[1] / num_days)
    for sample in data:
        for idx_value in range(num_values):
            sample[idx_value * num_days : (idx_value + 1) * num_days] = (
                sample[idx_value * num_days : (idx_value + 1) * num_days]
                - mean[idx_value]
            ) / std[idx_value]
    return data


# Define model
class MeteoBDNet(nn.Module):
    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()
        self.seq_modules = nn.Sequential(
            nn.Linear(in_dims, 4),  # Fully Connected: input 15 -> 4 dims
            nn.ReLU(),
            nn.Linear(4, 4),  # Fully Connected 4 -> 4 dims
            nn.ReLU(),
            nn.Linear(4, out_dims),  # Head 4 -> 4 output dims for list_discharges
        )

    def forward(self, x):
        return self.seq_modules(x)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn) -> float:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    pred_total = []
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred_total.append(pred.argmax(1))
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return test_loss


def gogo_network(data: np.ndarray, labels: np.ndarray, num_days: int) -> None:
    # Constants
    split_train_test = 0.8
    batch_size = 25
    epochs = 250

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Dataset preparation (tensors to GPU)
    # - Split train/test data
    latest_train_sample = round(data.shape[0] * split_train_test)
    train_data = data[:latest_train_sample, :]
    train_labels = labels[:latest_train_sample, :]
    test_data = data[latest_train_sample + 1 :, :]
    test_labels = labels[latest_train_sample + 1 :, :]
    # - Normalize training data
    mean, std = get_mean_and_std_per_variable(train_data, num_days)
    train_data = normalize_input_data_per_channel(train_data, num_days, mean, std)
    test_data = normalize_input_data_per_channel(test_data, num_days, mean, std)
    # - Creation of tensors
    tensor_train_data = torch.Tensor(train_data).to(device)
    tensor_train_labels = torch.Tensor(train_labels).to(device)
    tensor_test_data = torch.Tensor(test_data).to(device)
    tensor_test_labels = torch.Tensor(test_labels).to(device)
    train_dataloader = DataLoader(
        TensorDataset(tensor_train_data, tensor_train_labels),
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        TensorDataset(tensor_test_data, tensor_test_labels),
        batch_size=batch_size,
        shuffle=True,
    )

    # Load NN/Model
    in_dims = data.shape[1]
    out_dims = labels.shape[1]
    model = MeteoBDNet(in_dims, out_dims).to(device)
    print(model)

    # Train network
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, verbose=True)
    for t in range(epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {t + 1} with lr {Decimal(current_lr):.2E}\n-------------------------------"
        )
        train(train_dataloader, model, loss_fn, optimizer)
        test_loss = test(test_dataloader, model, loss_fn)
        scheduler.step(test_loss)
    print("Done!")

    torch.save(model.state_dict(), "models/meteo_bd.pth")

    # Test network
    model.eval()
    x, y = tensor_test_data[0], tensor_test_labels[0]
    max_discharge = [label.tolist().index(max(label)) for label in tensor_test_labels]
    percentage_max_discharge = sum([number == 3 for number in max_discharge]) / len(
        max_discharge
    )
    hist_discharges = [sum(i) for i in zip(*tensor_test_labels.tolist())]
    plt.bar(range(4), hist_discharges)
    with torch.no_grad():
        x = x.to(device)
        y_pred = model(x)
        print(f'Predicted: "{y_pred}", Actual: "{y}"')
