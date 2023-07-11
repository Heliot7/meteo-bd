import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


# Define model
class MeteoBDNet(nn.Module):
    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()
        self.seq_modules = nn.Sequential(
            nn.Linear(in_dims, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, out_dims),
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


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def gogo_network(data: np.ndarray, labels: np.ndarray) -> None:
    # Constants
    split_train_test = 0.8
    batch_size = 1

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Dataset preparation (tensors to GPU)
    # - Split train/test data
    latest_train_sample = round(data.shape[0] * split_train_test)
    train_data = data[:latest_train_sample, :]
    train_labels = labels[:latest_train_sample, :]
    tensor_train_data = torch.Tensor(train_data).to(device)
    tensor_train_labels = torch.Tensor(train_labels).to(device)
    test_data = data[latest_train_sample + 1 :, :]
    test_labels = labels[latest_train_sample + 1 :, :]
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
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), "models/meteo_bd.pth")

    # Test network
    model.eval()
    x, y = tensor_test_data[0], tensor_test_labels[0]
    with torch.no_grad():
        x = x.to(device)
        y_pred = model(x)
        print(f'Predicted: "{y_pred}", Actual: "{y}"')
