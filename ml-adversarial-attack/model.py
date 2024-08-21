import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

class ANNClassifier(nn.Module):
    def __init__(self, train_params):
        super(ANNClassifier, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

        # Training parameters
        self.batch_size = train_params.get("batch_size", 32)
        self.learning_rate = train_params.get("learning_rate", 0.001)
        self.epochs = train_params.get("epochs", 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_model(self, train_loader, val_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            self.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for i, (inputs, labels) in pbar:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                pbar.set_description(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
            epoch_loss = running_loss / len(train_loader.dataset)
            train_accuracy = 100 * correct_train / total_train
            train_losses.append(epoch_loss)

            # Calculate validation loss
            self.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            epoch_val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)
            val_accuracy = 100 * correct_val / total_val

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

        return train_losses, val_losses

def plot_graph(train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

def main():
    # Define training parameters
    train_params = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10
    }

    # Load the MNIST dataset
    X_train = np.load("train data/X_train.npy", allow_pickle=True)
    y_train = np.load("train data/y_train.npy", allow_pickle=True).astype(int)
    X_test = np.load("test data/X_test.npy", allow_pickle=True)
    y_test = np.load("test data/y_test.npy", allow_pickle=True).astype(int)

    # Convert the data into PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_tensor, y_train_tensor, test_size=0.2, random_state=42)

    # Create DataLoader for training, validation, and test sets
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=train_params["batch_size"], shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=train_params["batch_size"])
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=train_params["batch_size"])

    print("\nCreating Model...\n")

    # Initialize the model
    model = ANNClassifier(train_params)

    # Print model summary
    print("Model Summary:")
    summary(model, (784,))

    # Train the model
    print("\nTraining Model...\n")
    train_losses, val_losses = model.train_model(train_loader, val_loader)

    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')

    # Evaluate the model on test set
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    test_accuracy = 100 * correct_test / total_test

    print(f'Test Accuracy: {test_accuracy:.2f}%')

    plot_graph(train_losses, val_losses)

if __name__ == '__main__':
    main()