import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Example data (replace this with your actual data)
X_train = torch.tensor(np.random.rand(1000, num_features), dtype=torch.float32)  # num_features is the number of features per data point
y_train = torch.tensor(np.random.randint(2, size=(1000,)), dtype=torch.float32)    # Assuming binary labels (0 or 1)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Input layer with 64 neurons
        self.fc2 = nn.Linear(64, 32)          # Hidden layer with 32 neurons
        self.fc3 = nn.Linear(32, 1)           # Output layer with 1 neuron

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation for the first layer
        x = torch.relu(self.fc2(x))  # ReLU activation for the second layer
        x = torch.sigmoid(self.fc3(x))  # Sigmoid activation for binary classification
        return x

# Instantiate the model
model = NeuralNetwork(input_size=num_features)

criterion = nn.BCELoss()                           # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i+batch_size]
        targets = y_train[i:i+batch_size]

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))  # BCELoss expects targets to have the same shape as outputs

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Example evaluation (replace this with your own evaluation data)
X_val = torch.tensor(np.random.rand(200, num_features), dtype=torch.float32)
y_val = torch.tensor(np.random.randint(2, size=(200,)), dtype=torch.float32)

with torch.no_grad():
    outputs = model(X_val)
    val_loss = criterion(outputs, y_val.unsqueeze(1))
    predicted_labels = (outputs > 0.5).float()
    val_accuracy = (predicted_labels == y_val.unsqueeze(1)).float().mean()

print("Validation Loss:", val_loss.item())
print("Validation Accuracy:", val_accuracy.item())

# Example prediction (replace this with your actual prediction data)
X_test = torch.tensor(np.random.rand(50, num_features), dtype=torch.float32)
with torch.no_grad():
    predictions = (model(X_test) > 0.5).float()  # Convert probabilities to binary predictions
