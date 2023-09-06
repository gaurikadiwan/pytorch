import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the LFW dataset using scikit-learn's fetch_lfw_people
from sklearn.datasets import fetch_lfw_people

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, h, w = lfw_people.images.shape
X = lfw_people.data
n_features = X.shape[1]
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Perform PCA using PyTorch
n_components = 150
X_train = X_train - X_train.mean(axis=0)
X_test = X_test - X_train.mean(axis=0)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

# Compute the SVD
U, S, V = torch.svd(X_train)
components = V[:, :n_components]
eigenfaces = components.reshape(n_components, h, w)
X_transformed = torch.mm(X_train, components)
X_test_transformed = torch.mm(X_test, components)

# Plot the eigenfaces
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].view(h, w).numpy(), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()

# Calculate explained variance and cumulative explained variance ratio
explained_variance = (S ** 2) / (n_samples - 1)
total_var = explained_variance.sum()
explained_variance_ratio = explained_variance / total_var
ratio_cumsum = torch.cumsum(explained_variance_ratio, dim=0).numpy()

# Plot the explained variance ratio
eigenvalueCount = np.arange(n_components)
plt.plot(eigenvalueCount, ratio_cumsum[:n_components])
plt.title('Compactness')
plt.show()

# Create a Random Forest classifier as a PyTorch module
class RandomForestClassifier(nn.Module):
    def __init__(self, n_estimators, max_depth, max_features):
        super(RandomForestClassifier, self).__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = nn.ModuleList([self.build_tree() for _ in range(self.n_estimators)])


    def build_tree(self):
        return nn.Sequential(
            nn.Linear(self.max_features, self.max_features),
            nn.ReLU(),
            nn.Linear(self.max_features, self.max_features),
            nn.ReLU(),
            nn.Linear(self.max_features, n_classes)
        )

    def forward(self, x):
        predictions = torch.stack([tree(x) for tree in self.trees], dim=0)
        return torch.mean(predictions, dim=0)

# PCA and Random Forest setup
rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=10, max_features=n_components)

# Convert data to GPU if available
if torch.cuda.is_available():
    X_train = X_train.cuda()
    X_test = X_test.cuda()
    y_train = torch.tensor(y_train, dtype=torch.int64).cuda()
    y_test = torch.tensor(y_test, dtype=torch.int64).cuda()
    rf_classifier.cuda()

# Training Random Forest
optimizer = optim.Adam(rf_classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    optimizer.zero_grad()
    outputs = rf_classifier(X_transformed)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64)  # Convert y_train to a PyTorch tensor
    if torch.cuda.is_available():
        y_train_tensor = y_train_tensor.cuda()
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Testing Random Forest
rf_classifier.eval()
with torch.no_grad():
    predictions = torch.argmax(rf_classifier(X_test_transformed), dim=1).cpu().numpy()
    correct = predictions == y_test
    total_test = len(X_test_transformed)
    accuracy = np.sum(correct) / total_test

print("Total Testing", total_test)
print("Predictions", predictions)
print("Which Correct:", correct)
print("Total Correct:", np.sum(correct))
print("Accuracy:", accuracy)

# Generate classification report using scikit-learn
classification_rep = classification_report(y_test, predictions, target_names=target_names)
print(classification_rep)
