# -*- coding: utf-8 -*-
"""MAIN PROJ PHASE 2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1y7mUl63biWM17pdK8fXbXHV8N4KeLkMc
"""

from google.colab import drive
drive.mount('/content/drive')

# Step 1: Data Preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset and preprocess
data = pd.read_csv('/content/drive/MyDrive/paysim.csv')
data.dropna(inplace=True)

label_encoders = {}
for column in ['type', 'nameOrig', 'nameDest']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

features = data[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type']]
scaler = StandardScaler()
features.iloc[:, 1:6] = scaler.fit_transform(features.iloc[:, 1:6])

!pip install torch-geometric --upgrade  # This ensures you have the latest version

# Step 2: Feature Extraction using GCRNN + HGP
import networkx as nx, torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
# Instead of importing GRU from torch_geometric.nn, import it from torch
from torch.nn import GRU
# Import the from_networkx function
from torch_geometric.utils import from_networkx

G = nx.DiGraph()
for _, row in data.iterrows():
    G.add_edge(row['nameOrig'], row['nameDest'], amount=row['amount'], type=row['type'])

for node in G.nodes():
    G.nodes[node]['x'] = [1]

data_pg = from_networkx(G)
data_pg.x = data_pg.x.type(torch.float)

class GCRNNFeatureExtractor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.gru = GRU(hidden_dim, hidden_dim)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index)).unsqueeze(1)
        h, _ = self.gru(x)
        return h.squeeze(1)

class HGPFeatureExtractor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.pool = TopKPooling(input_dim, ratio=0.5)
    def forward(self, x, edge_index):
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None)
        return global_mean_pool(x, batch)

gcrnn_model = GCRNNFeatureExtractor(data_pg.num_node_features, 64)
hgp_model = HGPFeatureExtractor(64)

# Import torch.nn.functional as F
import torch.nn.functional as F
gcrnn_features = gcrnn_model(data_pg.x, data_pg.edge_index)
hgp_features = hgp_model(gcrnn_features, data_pg.edge_index)

# Step 3: Feature Reduction with PCA
from sklearn.decomposition import PCA
import torch

# Replicate hgp_features for each node
hgp_features_replicated = hgp_features.repeat(gcrnn_features.shape[0], 1)

# Now concatenate and apply PCA
# Detach the tensors from the computation graph before converting to NumPy
pca = PCA(n_components=20)
reduced_features = pca.fit_transform(torch.cat((gcrnn_features.detach(), hgp_features_replicated.detach()), dim=1).cpu().numpy())

# Step 4: Apply ADASYN
from imblearn.over_sampling import ADASYN
adasyn = ADASYN()
x_resampled, y_resampled = adasyn.fit_resample(reduced_features, data['isFraud'])

import matplotlib.pyplot as plt
import seaborn as sns

# Before balancing
plt.figure(figsize=(6, 4))
# Assign 'isFraud' column to the variable 'target'
target = data['isFraud']
sns.countplot(x=target)
plt.title('Class Distribution Before Balancing')
plt.xlabel('Class (0: Non-Fraudulent, 1: Fraudulent)')
plt.ylabel('Count')
plt.show()

# After balancing
plt.figure(figsize=(6, 4))
sns.countplot(x=y_resampled)
plt.title('Class Distribution After Balancing (ADASYN)')
plt.xlabel('Class (0: Non-Fraudulent, 1: Fraudulent)')
plt.ylabel('Count')
plt.show()

!pip install lime

from lime.lime_tabular import LimeTabularExplainer

# Assuming 'reduced_features' from your previous code is the training data
# and 'y_resampled' are the corresponding labels
training_data = reduced_features
# Create the Graph-LIME explainer
explainer = LimeTabularExplainer(training_data,
                                 feature_names=[f"Feature_{i}" for i in range(training_data.shape[1])], #add feature names
                                 class_names=['fraud', 'non-fraud'],
                                 mode='classification')

# Define a prediction function using a trained model
# Replace 'your_trained_model' with your actual trained model
# This function takes an array of instances and returns probabilities for each class
def predict_fn(x):
  # Ensure x is a 2D array, even if it's a single instance
  x = x.reshape(1, -1) if x.ndim == 1 else x
  # Implement your model's prediction logic here. This may be:
  # return your_trained_model.predict_proba(x)  # If your model has predict_proba
  # OR
  # return your_trained_model.predict(x)[:, 1]  # If your model only has predict, get prob for class 1
  # OR other appropriate logic based on your model
  # Here, we are using a dummy prediction for demonstration:
  # This dummy prediction function assigns probabilities randomly to both classes.
  # You should replace this with the prediction logic of your trained model.
  import numpy as np
  probabilities = np.random.rand(x.shape[0], 2)  # Generate random probabilities for 2 classes
  probabilities /= probabilities.sum(axis=1, keepdims=True)  # Normalize probabilities to sum to 1
  return probabilities

# Choose a specific node to explain
# Use the index to select a data point from reduced_features
node_idx = 0
# This will retrieve the selected data point's features:
instance = training_data[node_idx]
explanation = explainer.explain_instance(instance, predict_fn, num_features=10) # Replace with your prediction function

# Visualize the explanation
explanation.show_in_notebook()

from torch_geometric.utils import degree

# Get the node degrees for both source and destination nodes
# Replace 'graph_data' with 'data_pg'
node_degrees = degree(data_pg.edge_index[0], num_nodes=data_pg.x.size(0)) + degree(data_pg.edge_index[1], num_nodes=data_pg.x.size(0))

# Plotting the degree distribution
import matplotlib.pyplot as plt

plt.hist(node_degrees.cpu().numpy(), bins=50, alpha=0.7, color='blue')
plt.title('Node Degree Distribution')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Visualize correlation matrix
corr_matrix = pd.DataFrame(reduced_features).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# PCA explained variance
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio by Each Component: ", explained_variance)

# Plot the cumulative explained variance
cumulative_variance = explained_variance.cumsum()
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Cumulative Explained Variance')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a model (Random Forest for example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

# Get feature importances from RandomForest
importances = model.feature_importances_

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=[f"Feature {i}" for i in range(len(importances))])
plt.title("Feature Importance")
plt.show()

