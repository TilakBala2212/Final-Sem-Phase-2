#Loading the Dataset
import pandas as pd

# Load the Paysim dataset
df = pd.read_csv("paysim.csv")

# Display first few rows
print(df.head())

#Data Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encode categorical columns
categorical_cols = ['type', 'nameOrig', 'nameDest']  
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Normalize numerical features
scaler = StandardScaler()
numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Define fraud labels
df['isFraud'] = df['isFraud'].astype(int)  

#Feature Selection (Correlation Analysis)
import seaborn as sns
import matplotlib.pyplot as plt

# Plot correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.show()

#Graph Construction using NetworkX
import networkx as nx

# Create a graph
G = nx.from_pandas_edgelist(df, source='nameOrig', target='nameDest', edge_attr=['amount', 'isFraud'])

# Visualize the graph (optional)
plt.figure(figsize=(10, 6))
nx.draw(G, node_size=10, alpha=0.6, edge_color='gray')
plt.show()

#Convert Graph to PyTorch Geometric Format
import torch
from torch_geometric.data import Data

# Convert NetworkX graph to PyTorch Geometric Data
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
node_features = torch.tensor(df[numerical_cols].values, dtype=torch.float)
labels = torch.tensor(df['isFraud'].values, dtype=torch.long)

# Create PyG Data object
data = Data(x=node_features, edge_index=edge_index, y=labels)
print(data)

#Define the GCRNN Model with HGP (Hierarchical Graph Pooling)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GRU, HGPSLPool

class GCRNN_HGP(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super(GCRNN_HGP, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.gru = GRU(hidden_dim, hidden_dim)
        self.pool = HGPSLPool(hidden_dim, ratio=0.5)
        self.fc = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x, _ = self.gru(x)
        x, edge_index, _, _ = self.pool(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

#Train/Test Split
from sklearn.model_selection import train_test_split

# Split dataset into train and test
train_mask, test_mask = train_test_split(range(len(data.y)), test_size=0.2, stratify=data.y)

# Convert to tensor
train_mask = torch.tensor(train_mask, dtype=torch.long)
test_mask = torch.tensor(test_mask, dtype=torch.long)
#Train the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GCRNN_HGP(in_channels=data.x.shape[1], hidden_dim=64, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# Move data to device
data = data.to(device)

# Training loop
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

#Evaluate Model
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    acc = (pred[test_mask] == data.y[test_mask]).sum().item() / len(test_mask)
    print(f'Test Accuracy: {acc:.4f}')
#Explainability using SHAP, GNNExplainer & Graph-LIME
#SHAP
import shap

explainer = shap.Explainer(model)
shap_values = explainer(data.x)
shap.summary_plot(shap_values, data.x.cpu().numpy())

#GNNExplainer
from torch_geometric.nn import GNNExplainer

explainer = GNNExplainer(model, epochs=200)
node_idx = 0  # Example node
node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.x, data.edge_index)

#Graph-LIME
from torch_geometric.nn import GraphLIME

explainer = GraphLIME(model)
explained_features = explainer(data.x, data.edge_index)

#Hyperparameter Tuning using Optuna
import optuna

def objective(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)

    model = GCRNN_HGP(in_channels=data.x.shape[1], hidden_dim=hidden_dim, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training...
    return final_accuracy  # Return evaluation metric

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
