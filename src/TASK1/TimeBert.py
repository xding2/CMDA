import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertModel
from sklearn.preprocessing import LabelEncoder

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
BERT_EMBEDDING_SIZE = 768

# 1. Load and preprocess data
data = pd.read_csv('ecological_health_dataset.csv')
data = data.drop(columns=['Timestamp'])

# Convert non-numeric columns to numeric (if any exist)
for col in data.columns:
    if data[col].dtype == 'object' and col != 'Ecological_Health_Label':
        data[col] = data[col].astype('category').cat.codes

# Separate features and labels
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

# Encode labels as integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Convert to PyTorch tensors
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

# 2. Create Dataset and DataLoader classes
class EcologicalHealthDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Initialize dataset and split into train/test sets
dataset = EcologicalHealthDataset(features, labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Generate double-sized training set for better learning
double_train_dataset = train_dataset + train_dataset
double_train_loader = DataLoader(double_train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 3. Custom BERT Model for Time Series Classification
class BERTForTimeSeriesClassification(nn.Module):
    def __init__(self, num_classes):
        super(BERTForTimeSeriesClassification, self).__init__()
        # Map 14-dimensional input to BERT's 768 dimensions
        self.embedding = nn.Linear(14, BERT_EMBEDDING_SIZE)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(BERT_EMBEDDING_SIZE, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)  # Convert input features to 768 dimensions
        x = x.unsqueeze(1)     # Add dimension to match BERT's input format
        outputs = self.bert(inputs_embeds=x)
        cls_token = outputs.last_hidden_state[:, 0, :]  # Get [CLS] token output
        logits = self.fc(cls_token)
        return logits

# Initialize model, loss function, and optimizer
model = BERTForTimeSeriesClassification(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 4. Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for features, labels in double_train_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(double_train_loader):.4f}')

# 5. Model evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the test set: {100 * correct / total:.2f}%')