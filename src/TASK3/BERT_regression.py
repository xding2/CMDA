"""
Air Quality Route Classification Model
-----------------------------------
A deep learning model using LSTM for classifying air quality patterns into routes.
This model processes sequential air quality data to predict pollution pattern routes.

Author: [Your Name]
Date: [Current Date]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import os
from datetime import datetime
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Global Configuration
# ------------------
class Config:
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 20
    SEQ_LENGTH = 12
    NUM_CLASSES = 4
    HIDDEN_DIM = 128

    @classmethod
    def print_config(cls):
        print(f"Using device: {cls.DEVICE}")

# Dataset Implementation
# --------------------
class TimeSeriesDataset(Dataset):
    """Custom dataset for handling sequential air quality data"""
    
    def __init__(self, features, labels, seq_length=Config.SEQ_LENGTH):
        """
        Args:
            features: Normalized feature matrix
            labels: Route labels (encoded)
            seq_length: Length of input sequences
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        """Returns a sequence of features and its corresponding route label"""
        seq_x = self.features[idx:idx + self.seq_length]
        label = self.labels[idx + self.seq_length]
        return seq_x, label

# Model Architecture
# ----------------
class AQClassifier(nn.Module):
    """LSTM-based classifier for air quality route prediction"""
    
    def __init__(self, input_dim, hidden_dim=Config.HIDDEN_DIM, num_classes=Config.NUM_CLASSES):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Size of LSTM hidden states
            num_classes: Number of route classes
        """
        super(AQClassifier, self).__init__()
        
        # LSTM layer for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Input sequence [batch_size, seq_length, input_dim]
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits

# Data Processing
# -------------
def process_data(file_path):
    """
    Process raw data file into model-ready format
    
    Args:
        file_path: Path to raw data CSV file
    
    Returns:
        scaled_features: Normalized feature matrix
        labels: Encoded route labels
        scaler: Fitted MinMaxScaler
        feature_names: List of feature names
        label_encoder: Fitted LabelEncoder
    """
    # Load and preprocess data
    df = pd.read_csv(file_path)
    
    # Encode route labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['Route'])
    
    # Prepare features
    features = df.drop(['Route', 'DateNum'], axis=1)
    
    # Normalize features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, labels, scaler, features.columns.tolist(), label_encoder

# Training Functions
# ----------------
def train_model(model, train_loader, criterion, optimizer):
    """
    Train the model for one epoch
    
    Returns:
        avg_loss: Average training loss
        accuracy: Training accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        features, labels = features.to(Config.DEVICE), labels.to(Config.DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss, accuracy

def evaluate_model(model, test_loader, criterion, label_encoder):
    """
    Evaluate the model on test data
    
    Returns:
        avg_loss: Average test loss
        report: Classification report
        cm: Confusion matrix
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    class_names = [f"Route {i}" for i in range(Config.NUM_CLASSES)]
    
    report = classification_report(all_labels, all_predictions, 
                                 target_names=class_names, 
                                 output_dict=True)
    cm = confusion_matrix(all_labels, all_predictions)
    
    return avg_loss, report, cm

# Visualization Functions
# ---------------------
def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """Generate and save confusion matrix visualization"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Main Execution
# -------------
def main():
    """Main execution function"""
    # Setup
    Config.print_config()
    os.makedirs('results', exist_ok=True)
    
    # Data preparation
    features, labels, scaler, feature_names, label_encoder = process_data('honorviolateme.csv')
    num_features = features.shape[1]
    
    # Create datasets and dataloaders
    dataset = TimeSeriesDataset(features, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Model initialization
    model = AQClassifier(input_dim=num_features).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training loop
    print("Starting training...")
    best_accuracy = 0
    training_history = []
    
    for epoch in range(Config.EPOCHS):
        avg_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer)
        print(f'Epoch [{epoch+1}/{Config.EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': train_accuracy
        })
        
        # Save best model
        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy
            torch.save(model.state_dict(), 'results/best_model.pth')
    
    # Model evaluation
    test_loss, classification_report_dict, conf_matrix = evaluate_model(
        model, test_loader, criterion, label_encoder)
    
    # Generate visualizations and save results
    class_names = [f"Route {i}" for i in range(Config.NUM_CLASSES)]
    plot_confusion_matrix(conf_matrix, class_names, 'results/confusion_matrix.png')
    
    # Save metrics
    results = {
        'training_history': training_history,
        'test_loss': test_loss,
        'classification_report': classification_report_dict,
        'confusion_matrix': conf_matrix.tolist()
    }
    
    with open('results/metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print final results
    print("\nTraining completed. Results saved in results/ directory.")
    print(f"\nTest Loss: {test_loss:.4f}")
    print("\nClassification Report:")
    print(pd.DataFrame(classification_report_dict).T)

if __name__ == "__main__":
    main()