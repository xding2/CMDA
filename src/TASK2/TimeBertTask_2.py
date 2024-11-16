"""
TimeBERT for Air Quality Regression
---------------------------------
A BERT-based model for predicting air quality metrics using time series data.
Author: [Your Name]
Date: [Current Date]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertConfig
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime
import json

# Global Configuration
# -------------------
class Config:
    # Device setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 20
    BERT_EMBEDDING_SIZE = 768
    SEQ_LENGTH = 12
    PREDICTION_YEARS = [2024, 2025, 2026]
    
    # Directories
    DIRS = ['models', 'predictions', 'metrics']
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories for storing model outputs"""
        for directory in cls.DIRS:
            for year in cls.PREDICTION_YEARS:
                path = f'{directory}/{year}'
                os.makedirs(path, exist_ok=True)

# Dataset Implementation
# --------------------
class TimeSeriesDataset(Dataset):
    """Custom dataset for time series data handling"""
    
    def __init__(self, data, seq_length=Config.SEQ_LENGTH):
        """
        Args:
            data: Input time series data
            seq_length: Length of the sequence for prediction
        """
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq_x = self.data[idx:idx + self.seq_length]
        seq_y = self.data[idx + self.seq_length]
        return seq_x, seq_y

# Model Architecture
# -----------------
class TimeBERTRegressor(nn.Module):
    """BERT-based model for time series regression"""
    
    def __init__(self, input_dim):
        """
        Args:
            input_dim: Dimension of input features
        """
        super(TimeBERTRegressor, self).__init__()
        
        # BERT Configuration
        self.config = BertConfig(
            hidden_size=Config.BERT_EMBEDDING_SIZE,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=Config.SEQ_LENGTH + 2
        )
        
        # Model Layers
        self.input_projection = nn.Linear(input_dim, Config.BERT_EMBEDDING_SIZE)
        self.bert = BertModel(self.config)
        self.fc = nn.Sequential(
            nn.Linear(Config.BERT_EMBEDDING_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        """Forward pass of the model"""
        batch_size = x.size(0)
        x = self.input_projection(x)
        attention_mask = torch.ones((batch_size, Config.SEQ_LENGTH)).to(x.device)
        outputs = self.bert(inputs_embeds=x, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        prediction = self.fc(cls_output)
        return prediction

# Utility Functions
# ----------------
def save_monthly_predictions(predictions_df, pollutant, year):
    """Save predictions in monthly format"""
    monthly_dir = f'predictions/{year}/monthly'
    os.makedirs(monthly_dir, exist_ok=True)
    
    for month in range(1, 13):
        month_data = predictions_df.iloc[month-1:month]
        month_name = month_data.index[0].strftime('%B').lower()
        filename = f'{monthly_dir}/{pollutant}_{year}_{month_name}.csv'
        month_data.to_csv(filename)

def generate_prediction_summary(predictions_dict, pollutant):
    """Generate detailed summary of predictions"""
    summary = {}
    for year, pred_df in predictions_dict.items():
        yearly_stats = {
            'annual': {
                'mean': float(pred_df.values.mean()),
                'std': float(pred_df.values.std()),
                'min': float(pred_df.values.min()),
                'max': float(pred_df.values.max())
            },
            'monthly': {}
        }
        
        for month in range(12):
            month_name = pred_df.index[month].strftime('%B')
            month_data = pred_df.iloc[month]
            yearly_stats['monthly'][month_name] = {
                'mean': float(month_data.mean()),
                'std': float(month_data.std()),
                'min': float(month_data.min()),
                'max': float(month_data.max())
            }
        
        summary[str(year)] = yearly_stats
    
    return summary

def calculate_metrics(model, test_loader, criterion, device):
    """Calculate model performance metrics"""
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    return {
        'test_loss': float(avg_loss),
        'mse': float(np.mean((predictions - actuals) ** 2)),
        'mae': float(np.mean(np.abs(predictions - actuals))),
        'rmse': float(np.sqrt(np.mean((predictions - actuals) ** 2))),
        'mean_absolute_percentage_error': float(np.mean(np.abs((actuals - predictions) / actuals)) * 100)
    }

# Model Training and Prediction
# ---------------------------
def predict_future_months(model, last_sequence, scaler, feature_names, start_date, num_months):
    """Generate predictions for future months"""
    model.eval()
    with torch.no_grad():
        predictions = []
        current_sequence = last_sequence.clone()
        
        for _ in range(num_months):
            prediction = model(current_sequence).cpu().numpy()
            predictions.append(prediction[0])
            
            current_sequence = torch.cat([
                current_sequence[:, 1:, :],
                torch.FloatTensor(prediction).to(Config.DEVICE).unsqueeze(0)
            ], dim=1)

        predictions = np.array(predictions)
        predictions = scaler.inverse_transform(predictions)
        
        future_dates = pd.date_range(
            start=start_date,
            periods=num_months,
            freq='M'
        )
        
        return pd.DataFrame(
            predictions,
            index=future_dates,
            columns=feature_names
        )

def train_and_predict(pollutant):
    """Main training and prediction pipeline"""
    file_path = f'pollutant_csvs/{pollutant}_monthly.csv'
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return None, None

    print(f"\nProcessing {pollutant}...")
    
    # Data preparation
    data = pd.read_csv(file_path)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    feature_names = data.columns
    
    # Dataset creation
    dataset = TimeSeriesDataset(scaled_data)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Model initialization
    model = TimeBERTRegressor(input_dim=scaled_data.shape[1]).to(Config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training loop
    best_loss = float('inf')
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{Config.EPOCHS}], Loss: {avg_loss:.4f}')
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'models/{pollutant}_best_model.pth')

    # Generate predictions
    metrics = calculate_metrics(model, test_loader, criterion, Config.DEVICE)
    last_sequence = torch.FloatTensor(scaled_data[-Config.SEQ_LENGTH:]).unsqueeze(0).to(Config.DEVICE)
    
    predictions = {}
    start_date = pd.Timestamp('2023-10-01')
    
    for year in Config.PREDICTION_YEARS:
        predictions_df = predict_future_months(
            model, last_sequence, scaler, 
            feature_names, start_date, 12
        )
        predictions[year] = predictions_df
        last_sequence = torch.FloatTensor(
            scaler.transform(predictions_df.values)
        ).unsqueeze(0).to(Config.DEVICE)
        start_date = start_date + pd.DateOffset(months=12)
        
        print(f"\nPredictions for {pollutant} ({year}):")
        print(predictions_df.head())
    
    # Save results
    save_predictions_and_metrics(predictions, metrics, pollutant)
    return predictions, metrics

def main():
    """Main execution function"""
    Config.create_directories()
    
    pollutants = ['O3', 'CO', 'SO2', 'NO2']
    all_results = {}
    
    for pollutant in pollutants:
        predictions, metrics = train_and_predict(pollutant)
        if predictions is not None:
            all_results[pollutant] = {
                'metrics': metrics,
                'prediction_years': list(predictions.keys())
            }
    
    # Save final summary
    summary_path = 'predictions/summary_results.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("\nProcessing completed. Results saved in predictions/ and metrics/ directories.")

if __name__ == "__main__":
    main()