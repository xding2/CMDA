# TimeBERT for Air Quality Prediction
A comprehensive guide to collecting air quality data and implementing TimeBERT for regression analysis.

## Data Collection Process

### Source: EPA's AirData System
The data is collected from the EPA's AirData system, which provides access to outdoor air quality data collected from state, local, and tribal air pollution monitoring devices.

### Step-by-Step Data Collection:
1. Navigate to EPA's [AirData Website](https://www.epa.gov/outdoor-air-quality-data)
2. Select "Download Daily Data"
3. Configure the download parameters:
   - **Pollutants**: CO, NO2, Ozone (O3), SO2
   - **Year**: Select target years
   - **Geographic Area**: All US states
   - **Monitor Site**: All available monitors

### Data Processing Steps:
1. Download daily measurements for each pollutant
2. Aggregate data to monthly averages
3. Clean and normalize the data
4. Structure data into time series format

## Project Structure
```
project/
│
├── pollutant_csvs/          # Processed monthly data
│   ├── O3_monthly.csv
│   ├── CO_monthly.csv
│   ├── SO2_monthly.csv
│   └── NO2_monthly.csv
│
├── models/                  # Saved model states
│   └── {year}/
│
├── predictions/            # Model predictions
│   └── {year}/
│       ├── monthly/
│       └── summary_results.json
│
└── metrics/               # Performance metrics
    └── {pollutant}_metrics.json
```

## Code Implementation

### Prerequisites
```bash
pip install torch transformers pandas numpy scikit-learn
```

### Key Components

1. **Dataset Class**
```python
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length=SEQ_LENGTH):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
```

2. **Model Architecture**
```python
class TimeBERTRegressor(nn.Module):
    def __init__(self, input_dim):
        # BERT configuration
        self.config = BertConfig(
            hidden_size=768,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=1024
        )
```

### Usage Guide

1. **Data Preparation**
```python
# Ensure your monthly data is in the correct directory
pollutant_csvs/
    O3_monthly.csv
    CO_monthly.csv
    SO2_monthly.csv
    NO2_monthly.csv
```

2. **Running the Model**
```python
# Run the entire pipeline
python timebert_regression.py
```

3. **Accessing Results**
- Model predictions: `predictions/{year}/{pollutant}_{year}_predictions.csv`
- Monthly breakdowns: `predictions/{year}/monthly/{pollutant}_{year}_{month}.csv`
- Performance metrics: `metrics/{pollutant}_metrics.json`

### Configuration Parameters
```python
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
BERT_EMBEDDING_SIZE = 768
SEQ_LENGTH = 12
PREDICTION_YEARS = [2024, 2025, 2026]
```

### Output Format

1. **Predictions Structure**
```json
{
    "2024": {
        "annual": {
            "mean": float,
            "std": float,
            "min": float,
            "max": float
        },
        "monthly": {
            "January": {
                "mean": float,
                "std": float,
                "min": float,
                "max": float
            },
            ...
        }
    }
}
```

2. **Metrics Structure**
```json
{
    "test_loss": float,
    "mse": float,
    "mae": float,
    "rmse": float,
    "mean_absolute_percentage_error": float
}
```

## Model Features

- **Architecture**: Custom BERT-based model adapted for time series regression
- **Training**: Implements early stopping and learning rate scheduling
- **Prediction**: Generates monthly predictions for multiple years
- **Metrics**: Comprehensive evaluation metrics including MSE, MAE, RMSE
- **Output**: Detailed predictions with statistical summaries

## Best Practices

1. **Data Quality**
   - Ensure consistent monthly data format
   - Handle missing values appropriately
   - Normalize/scale data before training

2. **Model Training**
   - Monitor training loss for convergence
   - Use GPU acceleration when available
   - Adjust batch size based on available memory

3. **Prediction Generation**
   - Generate predictions in monthly increments
   - Save predictions in both detailed and summary formats
   - Include confidence intervals when possible

## Troubleshooting

Common issues and solutions:
1. **Memory Issues**: Reduce batch size or sequence length
2. **Convergence Problems**: Adjust learning rate or number of epochs
3. **Data Format Errors**: Ensure correct CSV structure and date formatting

## Additional Resources

- [EPA AirData Documentation](https://www.epa.gov/outdoor-air-quality-data)
- [BERT Documentation](https://huggingface.co/docs/transformers/model_doc/bert)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)