# CMDA

# BERT-based Ecological Health Classification / Air Quality Prediction / TBD

## Project Overview
This repository contains two advanced deep learning models for environmental monitoring and prediction:
1. **TimeBERT**: A time-series adaptation of BERT for air quality prediction
2. **EcoHealth-BERT**: A BERT-based classification model for ecological health assessment

## Problem Statement
### Key Challenges
- **Air Quality Forecasting**: Need for accurate long-term predictions of multiple pollutant levels (O3, CO, SO2, NO2)
- **Ecological Health Assessment**: Complex relationship analysis between environmental factors and ecosystem health
- **Multi-dimensional Analysis**: Processing multiple interrelated environmental metrics simultaneously

### Research Questions

1. Can BERT-based models effectively capture seasonal and temporal patterns in air quality data?
2. How can we leverage transformer architectures for environmental time series prediction?
3. TBD

## Models Architecture

### 1. TimeBERT for Air Quality Prediction
- **Base Architecture**: Modified BERT with time-series adaptations
- **Key Features**:
  ```python
  class TimeBERTRegressor(nn.Module):
      def __init__(self, input_dim):
          self.config = BertConfig(
              hidden_size=768,
              num_hidden_layers=4,
              num_attention_heads=8,
              intermediate_size=1024
          )
  ```
- **Innovation**: Custom embedding layer for time-series features
- **Output**: Monthly predictions for multiple years (2024-2026)

### 2. EcoHealth-BERT Classification
- **Base Architecture**: BERT with custom classification head
- **Key Features**:
  ```python
  class BERTForTimeSeriesClassification(nn.Module):
      def __init__(self, num_classes):
          self.embedding = nn.Linear(14, BERT_EMBEDDING_SIZE)
          self.bert = BertModel.from_pretrained("bert-base-uncased")
          self.fc = nn.Linear(BERT_EMBEDDING_SIZE, num_classes)
  ```
- **Innovation**: Adaptation for ecological health classification
- **Output**: Multi-class health status predictions

## Data Sources and Processing

### Air Quality Data
- **Source**: EPA's AirData system
- **Collection Method**: 
  1. Access through EPA's AirData portal
  2. Download Daily Data tool
  3. Parameters: CO, NO2, Ozone (O3), SO2
- **Coverage**: All US states
- **Temporal Resolution**: Daily measurements aggregated to monthly

### Ecological Health Data
- Multiple environmental parameters
- Health classification labels
- Temporal alignment with air quality metrics

## Implementation Highlights

### TimeBERT Features
```python
# Key Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
BERT_EMBEDDING_SIZE = 768
SEQ_LENGTH = 12
```

### Model Performance Metrics
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)

## Project Structure
```
project/
├── models/
│   ├── task1/
│   └── task2/
│   └── task3/
├── data/
│   ├── air_quality/
│   └── ecological_health/
├── predictions/
│   └── {year}/
└── metrics/
```

## Key Innovations
1. **Time Series Adaptation**: Modified BERT architecture for sequential data
2. **Multi-Year Prediction**: Extended forecasting capability
3. **Integrated Analysis**: Combined air quality and ecological health assessment
4. **Scalable Architecture**: Adaptable to different pollutants and metrics

## Results Preview
- **Air Quality Predictions**: Monthly forecasts up to 2026
- **Classification Accuracy**: Ecological health assessment
- **Comprehensive Metrics**: Performance evaluation across multiple dimensions

## Usage Example
```python
# Quick start example
python timebert_regression.py  # For air quality prediction
python ecohealth_classification.py  # For health assessment
```

## Model Performance Highlights
- **TASK1 - TimeBERT Accuracy**: 0.716
- **TASK2 - TimeBERT R2**: 0.698

## Future Work
1. Integration of additional environmental parameters
2. Enhanced seasonal pattern recognition
3. Real-time prediction capabilities
4. Cross-validation with ground truth data

## References
1. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
2. BiTimeBERT: Extending Pre-Trained Language Representations with Bi-Temporal Information


---
*Note: This project is part of ongoing research in environmental monitoring and prediction using advanced deep learning techniques.*