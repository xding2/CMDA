# BERT-based Ecological Health Classification Model

This guide explains how to use the BERT-based model for ecological health classification. The model uses a pre-trained BERT architecture adapted for time series classification tasks.

## Prerequisites

Install the required packages:
```bash
pip install pandas torch transformers sklearn
```

## Data Format

Your input file (`ecological_health_dataset.csv`) should contain:
- Numerical features (14 columns)
- A label column named 'Ecological_Health_Label'
- Optional: Timestamp column (will be dropped during processing)

## Model Architecture

The model consists of three main components:
1. Input embedding layer: Maps 14-dimensional features to BERT's 768-dimensional space
2. BERT layer: Processes the embedded features
3. Classification layer: Maps BERT output to class predictions

## Hyperparameters

The default hyperparameters are:
- Batch size: 32
- Learning rate: 0.001
- Training epochs: 20
- BERT embedding size: 768

## Usage Steps

1. **Prepare Your Data**
   - Ensure your CSV file follows the required format
   - The code automatically handles categorical variable encoding

2. **Model Training**
   - The dataset is split 80/20 for training/testing
   - Training data is doubled to improve learning
   - GPU acceleration is used if available

3. **Model Evaluation**
   - The model prints training loss after each epoch
   - Final accuracy is reported on the test set

## Example Usage

```python
# Load and preprocess your data
data = pd.read_csv('your_dataset.csv')

# Train the model
# The code will automatically:
# - Split the data
# - Train for 20 epochs
# - Print training progress
# - Evaluate on test set

# The final line will print test accuracy:
# "Accuracy on the test set: XX.XX%"
```

## Customization

To modify the model:
1. Adjust hyperparameters at the top of the script
2. Change the number of input features in the `BERTForTimeSeriesClassification` class
3. Modify the number of output classes in the model initialization

## Notes

- The model uses BERT's pre-trained weights from "bert-base-uncased"
- Training time will vary based on dataset size and hardware
- GPU is recommended for faster training