# Temporal Fusion Transformers (TFT) Implementation

This repository contains an implementation of the **Temporal Fusion Transformers (TFT)**, a model designed for interpretable multi-horizon time series forecasting, as described in the paper:

> Lim, Bryan, et al. "Temporal fusion transformers for interpretable multi-horizon time series forecasting." *International Journal of Forecasting* (2021).

The model has been applied to forecast **building-level electricity consumption** using a dataset of power consumption measurements.

## Repository Structure
```
.
├── src/                     # Source code
│   ├── config/              # Configuration files
│   │   └── config.json      # Model and training configuration
│   ├── dataset/             # Dataset files
│   │   ├── building_info.csv # Metadata about buildings
│   │   ├── train.csv        # Training data
│   │   └── test.csv         # Testing data
│   ├── layer.py             # Custom model layers
│   ├── main.py              # Main script for training and evaluation
│   ├── metric.py            # Metrics for evaluation
│   ├── model.py             # TFT model implementation
│   ├── preprocessing.py     # Data preprocessing scripts
│   └── util.py              # Utility functions
├── .gitignore               # Git ignore file
├── README.md                # This readme file
├── requirements.txt         # Python dependencies
```

## Getting Started

### Prerequisites
- Python 3.8 or later
- GPU (optional but recommended for faster training)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tft-electricity.git
   cd tft-electricity
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training the Model
Run the training script:
```bash
python ./src/main.py
```


