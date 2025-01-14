# Temporal Fusion Transformers (TFT) Implementation

This repository contains an implementation of the **Temporal Fusion Transformers (TFT)**, a model designed for interpretable multi-horizon time series forecasting, as described in the paper:

> Lim, Bryan, et al. "Temporal fusion transformers for interpretable multi-horizon time series forecasting." *International Journal of Forecasting* (2021).

The model has been applied to forecast **building-level electricity consumption** using a dataset of power consumption measurements.

## Repository Structure
```
.
├── data/                     # Sample data files or instructions for dataset access
├── models/                   # TFT model implementation
├── utils/                    # Utility functions for preprocessing and evaluation
├── experiments/              # Scripts for training and evaluation
├── outputs/                  # Saved model checkpoints and results
├── README.md                 # This readme file
├── requirements.txt          # Python dependencies
└── LICENSE                   # License file
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
python src/main.py
```


