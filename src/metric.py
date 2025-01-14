import torch
import torch.nn as nn
import numpy as np

class QuantileRiskLoss(nn.Module):
    def __init__(self, tau, quantiles, device):
        super(QuantileRiskLoss, self).__init__()
        self.quantiles = quantiles
        self.device = device
        self.q_arr = torch.tensor(quantiles).float().unsqueeze(0).unsqueeze(0).repeat(1, tau, 1).to(self.device)

    def forward(self, true, pred):

        true_rep = true.repeat(1, 1, len(self.quantiles)).to(self.device)
        ql = torch.maximum(self.q_arr * (true_rep - pred), (1-self.q_arr)*(pred - true_rep))

        return ql.mean()
    

def calculate_mae(true, pred):
    
    mae = np.mean(np.abs(true - pred))
    return mae

def calculate_rmse(true, pred):
    
    mse = np.mean((true - pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def calculate_mape(true, pred):
    
    true = np.where(true == 0, np.finfo(float).eps, true) # Avoid division by zero
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return mape

def calculate_smape(true, pred):
    
    denominator = (np.abs(true) + np.abs(pred)) / 2
    denominator = np.where(denominator == 0, np.finfo(float).eps, denominator) # Avoid division by zero
    smape = np.mean(np.abs(true - pred) / denominator) * 100
    return smape

def cal_metrics(true, pred):

    performance = dict()
    mae = calculate_mae(true, pred)
    rmse = calculate_rmse(true, pred)
    mape = calculate_mape(true, pred)
    smape = calculate_smape(true, pred)

    performance['MAE'] = mae
    performance['RMSE'] = rmse
    performance['MAPE'] = mape
    performance['SMAPE'] = smape

    return performance 