import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import os
import logging
from torch.utils.data import DataLoader, TensorDataset
from rspl import RSPLRNN
from torch.nn import GRU, LSTM

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_time_series(n_series, seq_len, num_inputs):
    """
    Generate synthetic multivariate time series mimicking M5 dataset structure.
    """
    t = np.linspace(0, 10, seq_len)
    data = []
    for _ in range(n_series):
        features = []
        for i in range(num_inputs):
            trend = 0.05 * t * (i + 1)  # Linear trend
            seasonal = 0.3 * np.sin(2 * np.pi * t / (100 - 10 * i))  # Seasonality
            noise = np.random.normal(0, 0.2, seq_len)  # Noise
            features.append(trend + seasonal + noise)
        series = np.stack(features, axis=1)  # (seq_len, num_inputs)
        data.append(series)
    return torch.from_numpy(np.stack(data, axis=0)).float()  # (n_series, seq_len, num_inputs)

def fit_linear_predictor(data, input_steps):
    """
    Fit a simple linear predictor to detrend the time series using normal equation.
    Runs on CPU to avoid MPS issues, ensures outputs are on the correct device.
    """
    device = data.device
    data = data.cpu()  # Move to CPU
    batch_size, seq_len, num_inputs = data.shape
    X = []
    y = []
    for i in range(seq_len - input_steps):
        X.append(data[:, i:i + input_steps].reshape(batch_size, -1))
        y.append(data[:, i + input_steps])
    X = torch.stack(X)  # (seq_len - input_steps, batch_size, input_steps * num_inputs)
    y = torch.stack(y)  # (seq_len - input_steps, batch_size, num_inputs)
    
    # Normal equation: W = (X^T X)^(-1) X^T y
    X_flat = X.reshape(-1, input_steps * num_inputs)  # (N, input_steps * num_inputs)
    y_flat = y.reshape(-1, num_inputs)  # (N, num_inputs)
    XtX = torch.matmul(X_flat.T, X_flat)  # (input_steps * num_inputs, input_steps * num_inputs)
    XtX_inv = torch.inverse(XtX + 1e-5 * torch.eye(XtX.shape[0]))  # Add regularization
    Xty = torch.matmul(X_flat.T, y_flat)  # (input_steps * num_inputs, num_inputs)
    W = torch.matmul(XtX_inv, Xty)  # (input_steps * num_inputs, num_inputs)
    
    # Compute predictions and residuals
    preds = torch.matmul(X, W)  # (seq_len - input_steps, batch_size, num_inputs)
    residuals = y - preds  # (seq_len - input_steps, batch_size, num_inputs)
    
    # Ensure outputs are on the correct device
    logger.info(f"fit_linear_predictor: Moving residuals and W to {device}")
    return residuals.to(device), W.to(device)

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rmsse(y_true, y_pred, train_data):
    """
    Compute RMSSE as per M5 competition definition.
    """
    h = y_true.shape[0]
    mse = torch.mean((y_true - y_pred) ** 2)
    scale = torch.mean((train_data[1:] - train_data[:-1]) ** 2)
    return torch.sqrt(mse / scale).item()

class RNNModel(nn.Module):
    """Wrapper for RNN modules with a linear output layer."""
    def __init__(self, rnn, config, num_outputs):
        super(RNNModel, self).__init__()
        self.rnn = rnn
        self.config = config
        self.linear = nn.Linear(rnn.hidden_size, num_outputs)

    def forward(self, x):
        output, h_n = self.rnn(x)
        pred = self.linear(output)
        return pred, h_n

def train_and_evaluate(model, train_loader, val_loader, test_loader, train_data_raw, linear_W, device, input_steps=2):
    """Train and evaluate model, returning training time, RMSSE, and parameter count."""
    model.to(device)
    linear_W = linear_W.to(device)  # Ensure linear_W is on the correct device
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    criterion = nn.MSELoss()

    start_time = time.time()
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    max_epochs_without_improvement = 20
    
    for epoch in range(10):
        model.train()
        train_loss = 0
        logger.info(f"Epoch {epoch + 1}: Starting training...")
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logger.info(f"Batch {batch_idx}: batch_x shape {batch_x.shape}, batch_y shape {batch_y.shape}")
            x = batch_x.transpose(0, 1)  # (seq_len, batch_size, num_inputs)
            y = batch_y.transpose(0, 1)  # (seq_len, batch_size, num_inputs)
            optimizer.zero_grad()
            logger.info("Computing forward pass...")
            pred, _ = model(x)
            logger.info("Computing loss...")
            loss = criterion(pred, y)
            logger.info("Computing backward pass...")
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        logger.info(f"Epoch {epoch + 1}: Training loss = {train_loss / len(train_loader)}")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                x = batch_x.transpose(0, 1)
                y = batch_y.transpose(0, 1)
                pred, _ = model(x)
                val_loss += criterion(pred, y).item()
        
        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch + 1}: Validation loss = {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f'best_model_{model.config}.pt')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= max_epochs_without_improvement and epoch >= 100:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
    
    training_time = time.time() - start_time
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(f'best_model_{model.config}.pt'))
    model.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            x = batch_x.transpose(0, 1)  # (seq_len, batch_size, num_inputs)
            y = batch_y.transpose(0, 1)  # (seq_len, batch_size, num_inputs)
            pred, _ = model(x)  # (seq_len, batch_size, num_outputs)
            
            # Compute linear predictor for each time step
            seq_len, batch_size, num_inputs = x.shape
            linear_preds = []
            for t in range(seq_len):
                start_idx = max(0, t - input_steps + 1)
                end_idx = t + 1
                input_window = x[start_idx:end_idx].reshape(-1, batch_size, num_inputs)
                if input_window.shape[0] < input_steps:
                    padding = torch.zeros(input_steps - input_window.shape[0], batch_size, num_inputs, device=device)
                    input_window = torch.cat([padding, input_window], dim=0)
                input_flat = input_window.reshape(batch_size, -1)[:, -input_steps * num_inputs:]
                logger.info(f"Test loop, t={t}: input_flat device={input_flat.device}, linear_W device={linear_W.device}")
                linear_pred_t = torch.matmul(input_flat, linear_W)  # (batch_size, num_outputs)
                linear_preds.append(linear_pred_t)
            linear_pred = torch.stack(linear_preds)  # (seq_len, batch_size, num_outputs)
            
            final_pred = pred + linear_pred
            test_predictions.append(final_pred.reshape(-1, final_pred.shape[-1]))
            test_targets.append(y.reshape(-1, y.shape[-1]))
    
    test_predictions = torch.cat(test_predictions)
    test_targets = torch.cat(test_targets)
    test_rmsse = rmsse(test_targets, test_predictions, train_data_raw.reshape(-1, train_data_raw.shape[-1]))
    
    return {
        'training_time': training_time,
        'test_rmsse': test_rmsse,
        'parameters': count_parameters(model)
    }

def main():
    """Run benchmark comparing RSPLRNN, GRU, and LSTM."""
    configs = {
        'C1': {'num_inputs': 2, 'num_layers': 1, 'hidden_size': 4},
        'C2': {'num_inputs': 4, 'num_layers': 3, 'hidden_size': 8}
    }
    n_series = 30  # Mimic 30 M5 time series
    seq_len = 100  # Training points as in M5
    test_len = 10   # Test points as in M5
    val_len = 20    # Validation points as in paper
    train_len = seq_len - val_len
    batch_size = 16  # Reduced to reduce MPS memory pressure
    input_steps = 2  # Number of past steps for linear predictor
    
    # Device selection: Prefer MPS for M1, fall back to CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
        logger.info("Using MPS device for Apple Silicon acceleration")
    else:
        device = torch.device('cpu')
        logger.info("MPS not available, falling back to CPU")
    
    results = []
    for config_name, config in configs.items():
        logger.info(f"\nRunning benchmark for configuration {config_name}")
        input_size = config['num_inputs']
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        num_outputs = input_size
        
        # Generate dataset
        full_data = generate_synthetic_time_series(n_series, seq_len + test_len, input_size).to(device)
        train_val_data = full_data[:, :seq_len]
        test_data_raw = full_data[:, seq_len - input_steps:seq_len + test_len]
        
        # Split into train and validation
        train_data_raw = train_val_data[:, :train_len]
        val_data_raw = train_val_data[:, train_len:]
        
        logger.info(f"train_data_raw shape: {train_data_raw.shape}, device: {train_data_raw.device}")
        logger.info(f"val_data_raw shape: {val_data_raw.shape}, device: {val_data_raw.device}")
        logger.info(f"test_data_raw shape: {test_data_raw.shape}, device: {test_data_raw.device}")
        
        # Fit linear predictor and get residuals
        train_residuals, linear_W = fit_linear_predictor(train_data_raw, input_steps)
        val_residuals, _ = fit_linear_predictor(val_data_raw, input_steps)
        test_residuals, _ = fit_linear_predictor(test_data_raw, input_steps)
        
        logger.info(f"train_residuals shape: {train_residuals.shape}, device: {train_residuals.device}")
        logger.info(f"val_residuals shape: {val_residuals.shape}, device: {val_residuals.device}")
        logger.info(f"test_residuals shape: {test_residuals.shape}, device: {test_residuals.device}")
        logger.info(f"linear_W shape: {linear_W.shape}, device: {linear_W.device}")
        
        # Prepare data loaders
        # Align input data to match residuals
        train_inputs = train_data_raw[:, input_steps:]  # (n_series, train_len - input_steps, num_inputs)
        val_inputs = val_data_raw[:, input_steps:]      # (n_series, val_len - input_steps, num_inputs)
        test_inputs = test_data_raw[:, input_steps:]    # (n_series, test_len, num_inputs)
        
        logger.info(f"train_inputs shape: {train_inputs.shape}, device: {train_inputs.device}")
        logger.info(f"val_inputs shape: {val_inputs.shape}, device: {val_inputs.device}")
        logger.info(f"test_inputs shape: {test_inputs.shape}, device: {test_inputs.device}")
        
        train_dataset = TensorDataset(train_inputs, train_residuals.transpose(0, 1))
        val_dataset = TensorDataset(val_inputs, val_residuals.transpose(0, 1))
        test_dataset = TensorDataset(test_inputs, test_residuals.transpose(0, 1))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize models
        models = {
            'RSPLRNN': RNNModel(RSPLRNN(input_size, hidden_size, num_layers=num_layers), config_name, num_outputs),
            'GRU': RNNModel(GRU(input_size, hidden_size, num_layers=num_layers, batch_first=False), config_name, num_outputs),
            'LSTM': RNNModel(LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=False), config_name, num_outputs)
        }
        
        # Run benchmark
        for name, model in models.items():
            logger.info(f"Training {name} for {config_name}...")
            result = train_and_evaluate(
                model, train_loader, val_loader, test_loader, train_data_raw, linear_W, device, input_steps
            )
            result['model'] = name
            result['config'] = config_name
            results.append(result)
    
    # Create and display results
    df = pd.DataFrame([
        {
            'Model': r['model'],
            'Config': r['config'],
            'Training Time (s)': r['training_time'],
            'Test RMSSE': r['test_rmsse'],
            'Parameters': r['parameters']
        } for r in results
    ])
    
    logger.info("\nBenchmark Results:")
    print(df.to_string(index=False))
    
    os.makedirs('benchmarks/results', exist_ok=True)
    df.to_csv('benchmarks/results/benchmark_rspl_gru_lstm.csv', index=False)
    logger.info("Results saved to benchmarks/results/benchmark_rspl_gru_lstm.csv")

if __name__ == '__main__':
    main()