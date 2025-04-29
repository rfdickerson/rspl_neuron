import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import os
import logging
from torch.utils.data import DataLoader
from rspl import RSPLRNN
from torch.nn import GRU, LSTM

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_time_series(n_samples, seq_len, num_inputs):
    """
    Generate a synthetic multivariate time series dataset.
    """
    t = np.linspace(0, 10, seq_len)
    data = []
    for _ in range(n_samples):
        features = []
        for i in range(num_inputs):
            trend = 0.1 * t * (i + 1)
            seasonal = 0.5 * np.sin(2 * np.pi * t / (50 - 5 * i))
            noise = np.random.normal(0, 0.1, seq_len)
            features.append(trend + seasonal + noise)
        series = np.stack(features, axis=1)  # (seq_len, num_inputs)
        data.append(series)
    data = np.stack(data, axis=0)  # (n_samples, seq_len, num_inputs)
    return torch.from_numpy(data).float()

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rmsse(y_true, y_pred, scale):
    mse = torch.mean((y_true - y_pred) ** 2, dim=[0, 1])  # mean over batch and seq
    return torch.sqrt(mse / scale).mean().item()

class RNNModel(nn.Module):
    """Wrapper for RNN modules (RSPLRNN, GRU, LSTM) with a linear output layer."""
    def __init__(self, rnn, config, num_outputs):
        super(RNNModel, self).__init__()
        self.rnn = rnn
        self.config = config
        self.linear = nn.Linear(rnn.hidden_size, num_outputs)

    def forward(self, x):
        output, h_n = self.rnn(x)
        pred = self.linear(output)
        return pred, h_n

def train_and_evaluate(model, train_loader, val_loader, test_loader, scale, device, epochs=1000, lr=0.001):
    """Train and evaluate a model, returning training time, RMSSE, and parameter count."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    criterion = nn.MSELoss()
    
    start_time = time.time()
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    max_epochs_without_improvement = 20
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            x = batch[:, :-1].transpose(0, 1)  # (seq_len-1, batch_size, num_inputs)
            y = batch[:, 1:].transpose(0, 1)   # (seq_len-1, batch_size, num_inputs)
            optimizer.zero_grad()
            pred, _ = model(x)  # (seq_len-1, batch_size, num_outputs)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x = batch[:, :-1].transpose(0, 1)
                y = batch[:, 1:].transpose(0, 1)
                pred, _ = model(x)
                val_loss += criterion(pred, y).item()
        
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f'best_model_{model.config}.pt')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= max_epochs_without_improvement and epoch >= 100:
                break
    
    training_time = time.time() - start_time
    
    model.load_state_dict(torch.load(f'best_model_{model.config}.pt'))
    model.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            x = batch[:, :-1].transpose(0, 1)
            y = batch[:, 1:].transpose(0, 1)
            pred, _ = model(x)
            test_predictions.append(pred.reshape(-1, pred.shape[-1]))
            test_targets.append(y.reshape(-1, y.shape[-1]))
    
    test_predictions = torch.cat(test_predictions)
    test_targets = torch.cat(test_targets)
    test_rmsse = rmsse(test_targets, test_predictions, scale)
    
    return {
        'training_time': training_time,
        'test_rmsse': test_rmsse,
        'parameters': count_parameters(model)
    }

def main():
    """Run benchmark comparing RSPLRNN, GRU, and LSTM for configurations C1 and C2."""
    # Configuration parameters from the paper
    configs = {
        'C1': {'num_inputs': 2, 'num_layers': 1, 'hidden_size': 4},
        'C2': {'num_inputs': 4, 'num_layers': 3, 'hidden_size': 8}
    }
    batch_size = 32
    seq_len = 50
    n_samples = 1000
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    epochs = 1000
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = []
    for config_name, config in configs.items():
        logger.info(f"\nRunning benchmark for configuration {config_name}")
        input_size = config['num_inputs']
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        num_outputs = input_size  # Predict all features
        
        # Generate dataset for this config
        dataset = generate_synthetic_time_series(n_samples, seq_len, input_size)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        train_data = dataset[:n_train]
        val_data = dataset[n_train:n_train + n_val]
        test_data = dataset[n_train + n_val:]
        
        # Compute RMSSE scale (average over all features)
        diff = train_data[:, 1:, :] - train_data[:, :-1, :]
        scale = torch.mean(diff ** 2)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        test_loader = DataLoader(test_data, batch_size=batch_size)
        
        # Initialize models
        models = {
            'RSPLRNN': RNNModel(RSPLRNN(input_size, hidden_size, num_layers=num_layers), config_name, num_outputs),
            'GRU': RNNModel(GRU(input_size, hidden_size, num_layers=num_layers, batch_first=False), config_name, num_outputs),
            'LSTM': RNNModel(LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=False), config_name, num_outputs)
        }
        
        # Run benchmark for each model
        for name, model in models.items():
            logger.info(f"Training {name} for {config_name}...")
            result = train_and_evaluate(model, train_loader, val_loader, test_loader, scale, device, epochs, lr)
            result['model'] = name
            result['config'] = config_name
            results.append(result)
    
    # Create results table
    df = pd.DataFrame([
        {
            'Model': r['model'],
            'Config': r['config'],
            'Training Time (s)': r['training_time'],
            'Test RMSSE': r['test_rmsse'],
            'Parameters': r['parameters']
        } for r in results
    ])
    
    # Print results
    logger.info("\nBenchmark Results:")
    print(df.to_string(index=False))
    
    # Save results
    os.makedirs('benchmarks/results', exist_ok=True)
    df.to_csv('benchmarks/results/benchmark_rspl_gru_lstm.csv', index=False)
    logger.info("Results saved to benchmarks/results/benchmark_rspl_gru_lstm.csv")

if __name__ == '__main__':
    main()