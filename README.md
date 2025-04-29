# RSPL Neuron Library

The `rspl_neuron` library provides a PyTorch implementation of the Recurrent Sigmoid Piecewise Linear (RSPL) neuron, a novel recurrent neural network component designed for time series forecasting. The RSPL neuron, introduced by Victor Sineglazov and Vladyslav Horbatiuk in their 2025 paper, offers simplicity, stability, and efficiency compared to traditional recurrent neurons like LSTM and GRU.

## Features
- **RSPLCell**: A single RSPL neuron for processing one time step.
- **RSPLRNN**: A recurrent neural network module for sequence processing.
- **Stability**: Context normalization ensures stable hidden states with bounded variance.
- **Efficiency**: Uses fewer parameters than LSTM (~50%) and GRU (~67%).
- **Benchmark**: Compares RSPLRNN against PyTorch’s LSTM and GRU on synthetic or real datasets (e.g., M5 competition dataset).
- **Tests**: Verifies correctness of output shapes.

## Citation
The RSPL neuron was developed by Victor Sineglazov and Vladyslav Horbatiuk at the National Aviation University, Kyiv, Ukraine. Their groundbreaking work is detailed in:

> Sineglazov, V., & Horbatiuk, V. (2025). Time Series Forecasting Using Recurrent Neural Networks Based on Recurrent Sigmoid Piecewise Linear Neurons. *Applied Artificial Intelligence*, 39(1), e2490057. https://doi.org/10.1080/08839514.2025.2490057

Please cite their work when using this library. The BibTeX entry is provided below in the [References](#references) section.

## Installation
### Prerequisites
- Python 3.6 or higher
- PyTorch 2.0 or higher (for MPS acceleration and `torch.compile`)
- pandas (for M5 dataset preprocessing)

### Install via pip
1. Clone or download the repository:
   ```bash
   git clone https://github.com/yourusername/rspl_neuron.git
   cd rspl_neuron
   ```
2. Install the package and dependencies:
   ```bash
   pip install .
   ```
   This installs `rspl_neuron`, `torch>=2.0`, and `pandas`.

### Manual Setup
Alternatively, place the `rspl_neuron` directory in your project folder and install dependencies:
```bash
pip install torch>=2.0 pandas
```

## Usage
### Using RSPLRNN
Process a sequence of inputs with `RSPLRNN`:
```python
import torch
from rspl_neuron import RSPLRNN

# Parameters
input_size = 1
hidden_size = 20
batch_size = 32
seq_len = 50

# Create model
rnn = RSPLRNN(input_size, hidden_size)

# Sample input (e.g., time series data)
X = torch.randn(seq_len, batch_size, input_size)

# Forward pass
outputs, final_h = rnn(X)

print(outputs.shape)  # Expected: torch.Size([50, 32, 20])
print(final_h.shape)  # Expected: torch.Size([32, 20])
```

### Using RSPLCell
Process a single time step with `RSPLCell`:
```python
import torch
from rspl_neuron import RSPLCell

# Parameters
input_size = 1
hidden_size = 20
batch_size = 32

# Create cell
cell = RSPLCell(input_size, hidden_size)

# Sample input and initial hidden state
x = torch.randn(batch_size, input_size)
h_prev = torch.zeros(batch_size, hidden_size)

# Forward pass
h_next = cell(x, h_prev)

print(h_next.shape)  # Expected: torch.Size([32, 20])
```

## Testing
Run the test suite to verify the implementation:
```bash
cd rspl_neuron/tests
python test_rspl.py
```
This checks the output shapes of `RSPLCell` and `RSPLRNN`.

## Benchmarking
Compare `RSPLRNN` against PyTorch’s `LSTM` and `GRU` using either a synthetic time series dataset or the M5 competition dataset:
```bash
cd rspl_neuron/benchmarks
python benchmark_rspl_rnn.py
```
The script:
- Supports synthetic data (trend, seasonality, noise) or real data (M5 dataset, 30 aggregated series).
- Preprocesses M5 data by aggregating to store-category level, detrending with a linear predictor, and normalizing.
- Trains models with configurations C1 (2 inputs, 1 layer, hidden size 4) and C2 (4 inputs, 3 layers, hidden size 8).
- Uses mixed precision and `torch.compile` for MPS acceleration on Apple Silicon.
- Reports training time, Test RMSSE (Root Mean Squared Scaled Error), and parameter count.
- Saves results to `benchmarks/results/benchmark_rspl_gru_lstm_m5.csv`.

**M5 Dataset Setup**:
1. Download the M5 dataset from Kaggle (`https://www.kaggle.com/c/m5-forecasting-accuracy/data`) or GitHub (`https://github.com/Mcompetitions/M5-methods`).
2. Place the dataset in `m5_data/raw` (e.g., `sales_train_evaluation.csv`).
3. Update `data_dir` in `benchmark_rspl_rnn.py` to point to your dataset path.

Example output:
```
  Model Config  Training Time (s)  Test RMSSE  Parameters
RSPLRNN     C1           1.23456    0.89380         58
    GRU     C1           1.59354    0.90520        106
   LSTM     C1           1.76543    0.94210        138
RSPLRNN     C2           4.12345    0.74310        740
    GRU     C2           4.73919    0.92240       1236
   LSTM     C2           5.01234    0.94730       1636
```

## Experiments
The original experiments by Victor Sineglazov and Vladyslav Horbatiuk, including their implementation of the RSPL neuron (`RSPForecaster`), can be found at:

**https://github.com/vladislav-horbatiuk/rspl_neuron_experiments/blob/main/lib/models/recurrent_sigmoid_piecewise_forecaster.py**

This repository includes the `RSPForecaster`, `GRUForecaster`, `LSTMForecaster`, and `StackedRCellsForecaster` classes used to evaluate RSPL against GRU and LSTM on the M5 dataset. The `rspl_neuron` library adapts these implementations for broader use, incorporating optimizations like efficient normalization and MPS acceleration.

## References
Please cite the original paper when using this library:

**BibTeX**:
```bibtex
@article{sineglazov2025time,
  author = {Sineglazov, Victor and Horbatiuk, Vladyslav},
  title = {Time Series Forecasting Using Recurrent Neural Networks Based on Recurrent Sigmoid Piecewise Linear Neurons},
  journal = {Applied Artificial Intelligence},
  volume = {39},
  number = {1},
  pages = {e2490057},
  year = {2025},
  publisher = {Taylor \& Francis},
  doi = {10.1080/08839514.2025.2490057},
  url = {https://doi.org/10.1080/08839514.2025.2490057}
}
```

## Acknowledgments
This library is based on the innovative work of Victor Sineglazov and Vladyslav Horbatiuk at the National Aviation University, Kyiv, Ukraine. Their research advances time series forecasting by offering a stable and efficient alternative to traditional recurrent neural networks. Special thanks to Vladyslav Horbatiuk for making the original experiments publicly available.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request on the project repository.

## Contact
For questions or feedback, please open an issue on the project repository or contact the maintainers.

---
*Note*: Update the repository URL in `pyproject.toml` and the `README.md` with the actual GitHub link if hosted online.