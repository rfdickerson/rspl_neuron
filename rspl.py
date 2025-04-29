import torch
import torch.nn as nn

class RSPLCell(nn.Module):
    """
    Recurrent Sigmoid Piecewise Linear (RSPL) Cell.

    This cell implements the RSPL neuron as described in the paper
    "Time Series Forecasting Using Recurrent Neural Networks Based on Recurrent Sigmoid Piecewise Linear Neurons"
    by Victor Sineglazov and Vladyslav Horbatiuk.

    Parameters:
    - input_size (int): The size of the input vector.
    - hidden_size (int): The size of the hidden state vector.
    """
    def __init__(self, input_size, hidden_size):
        super(RSPLCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Define weight matrices
        self.S = nn.Parameter(torch.Tensor(hidden_size + input_size, hidden_size))
        self.W_c = nn.Parameter(torch.Tensor(hidden_size + input_size, hidden_size))
        
        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights using Xavier uniform initialization
        nn.init.xavier_uniform_(self.S)
        nn.init.xavier_uniform_(self.W_c)

    def forward(self, x, h_prev):
        """
        Forward pass of the RSPL cell.

        Parameters:
        - x (Tensor): Input tensor of shape (batch_size, input_size)
        - h_prev (Tensor): Previous hidden state tensor of shape (batch_size, hidden_size)

        Returns:
        - h_t (Tensor): New hidden state tensor of shape (batch_size, hidden_size)
        """
        # Step 1: Concatenate h_prev and x
        p_t = torch.cat((h_prev, x), dim=1)  # (batch_size, hidden_size + input_size)
        
        # Step 2: Compute sigmoid gate z_t
        z_t = torch.sigmoid(torch.matmul(p_t, self.S))  # (batch_size, hidden_size)
        
        # Step 3: Compute candidate vector \tilde{h}_t
        h_tilde = torch.matmul(p_t, self.W_c)  # (batch_size, hidden_size)
        
        # Step 4: Compute update vector u_t
        u_t = (1 - z_t) * h_prev + z_t * h_tilde  # (batch_size, hidden_size)
        
        # Step 5: Normalize u_t to get h_t
        norm_u_t = torch.norm(u_t, p=2, dim=1, keepdim=True)  # (batch_size, 1)
        h_t = u_t / (norm_u_t + 1e-8)  # Add small epsilon to avoid division by zero
        
        return h_t

class RSPLRNN(nn.Module):
    """
    Recurrent Neural Network using RSPL cells, supporting multiple layers.

    This module processes a sequence of inputs using stacked RSPLCell layers.

    Parameters:
    - input_size (int): The size of the input vector.
    - hidden_size (int): The size of the hidden state vector.
    - num_layers (int, optional): Number of recurrent layers. Default: 1.
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RSPLRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create a list of RSPLCell layers
        self.cells = nn.ModuleList()
        # First layer takes input_size
        self.cells.append(RSPLCell(input_size, hidden_size))
        # Subsequent layers take hidden_size as input
        for _ in range(num_layers - 1):
            self.cells.append(RSPLCell(hidden_size, hidden_size))

    def forward(self, X, h0=None):
        seq_len, batch_size, _ = X.size()
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=X.device)
        h_t = h0.clone()  # Avoid in-place modification
        outputs = []
        for t in range(seq_len):
            x_t = X[t]
            new_h_t = []
            for l in range(self.num_layers):
                h_prev = h_t[l]
                x_t = self.cells[l](x_t, h_prev)
                new_h_t.append(x_t)
            h_t = torch.stack(new_h_t, dim=0)
            outputs.append(x_t)
        outputs = torch.stack(outputs, dim=0)
        return outputs, h_t