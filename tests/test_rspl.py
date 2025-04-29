import unittest
import torch
from rspl import RSPLCell, RSPLRNN

class TestRSPL(unittest.TestCase):
    def test_rspl_cell(self):
        """
        Test the forward pass of RSPLCell to ensure correct output shape.
        """
        input_size = 10
        hidden_size = 20
        batch_size = 32

        cell = RSPLCell(input_size, hidden_size)

        x = torch.randn(batch_size, input_size)
        h_prev = torch.randn(batch_size, hidden_size)

        h_next = cell(x, h_prev)

        self.assertEqual(h_next.shape, (batch_size, hidden_size))

    def test_rspl_rnn(self):
        """
        Test the forward pass of RSPLRNN to ensure correct output and final hidden state shapes.
        """
        input_size = 10
        hidden_size = 20
        batch_size = 32
        seq_len = 5

        rnn = RSPLRNN(input_size, hidden_size)

        X = torch.randn(seq_len, batch_size, input_size)

        outputs, final_h = rnn(X)

        self.assertEqual(outputs.shape, (seq_len, batch_size, hidden_size))
        self.assertEqual(final_h.shape, (batch_size, hidden_size))

if __name__ == '__main__':
    unittest.main()