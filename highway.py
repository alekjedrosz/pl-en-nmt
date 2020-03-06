import torch
import torch.nn.functional as F


class Highway(torch.nn.Module):
    """Class representing a highway network.

    :param e_word: Dimensionality of the input vector(s).
    """

    def __init__(self, e_word):
        super().__init__()
        self.projection_layer = torch.nn.Linear(e_word, e_word)
        self.gate_layer = torch.nn.Linear(e_word, e_word)

    def forward(self, x_conv_out):
        """
        Applies a gated highway layer to the input vector(s).

        :param x_conv_out: Batch of input vectors of shape (batch_size, e_word).
        :return: Batch of output vectors of shape (batch_size, e_word).
        """
        x_proj = F.relu(self.projection_layer(x_conv_out))
        x_gate = torch.sigmoid(self.gate_layer(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out

        return x_highway


if __name__ == '__main__':
    x_conv_out = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).float()
    print(x_conv_out)
    assert x_conv_out.shape == (3, 4)
    model = Highway(4)
    x_highway = model(x_conv_out)
    print(x_highway)
    assert x_highway.shape == (3, 4)