import torch


class CNN(torch.nn.Module):
    """Represents a 1d convolutional neural network with max pooling.

    :param in_channels: Number of channels in the input matrices (i.e. the dimensionality of char embeddings e_char).
    :param out_channels: Number of channels in the output matrices / vectors (i.e. the number of filters, should be
    equal to the desired dimensionality of word embeddings e_word).
    :param window_size: Size of the convolving kernel. Determines the length of the output sequence which we max pool
    over.
    """
    def __init__(self, in_channels, out_channels, window_size=5):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, window_size)

    def forward(self, x_reshaped):
        """Applies a convolutional layer to a batch of padded words. Additionally, uses max pooling over
        windows to reduce the output matrix for each item in the batch, to a single output vector.

        :param x_reshaped: Tensor of size (batch_size, char_embedding_size (e_char), max_word_length (m_word)).
        :return: Matrix of size (batch_size, word_embedding_size (f = e_word)).
        """
        x_conv = self.conv1d(x_reshaped)
        x_conv_out, _ = torch.max(x_conv, 2)

        return x_conv_out


if __name__ == '__main__':
    inp = torch.tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]],
                        [[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]]).float()
    print(inp)
    e_char = 3  # Char embedding dimensionality (number of input channels).
    m_word = 2  # Maximum number of characters in a word (i.e. length of each word after padding).
    batch_size = 4  # Number of words in a batch.
    window_size = 1  # Length of the convolving kernel.
    assert inp.shape == (batch_size, e_char, m_word)
    f = 6  # Number of filters, should be equal to the desired dim. of word embeddings e_word (# of output channels).
    model = CNN(e_char, f, window_size=1)
    x_conv_out = model(inp)
    assert x_conv_out.shape == (batch_size, f)