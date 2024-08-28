import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EmbeddingShareWeights(nn.Module):
    """
    Calculates input embeddings
    """

    def __init__(self, vocab_size, hidden_size):
        """
        Specify characteristic parameters of embedding layer
        Args:
            vocab_size: Number of tokens in the embedding
            hidden_size: Dimensionality of the embedding.
        """
        super(EmbeddingShareWeights, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.shared_weight = nn.Parameter(torch.randn(vocab_size, hidden_size) * (hidden_size ** -0.5))

    def forward(self, x):
        """
        Get token embeddings of x
        Args:
            x: An int64 tensor with shape [batch, length]
        Returns:
            embeddings: float32. Tensor with shape [batch, length, embedding_size]
            padding: float32. Tensor with shape [batch, length] indicating the locations of the padding tokens in x.
        """
        mask = (x != 0).float()
        embeddings = F.embedding(x, self.shared_weight)
        embeddings *= mask.unsqueeze(-1)

        embeddings *= self.hidden_size ** 0.5  # scale embedding by the sqrt of the hidden size
        return embeddings


class MultiHeadAttention(nn.Module):
    """Multi-headed attention layer"""

    def __init__(self, hidden_size, num_heads, attention_dropout, is_training):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden_size must be evenly divisible by the number of heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.is_training = is_training

        self.q_dense_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_dense_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_dense_layer = nn.Linear(hidden_size, hidden_size, bias=False)

        self.output_dense_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def split_heads(self, x):
        """
        Split x into different heads, and transpose the resulting value
        Args:
            x: A tensor with shape [batch, length, hidden_size]
        Returns:
            A tensor with shape [batch, num_heads, length, hidden_size/num_heads]
        """
        batch_size, length = x.size(0), x.size(1)
        depth = self.hidden_size // self.num_heads
        x = x.view(batch_size, length, self.num_heads, depth)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x):
        """
        Combine tensor that has been split
        Args:
            x: A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
        Returns:
            A tensor with shape [batch, length, hidden_size]
        """
        batch_size, length = x.size(0), x.size(2)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, length, self.hidden_size)

    def forward(self, x, y, bias, cache=None):
        """
        Apply attention mechanism to x and y
        Args:
            x: A tensor with shape [batch, length_x, hidden_size]
            y: A tensor with shape [batch, length_y, hidden_size]
            bias: attention bias that will be added to the result of the dot product.
            cache: (Used during prediction) dictionary with tensor containing results of previous attentions.
        Returns:
            Attention layer output with shape [batch, length_x, hidden_size]
        """
        length = x.size(1)
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        if cache is not None:
            k = torch.cat([cache['k'], k], dim=1)
            v = torch.cat([cache['v'], v], dim=1)
            cache['k'] = k
            cache['v'] = v

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        depth = self.hidden_size // self.num_heads
        q = q / math.sqrt(depth)

        # calculate dot product attention
        logits = torch.matmul(q, k.transpose(-2, -1))

        # print("logits:", logits.shape)
        # print("bias:", bias.shape)
        # logits += bias

        # add mask to prevent future words
        mask = create_look_ahead_mask(length).to(logits.device)
        logits += mask

        attention_weights = F.softmax(logits, dim=-1)

        if self.is_training:
            attention_weights = F.dropout(attention_weights, p=self.attention_dropout)

        attention_output = torch.matmul(attention_weights, v)
        attention_output = self.combine_heads(attention_output)
        attention_output = self.output_dense_layer(attention_output)

        return attention_output


class SelfAttention(MultiHeadAttention):
    """Self-attention layer."""

    def forward(self, x, y, bias, cache=None):
        return super(SelfAttention, self).forward(x, x, bias, cache)


class FeedForwardNetwork(nn.Module):
    """Fully connected feedforward network"""

    def __init__(self, hidden_size, filter_size, relu_dropout, is_training, allow_pad):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.is_training = is_training
        self.allow_pad = allow_pad

        self.filter_dense_layer = nn.Linear(hidden_size, filter_size)
        self.output_dense_layer = nn.Linear(filter_size, hidden_size)

    def forward(self, x, padding=None):
        """
        Return outputs of the feedforward network
        Args:
            x: Tensor with shape [batch_size, length, hidden_size]
            padding: Optional, if set, the padding values are temporarily removed from x.
                The padding values are placed back in the output tensor in the same locations.
                Shape [batch, length]
        Returns:
            Output of the feedforward network
            Shape [batch, length, hidden_size]
        """
        if padding is not None and self.allow_pad:
            nonpad_ids = torch.nonzero(padding.view(-1) == 0).squeeze(1)
            x = x.view(-1, self.hidden_size)
            x = x.index_select(0, nonpad_ids)
            x = x.view(-1, x.size(-1))

        output = self.filter_dense_layer(x)
        output = F.relu(output)
        if self.is_training:
            output = F.dropout(output, p=self.relu_dropout)

        output = self.output_dense_layer(output)

        if padding is not None and self.allow_pad:
            batch_size, length = padding.size()
            output_shape = (batch_size * length, self.hidden_size)
            padded_output = torch.zeros(output_shape, dtype=output.dtype, device=output.device)
            padded_output.index_copy_(0, nonpad_ids, output)
            output = padded_output.view(batch_size, length, self.hidden_size)

        return output


class LayerNormalization(nn.Module):
    """
    Apply layer normalization
    """

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, epsilon=1e-6):
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


def position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """
    Calculate the position encoding as a mix of sine and cosine functions with geometrically
    increasing wavelengths.
    Args:
        length: sequence length
        hidden_size: size of the embedding
        min_timescale: Minimum scale that will be applied at each position
        max_timescale: Maximum scale that will be applied at each position
    Returns:
        Tensor with shape [length, hidden_size]
    """
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    num_timescales = hidden_size // 2
    log_timescale_increment = math.log(max_timescale / min_timescale) / (num_timescales - 1)
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment)
    scaled_time = position * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

    return signal


def get_padding(x, padding_value=0):
    """
    Args:
        x: int tensor with any shape
        padding_value: int value which padding value set
    Returns:
        float tensor with the same shape as x containing value 0,1
        0 means non-padding, 1 means padding
    """
    return (x == padding_value).float()


def get_padding_bias(x):
    """
    Calculate bias tensor from padding values in tensor
    Args:
        x: int tensor with shape [batch_size, length]
    Returns:
        Attention bias tensor of shape [batch_size, 1, 1, length]
    """
    padding = get_padding(x)
    attention_bias = padding * -1e9

    return attention_bias.unsqueeze(1)


def create_look_ahead_mask(length):
    """
    Calculate bias for decoder that maintains model's autoregressive property.
    Args:
        length: int length of sequences in batch.
    Returns:
        float tensor of shape [1, 1, length, length]
    """
    neg_inf = -1e9
    valid_locs = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    decoder_bias = neg_inf * (1.0 - valid_locs)
    return decoder_bias
