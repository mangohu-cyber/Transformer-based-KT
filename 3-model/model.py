import torch
# import torch.nn as nn
import torch.nn.functional as F
from modules import *


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank."""
    actual_rank = tensor.dim()
    if isinstance(expected_rank, int):
        expected_rank = [expected_rank]
    if actual_rank not in expected_rank:
        raise ValueError(
            f"For the tensor `{name}`, the actual rank `{actual_rank}` (shape = {tensor.shape}) "
            f"is not equal to the expected rank `{expected_rank}`"
        )


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions."""
    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = list(tensor.size())
    return shape


class DKT(nn.Module):
    def __init__(self, params, train=True):
        """
        Initialize layers to build DKT model
        Args:
            params: hyperparameter object defining layer size, dropout value etc.
            train: boolean indicating whether the model is in training mode
        """
        super(DKT, self).__init__()
        self.train_mode = train
        self.params = params
        self.inputs_embedding = EmbeddingShareWeights(2 * params['vocab_size'], params['hidden_size'])
        self.target_idx_embedding = EmbeddingShareWeights(params['vocab_size'], params['hidden_size'])
        self.encoder = EncoderStack(self.params, self.train_mode)
        # 线性输出层
        self.output_layer = nn.Linear(self.params['hidden_size'], self.params['vocab_size'])

    def forward(self, inputs, target_ids):
        """
        Calculate logits or inferred target sequence
        Args:
            inputs: int tensor with shape [batch, input_length], question & reaction encoding
            target_ids: int tensor with shape [batch, input_length], question encoding

        Returns:
            logits: tensor with shape [batch, length, vocab_size]
        """
        input_shape = get_shape_list(inputs, expected_rank=2)
        batch_size, length = input_shape

        # 输出的大小为(batch_size, length)

        target_ids = target_ids.view(batch_size, length)

        inputs_embeddings = self.inputs_embedding(inputs)  # shape = [batch, length, hidden_size]
        target_ids_embeddings = self.target_idx_embedding(target_ids)  # shape = [batch, length, hidden_size]

        length = inputs_embeddings.size(1)
        pos_encoding = position_encoding(length, self.params['hidden_size']).to(inputs.device)

        encoder_key = inputs_embeddings + pos_encoding
        encoder_query = target_ids_embeddings + pos_encoding

        if self.train_mode:
            encoder_key = F.dropout(encoder_key, p=self.params['layer_postprocess_dropout'])
            encoder_query = F.dropout(encoder_query, p=self.params['layer_postprocess_dropout'])

        attention_bias = get_padding_bias(encoder_key)
        inputs_padding = get_padding(encoder_key)

        transformer_output = self.encoder(
            encoder_query=encoder_query,
            encoder_key=encoder_key,
            attention_bias=attention_bias,
            inputs_padding=inputs_padding
        )

        logits = self.output_layer(transformer_output)

        return logits


class PrepostProcessingWrapper(nn.Module):
    """Wrapper class that applies layer pre-processing and post-processing"""

    def __init__(self, layer, params, train=True):
        super(PrepostProcessingWrapper, self).__init__()
        self.layer = layer
        self.postprocess_dropout = params['layer_postprocess_dropout']
        self.train_mode = train
        self.layer_norm = nn.LayerNorm(params['hidden_size'])

    def forward(self, x, *args, **kwargs):
        y = self.layer_norm(x)
        y = self.layer(y, *args, **kwargs)
        if self.train_mode:
            y = F.dropout(y, p=self.postprocess_dropout)
        return x + y


class EncoderStack(nn.Module):
    """Transformer encoder stack"""

    def __init__(self, params, train=True):
        super(EncoderStack, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(params['num_hidden_layers']):
            self_attention_layer = SelfAttention(
                hidden_size=params['hidden_size'],
                num_heads=params['num_heads'],
                attention_dropout=params['attention_dropout'],
                is_training=train
            )
            feed_forward_network = FeedForwardNetwork(
                hidden_size=params['hidden_size'],
                filter_size=params['filter_size'],
                relu_dropout=params['relu_dropout'],
                allow_pad=params['allow_ffn_pad'],
                is_training=train
            )
            self.layers.append(
                nn.ModuleList([
                    PrepostProcessingWrapper(self_attention_layer, params, train),
                    PrepostProcessingWrapper(feed_forward_network, params, train)
                ])
            )

        self.output_normalization = nn.LayerNorm(params['hidden_size'])

    def forward(self, encoder_query, encoder_key, attention_bias, inputs_padding):
        """
        Return the output of the encoder of layer stacks
        Args:
            encoder_query: tensor with shape [batch_size, input_length, hidden_size], query
            encoder_key: tensor with shape [batch_size, input_length, hidden_size], key & value
            attention_bias: bias for encoder self-attention layer [batch, 1, 1, input_length]
            inputs_padding: Padding

        Returns:
            output of encoder layer stack, float32 tensor with shape [batch_size, input_length, hidden_size]
        """
        encoder_inputs = encoder_query

        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            encoder_inputs = self_attention_layer(x=encoder_inputs, y=encoder_key, bias=attention_bias)
            encoder_inputs = feed_forward_network(encoder_inputs, padding=None)

        return self.output_normalization(encoder_inputs)
