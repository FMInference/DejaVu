import math
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint
from .task_modules import SeqClassification, Seq2SeqClassification
from utils.dist_debug_utils import *


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, head_num):
        super(MultiHeadAttention, self).__init__()
        # in Attention: model_dim=768 (nx=n_embd)
        assert model_dim % head_num == 0
        self.model_dim = model_dim
        self.head_num = head_num
        self.split_size = model_dim // head_num
        self.q_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.scale = math.sqrt(self.split_size)

        # self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(model_dim, model_dim)

    def forward(self, input):
        bs = input.size(0)
        # perform linear operation and split into N heads
        k = self.k_linear(input).view(bs, -1, self.head_num, self.split_size)
        q = self.q_linear(input).view(bs, -1, self.head_num, self.split_size)
        v = self.v_linear(input).view(bs, -1, self.head_num, self.split_size)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        scores = functional.softmax(scores, dim=-1)
        scores = torch.matmul(scores, v)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.model_dim)
        output = self.out(concat)
        return output + input # Put residual connection here.


class TwoLayerMLP(nn.Module):
    def __init__(self, model_dim, feedford_dim):
        super(TwoLayerMLP, self).__init__()
        self.linear1 = nn.Linear(model_dim, feedford_dim)
        self.linear2 = nn.Linear(feedford_dim, model_dim)

    def forward(self, input):
        a1 = functional.relu(self.linear1(input))
        a2 = self.linear2(a1)
        return input + a2


class GPTTransformerLayer(nn.Module):
    def __init__(self, model_dim, head_num, feedforward_dim=2048, use_checkpoint=True, layer_norm_eps=1e-5) -> None:
        super(GPTTransformerLayer, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.attn = MultiHeadAttention(model_dim, head_num)
        # Implementation of Feedforward model
        self.mlp = TwoLayerMLP(model_dim, feedforward_dim)
        self.norm1 = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        # x = x + self.dropout_1(self.attn(x2, x2, x2))
        if self.use_checkpoint:
            # x.requires_grad_(True)
            x = checkpoint(self.attn, x)
        else:
            x = self.attn(x)
        x = self.norm2(x)
        # x = x + self.dropout_2(self.ff(x2))
        if self.use_checkpoint:
            # x.requires_grad_(True)
            x = checkpoint(self.mlp, x)
        else:
            x = self.mlp(x)
        return x


def get_position_id(seq_length, size_input, device):
    return torch.arange(seq_length, device=device).unsqueeze(0).expand(size_input, seq_length)


class GPTEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.
    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, vocab_size, embedding_dim, seq_length, num_token_types=0):
        super(GPTEmbedding, self).__init__()
        # Keep the input dimensions.
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_token_types = num_token_types

        self.vocab_embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=None,
                                                  max_norm=None,  norm_type=2, scale_grad_by_freq=False, sparse=False)
        torch.nn.init.xavier_normal_(self.vocab_embedding.weight)
        self.position_embedding = torch.nn.Embedding(seq_length, embedding_dim)
        torch.nn.init.xavier_normal_(self.position_embedding.weight)
        if num_token_types > 0:
            self.token_type_embedding = torch.nn.Embedding(num_token_types, embedding_dim)
        else:
            self.token_type_embedding = None

    def forward(self, input_ids, position_ids=None, tokentype_ids=None):
        word_embeddings = self.vocab_embedding(input_ids)
        if position_ids is None:
            position_ids = get_position_id(self.seq_length, word_embeddings.shape[0], word_embeddings.device)
        pos_embeddings = self.position_embedding(position_ids)
        embeddings = word_embeddings + pos_embeddings
        if tokentype_ids:
            assert self.token_type_embedding is not None
            embeddings = embeddings + self.token_type_embedding(tokentype_ids)
        return embeddings


class GlueSeqClassificationModel(torch.nn.Module):
    def __init__(self, args, vocab_size, num_classes, use_checkpoint=True):
        super(GlueSeqClassificationModel, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.embedding = GPTEmbedding(vocab_size, args.embedding_dim, args.seq_length)

        module_list = []
        for _ in range(args.num_layers):
            module_list.append(GPTTransformerLayer(args.embedding_dim, args.num_heads, args.embedding_dim*4,
                                                   use_checkpoint=use_checkpoint))
        self.transformers = torch.nn.Sequential(*module_list)
        self.classifier = SeqClassification(args.embedding_dim, num_classes)

    def forward(self, input_ids, position_ids=None):
        input_emb = self.embedding(input_ids, position_ids)
        output_emb = self.transformers(input_emb)
        return self.classifier(output_emb)


class GlueSeq2SeqClassificationModel(torch.nn.Module):
    def __init__(self, args, vocab_size, use_checkpoint=True):
        super(GlueSeq2SeqClassificationModel, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.embedding = GPTEmbedding(vocab_size, args.embedding_dim, args.seq_length)

        module_list = []
        for _ in range(args.num_layers):
            module_list.append(GPTTransformerLayer(args.embedding_dim, args.num_heads, args.embedding_dim*4,
                                                   use_checkpoint=use_checkpoint))
        self.transformers = torch.nn.Sequential(*module_list)
        self.classifier = Seq2SeqClassification(vocab_size, args.embedding_dim)

    def forward(self, input_ids, target_ids, position_ids=None):
        input_emb = self.embedding(input_ids, position_ids)
        output_emb = self.transformers(input_emb)
        return self.classifier(output_emb, target_ids)
