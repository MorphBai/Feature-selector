from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor, dropout
import numpy as np
from transformers.models.bert.modeling_bert import BertEncoder, BertPreTrainedModel
from model.FFC import  FFCResnetBlock

class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(Attention, self).__init__()
        self.attention_dim = attention_dim
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        scores = torch.matmul(query, key.transpose(-2, -1))
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, value)

        return output

class PatchEmbedding(nn.Module):
    """
    ECG sequence to Patch Embedding.
    """

    def __init__(self, seq_length=1000, patch_length=100, embed_dim=768):
        super().__init__()
        num_patches = (seq_length // patch_length)
        self.seq_length = seq_length
        self.patch_length = patch_length
        self.num_patches = num_patches

        self.projection = nn.Conv1d(
            1, embed_dim, kernel_size=patch_length, stride=patch_length)

    def forward(self, ecg_seq):
        batch_size, lead_num, seq_length = ecg_seq.shape
        if seq_length != self.seq_length:
            raise ValueError(
                f"Input ecg seq length {seq_length} doesn't match model {self.seq_length}.")

        x = self.projection(ecg_seq).transpose(1, 2)
        return x


class ECGEmbedding(nn.Module):
    def __init__(self, config, lead_num=12, seq_length=1000, patch_length=25, embed_dim=768):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embeddings = nn.ModuleList([PatchEmbedding(seq_length=seq_length,
                                                              patch_length=patch_length,
                                                              embed_dim=embed_dim) for i in range(lead_num)])
        self.token_type_embeddings = nn.Embedding(lead_num, embed_dim)
        self.position_embeddings = nn.Embedding(
            lead_num * seq_length // patch_length + 1, embed_dim)

        self.register_buffer("position_ids",
                             torch.arange(lead_num * seq_length // patch_length + 1).expand((1, -1)))

        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        for i in range(len(self.patch_embeddings)):
            w = self.patch_embeddings[i].projection.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(
            self.cls_token, std=self.config.initializer_range)

    def forward(self, ecg_seq):
        cls_tokens = self.cls_token.expand(ecg_seq.shape[0], -1, -1)
        seq_embeddings = [cls_tokens]
        token_type_ids = [0]
        for i in range(ecg_seq.shape[1]):
            seq = ecg_seq[:, i:i+1, :]
            seq_embedding = self.patch_embeddings[i](seq)
            seq_embeddings.append(seq_embedding)
            token_type_ids += [i] * seq_embedding.shape[1]

        seq_embeddings = torch.cat(seq_embeddings, dim=1)
        token_type_ids = torch.tensor(
            token_type_ids, dtype=torch.long, device=ecg_seq.device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # TODO 判断位置编码 token编码是否需要batch 维度  pos token 有问题 需要修改
        position_embeddings = self.position_embeddings(self.position_ids)
        embeddings = seq_embeddings  # + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Selector(BertPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.embeddings = ECGEmbedding(config)
        self.encoder = BertEncoder(config)
        self.attention = Attention(config.hidden_size, config.hidden_size)  # 添加注意力层
        self.ffcnet = FFCResnetBlock()  # 添加FFC层
        self.classifier = nn.Linear(768, config.num_labels)

    def forward(self, features, labels):
        ecg_embeddings = self.embeddings(features)
        sequence_outputs = self.encoder(
            ecg_embeddings, return_dict=True).last_hidden_state
        attention_outputs = self.attention(sequence_outputs)  # 应用注意力层
        ffc_outputs = self.ffcnet(attention_outputs)    # 应用FFC层
        logits = self.classifier(ffc_outputs[:, 0, :])
        logits = torch.sigmoid(logits)

        if labels is not None:
            loss = F.binary_cross_entropy(logits, labels)
            return logits, loss
        else:
            return logits
