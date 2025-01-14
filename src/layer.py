import torch
import torch.nn as nn

"""## 레이어 구축

사용되는 레이어는 다음와 같음
- CateFeatureEmbedding
- ContiFeatureEmbedding
- GatedLinearUnit
- GatedResidualNetwork
- VariableSelectionNetwork
- GateAddNorm
- MultiHeadAttention
- QuantileOutput

### 3-1. FeatureEmbedding
- CateFeatureEmbedding: 범주형 변수를 임베딩하는 레이어
  - input: categorical variables
  - output: categorical embeddings
- ContiFeatureEmbedding: 연속형 변수를 임베딩하는 레이어
  - input: categorical variables
  - output: categorical embeddings
"""

class CateFeatureEmbedding(nn.Module):
    def __init__(self, d_embed, cate_list):
        super(CateFeatureEmbedding, self).__init__()

        self.embedding = nn.ModuleList([nn.Embedding(num, d_embed) for num in cate_list])

    def forward(self, input):

        cate_embed_list = []

        for idx, emb in enumerate(self.embedding):
            cate_embed = emb(input[:, :, idx:idx+1]) # (batch_size,sequence_len,1,d_model)
            cate_embed_list.append(cate_embed)

        return torch.cat(cate_embed_list, dim=2) # (batch_size,sequence_len,num_feature,d_model)

class ContiFeatureEmbedding(nn.Module):
    def __init__(self, d_embed, conti_num):
        super(ContiFeatureEmbedding, self).__init__()

        self.embedding = nn.ModuleList([nn.Linear(1, d_embed) for _ in range(conti_num)])

    def forward(self, input):

        continuous_output = []

        for idx, emb in enumerate(self.embedding):
            output = emb(input[:, :, idx:idx+1]).unsqueeze(-2) # (batch_size,sequence_len,1,d_model)
            continuous_output.append(output)

        return torch.cat(continuous_output, dim=2) # (batch_size,sequence_len,num_feature,d_model)


class GatedLinearUnit(nn.Module):
    def __init__(self, d_model, output_dim):
        super(GatedLinearUnit, self).__init__()

        self.d_model = d_model
        self.output_dim = output_dim

        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(self.d_model, self.output_dim)
        self.linear2 = nn.Linear(self.d_model, self.output_dim)

    def forward(self, input):

        out1 = self.linear1(input)
        out2 = self.linear2(input)
        output = self.sigmoid(out1) * out2

        return output

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, d_model, output_dim, dropout):
        super(GatedResidualNetwork, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.dropout = dropout

        self.elu = nn.ELU()
        self.linear1 = nn.Linear(self.d_model, self.d_model)
        self.linear2 = nn.Linear(self.input_dim, self.d_model)
        self.linear3 = nn.Linear(self.d_model, self.d_model, bias=False)

        if self.input_dim != self.output_dim:
            self.skip_layer = nn.Linear(self.input_dim, self.output_dim)

        self.layernorm = nn.LayerNorm(self.output_dim)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

        self.gate_add_norm = GateAddNorm(self.d_model, self.output_dim)

    def forward(self, a, c=None):

        if self.input_dim != self.output_dim:
            resid = self.skip_layer(a)
        else:
            resid = a

        if c is not None:
            eta2 = self.elu(self.linear2(a) + self.linear3(c))
        else:
            eta2 = self.elu(self.linear2(a))

        eta1 = self.linear1(eta2)
        eta1 = self.dropout1(eta1)

        output = self.gate_add_norm(eta1, resid)

        return output

class GateAddNorm(nn.Module):
    def __init__(self, d_model, output_dim):
        super(GateAddNorm, self).__init__()

        self.glu = GatedLinearUnit(d_model, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x, resid):

        x = self.glu(x)
        x = self.layer_norm(x + resid)

        return x


class VariableSelectionNetworks(nn.Module):
    def __init__(self, d_model:int, num_inputs:int, dropout=0.1):
        super(VariableSelectionNetworks, self).__init__()

        self.d_model = d_model
        self.num_inputs = num_inputs
        self.dropout = dropout

        self.grn_v = GatedResidualNetwork(self.d_model*self.num_inputs, self.d_model, self.num_inputs, self.dropout)
        self.grn_xi = nn.ModuleList([
            GatedResidualNetwork(self.d_model, self.d_model, self.d_model, self.dropout) for _ in range(self.num_inputs)
            ])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, xi, c=None):

        XI = torch.flatten(xi, -2)
        v = self.softmax(self.grn_v(XI, c)) # (batch_size,sequence_len,num_inputs)

        xi_set = []
        for idx, grn in enumerate(self.grn_xi):
            xi_tilde = grn(xi[:, :, idx:idx+1], None) # (batch_size,sequence_len,1,d_model)
            xi_set.append(xi_tilde)
        xi_tilde_set = torch.cat(xi_set, dim=-2) # (batch_size,sequence_len,num_inputs,d_model)

        output = torch.matmul(v.unsqueeze(-2), xi_tilde_set).squeeze(-2)

        return output # (batch_size,sequence_len,d_model)


class QuantileOutput(nn.Module):
    def __init__(self, d_model, quantile:list, tau):
        super(QuantileOutput, self).__init__()

        self.fc = nn.Linear(d_model, len(quantile))

    def forward(self, psi):

        quantile_output = self.fc(psi)

        return quantile_output # (batch, tau, len(quantile))