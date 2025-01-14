import torch
import torch.nn as nn
from layer import (
    CateFeatureEmbedding,
    ContiFeatureEmbedding,
    GatedResidualNetwork,
    VariableSelectionNetworks,
    GateAddNorm,
    QuantileOutput
)
from util import Supervisor

class TemporalFusionTransformer(nn.Module):
    def __init__(
            self,
            sequence_len:int,
            tau:int,
            d_model:int,
            dropout:float,
            static_cate_list:list,
            static_conti_num:int,
            past_cate_list:list,
            past_conti_num:int,
            future_cate_list:list,
            num_heads:int,
            quantiles:list,
            device
            ):

        super(TemporalFusionTransformer, self).__init__()

        self.sequence_len = sequence_len
        self.dropout = dropout
        self.tau = tau

        num_static_input = len(static_cate_list) + static_conti_num
        num_past_input = len(past_cate_list) + past_conti_num
        num_future_input = len(future_cate_list)

        # embedding
        self.static_cate_emb = CateFeatureEmbedding(d_model, static_cate_list)
        self.static_conti_emb = ContiFeatureEmbedding(d_model, static_conti_num)
        self.category_emb = CateFeatureEmbedding(d_model, past_cate_list)
        self.continuous_emb = ContiFeatureEmbedding(d_model, past_conti_num)
        self.future_emb = CateFeatureEmbedding(d_model, future_cate_list)

        # static covariate encoders
        self.grn_cs = GatedResidualNetwork(d_model, d_model, d_model, dropout) # variable selection context
        self.grn_cc = GatedResidualNetwork(d_model, d_model, d_model, dropout) # lstm initial cell state
        self.grn_ch = GatedResidualNetwork(d_model, d_model, d_model, dropout) # lstm initial hidden state
        self.grn_ce = GatedResidualNetwork(d_model, d_model, d_model, dropout) # static enrichment context

        # variable selection networks
        self.static_variable_selection = VariableSelectionNetworks(d_model, num_static_input, dropout)
        self.past_variable_selection = VariableSelectionNetworks(d_model, num_past_input, dropout)
        self.future_variable_selection = VariableSelectionNetworks(d_model, num_future_input, dropout)

        # lstm encoder-decoder
        self.lstm_encoder = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        self.lstm_decoder = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)

        # static enrichment
        self.local_gate_add_norm = GateAddNorm(d_model, d_model)
        self.static_enrich_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)

        # temporal self-attention
        self.temporal_mask = torch.triu(torch.ones([sequence_len, sequence_len], dtype=torch.bool, device=device), diagonal=1)
        self.mha = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.attn_gate_add_norm = GateAddNorm(d_model, d_model)

        # position-wise feed-forward
        self.position_wise_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.position_wise_gate_add_norm = GateAddNorm(d_model, d_model)

        # quantile output
        self.quantile_output = QuantileOutput(d_model, quantiles, tau)

    def forward(self,
                static_cate_input,
                static_conti_input,
                past_cate_input,
                past_conti_input,
                future_input,
                ):

        # embedding
        static_cate_input = self.static_cate_emb(static_cate_input.to(torch.long)) # (batch,1,static_cate_num,d_model)
        static_conti_input = self.static_conti_emb(static_conti_input.to(torch.float)) # (batch,1,static_conti_num,d_model)
        static_input = torch.cat([static_cate_input, static_conti_input], dim=-2)

        past_cate_input = self.category_emb(past_cate_input.to(torch.long)) # (batch,encoder_len,past_cate_num,d_model)
        past_conti_input = self.continuous_emb(past_conti_input.to(torch.float)) # (batch,encoder_len,past_conti_num,d_model)
        past_input = torch.cat([past_cate_input,past_conti_input], dim=2)

        future_input = self.future_emb(future_input.to(torch.long)) # (batch,decoder_len,future_num,d_model)

        # static covariates encoders
        static_encoder_input = self.static_variable_selection(static_input) # (batch,1,d_model)
        c_s = self.grn_cs(static_encoder_input) # (batch,1,d_model)
        c_lstm_cell = self.grn_cc(static_encoder_input) # (batch,1,d_model)
        c_lstm_hidden = self.grn_ch(static_encoder_input) # (batch,1,d_model)
        c_e = self.grn_ce(static_encoder_input) # (batch,1,d_model)

        # variable selection networks
        # 여기 순서를 바꾸려고 했는데 VSN 적용 전에 c_s를 만들어야 해서 그대로 두었습니다.
        past_input = self.past_variable_selection(past_input, c_s) # (batch,encoder_len,d_model)
        future_input = self.future_variable_selection(future_input, c_s) # (batch,decoder_len,d_model)
        resid_vsn = torch.cat([past_input, future_input], dim=1)

        # lstm encoder-decoder
        enc_output, (enc_c_lstm_cell, enc_c_lstm_hidden) = self.lstm_encoder(past_input, (c_lstm_cell.permute(1,0,-1), c_lstm_hidden.permute(1,0,-1)))
        dec_output, _ = self.lstm_decoder(future_input, (enc_c_lstm_cell, enc_c_lstm_hidden))
        enc_dec_output = torch.cat([enc_output, dec_output], dim=1)

        static_enrich_input = self.local_gate_add_norm(enc_dec_output, resid_vsn)

        # static enrichment
        mha_input = self.static_enrich_grn(static_enrich_input, c_e)

        # temporal self-attention
        mha_output, attn_weights = self.mha(query=mha_input,
                                         key=mha_input,
                                         value=mha_input,
                                         attn_mask=self.temporal_mask) # (batch,sequence_len,d_model)
        mha_output = mha_output[:, -self.tau:, :]

        static_enrich_input = static_enrich_input[:, -self.tau:, :]
        mha_input = mha_input[:, -self.tau:, :]

        position_wise_input = self.attn_gate_add_norm(mha_output, mha_input)

        # position-wise feed-forward
        position_wise_output = self.position_wise_grn(position_wise_input)
        quantile_input = self.position_wise_gate_add_norm(position_wise_output, static_enrich_input)

        # quantile output
        quantile_forecasts = self.quantile_output(quantile_input)

        return quantile_forecasts, attn_weights
    

class TFT(Supervisor):
    def __init__(self, config, **kwargs) -> None:
        super(TFT, self).__init__(config, **kwargs)

        self.sequence_len = config['model']['sequence_len']
        self.tau = config['model']['tau']
        self.static_cate_list = config['model']['static_cate_list']
        self.static_conti_num = config['model']['static_conti_num']
        self.past_cate_list = config['model']['past_cate_list']
        self.past_conti_num = config['model']['past_conti_num']
        self.future_cate_list = config['model']['future_cate_list']
        self.d_model = config['model']['d_model']
        self.dropout = config['model']['dropout']
        self.num_heads = config['model']['num_heads']
        self.quantiles = config['model']['quantiles']
        self.device = config['model']['device']

        self.model_load()

    def model_load(self):

        print("Model Initialization...")

        self.model = TemporalFusionTransformer(
            sequence_len=self.sequence_len,
            tau=self.tau,
            d_model=self.d_model,
            dropout=self.dropout,
            static_cate_list=self.static_cate_list,
            static_conti_num=self.static_conti_num,
            past_cate_list=self.past_cate_list,
            past_conti_num=self.past_conti_num,
            future_cate_list=self.future_cate_list,
            num_heads=self.num_heads,
            quantiles=self.quantiles,
            device=self.device
        )
        
        super(TFT, self).init_model()

