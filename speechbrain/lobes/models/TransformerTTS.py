"""
Neural network modules for the FastSpeech end-to-end neural
Text-to-Speech (TTS) model
Authors
* Sathvik Udupa 2022
"""

import torch
from torch import nn
import sys
sys.path.append('../../../')
import torch.nn as nn
from torch.nn import functional as F
from speechbrain.nnet.embedding import Embedding
from speechbrain.lobes.models.transformer.Transformer import TransformerEncoder, TransformerDecoder, get_key_padding_mask, get_lookahead_mask

class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.demb = embed_dim
        inv_freq = 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, mask, device, dtype):
        pos_seq = torch.arange(seq_len, device=device).to(dtype)

        sinusoid_inp = torch.matmul(torch.unsqueeze(pos_seq, -1),
                                    torch.unsqueeze(self.inv_freq, 0))
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        return pos_emb[None, :, :] * mask[:, :, None]

class EncoderPreNet(nn.Module):
    def __init__(self, 
                    n_vocab, 
                    blank_id, 
                    symbol_dim, 
                    conv_channels, 
                    out_channels, 
                    num_layers, 
                    kernel_size):
        super().__init__()
        self.token_embedding = Embedding(num_embeddings=n_vocab, embedding_dim=symbol_dim, blank_id=blank_id)
        conv_layers = []
        for idx in range(num_layers):
            conv_layers.append(nn.Conv1d(
                                        in_channels=symbol_dim if idx == 0 else conv_channels,
                                        out_channels=conv_channels,
                                        kernel_size=kernel_size,
                                        stride=(kernel_size // 2),
                                        padding=(kernel_size // 2,) 
                                        ))
            conv_layers.append(nn.BatchNorm1d(conv_channels))
            conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv1d(
                                    in_channels=conv_channels,
                                    out_channels=out_channels,
                                    kernel_size=1
                                    ))                 
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.token_embedding(x).permute(0, 2, 1)
        x = self.conv_layers(x)
        return x.permute(0, 2, 1)

class DecoderPreNet(nn.Module):
    def __init__(self, 
                    out_channels,
                    num_layers,
                    in_channels):
        super().__init__()
        fc_layers = []
        for idx in range(num_layers):
            fc_layers.append(nn.Linear(in_features=in_channels if idx == 0 else out_channels,
                                        out_features=out_channels))
            fc_layers.append(nn.ReLU())
        
                                                    
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.fc_layers(x)
        return x

class DecoderPostNet(nn.Module):
    def __init__(self, 
                    out_channels,
                    num_layers,
                    kernel_size):
        super().__init__()
        conv_layers = []
        for idx in range(num_layers):
            conv_layers.append(nn.Conv1d(
                                        in_channels=out_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=(kernel_size // 2),
                                        padding=(kernel_size // 2,) 
                                        ))
            if idx != num_layers-1:
                conv_layers.append(nn.LeakyReLU())
        
                                                    
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.conv_layers(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x

class TransformerTTS(nn.Module):
    def __init__(self, enc_pre_net_symbol_embed,
                    enc_pre_net_num_layers,
                    enc_pre_net_conv_kernel_size,
                    enc_pre_net_conv_num_channels,
                    dec_pre_net_symbol_embed,
                    dec_pre_net_num_layers,
                    dec_pre_net_in_channels,
                    enc_num_layers,
                    enc_num_head,
                    enc_d_model,
                    enc_ffn_dim,
                    enc_k_dim,
                    enc_v_dim,
                    enc_dropout,
                    dec_num_layers,
                    dec_num_head,
                    dec_d_model,
                    dec_ffn_dim,
                    dec_k_dim,
                    dec_v_dim,
                    dec_dropout,
                    normalize_before,
                    ffn_type,
                    n_char,
                    n_mels,
                    padding_idx,
                    dec_post_net_num_layers,
                    dec_post_net_conv_kernel_size):
        super().__init__()
        self.enc_num_head = enc_num_head
        self.dec_num_head = dec_num_head
        self.padding_idx = padding_idx

        self.sinusoidal_positional_embed_encoder = PositionalEmbedding(enc_d_model)
        self.sinusoidal_positional_embed_decoder = PositionalEmbedding(dec_d_model)

        self.encPreNet = EncoderPreNet(n_vocab=n_char, 
                                        blank_id=padding_idx, 
                                        symbol_dim=enc_pre_net_symbol_embed,
                                        num_layers=enc_pre_net_num_layers,
                                        conv_channels=enc_pre_net_conv_num_channels,
                                        out_channels=enc_d_model,
                                        kernel_size=enc_pre_net_conv_kernel_size)
        
        self.decPreNet = DecoderPreNet(out_channels=dec_d_model, 
                                        num_layers=dec_pre_net_num_layers,
                                        in_channels=dec_pre_net_in_channels)

        self.encoder = TransformerEncoder(num_layers=enc_num_layers,
                                        nhead=enc_num_head,
                                        d_ffn=enc_ffn_dim,
                                        d_model=enc_d_model,
                                        kdim=enc_k_dim,
                                        vdim=enc_v_dim,
                                        dropout=enc_dropout,
                                        activation=nn.ReLU,
                                        normalize_before=normalize_before,
                                        ffn_type=ffn_type)

        self.decoder = TransformerDecoder(num_layers=dec_num_layers,
                                        nhead=dec_num_head,
                                        d_ffn=dec_ffn_dim,
                                        d_model=dec_d_model,
                                        kdim=dec_k_dim,
                                        vdim=dec_v_dim,
                                        dropout=dec_dropout,
                                        activation=nn.ReLU,
                                        normalize_before=normalize_before)

        self.mel_linear = nn.Linear(dec_d_model, n_mels)
        self.stop_linear = nn.Linear(dec_d_model, 1)

        self.postNet = DecoderPostNet(num_layers=dec_post_net_num_layers,
                                    kernel_size=dec_post_net_conv_kernel_size,
                                    out_channels=n_mels)

    def forward(self, tokens, shifted_spectogram):
        
        token_feats = self.encPreNet(tokens)
        

        decoder_feats = self.decPreNet(shifted_spectogram)
        srcmask = get_key_padding_mask(tokens, pad_idx=self.padding_idx)
        srcmask_inverted = (~srcmask).unsqueeze(-1)
        pos = self.sinusoidal_positional_embed_encoder(token_feats.shape[1], srcmask, token_feats.device, token_feats.dtype)
        token_feats = torch.add(token_feats, pos) * srcmask_inverted
        attn_mask = srcmask.unsqueeze(-1).repeat(self.enc_num_head, 1, token_feats.shape[1]).permute(0, 2, 1).bool()

        token_feats, _ = self.encoder(token_feats, src_mask=attn_mask, src_key_padding_mask=srcmask)
        token_feats = token_feats * srcmask_inverted
        srcmask = get_key_padding_mask(decoder_feats, pad_idx=self.padding_idx)
        srcmask_inverted = (~srcmask).unsqueeze(-1)
        pos = self.sinusoidal_positional_embed_decoder(decoder_feats.shape[1], srcmask, decoder_feats.device, decoder_feats.dtype)
        decoder_feats = torch.add(decoder_feats, pos)* srcmask_inverted
        dec_attn_mask = get_lookahead_mask(decoder_feats)
        spec_feats = self.decoder(
                                tgt=decoder_feats,
                                memory=token_feats,
                                tgt_mask=dec_attn_mask,
                                )[0]
        mel_linear = self.mel_linear(spec_feats)
        mel_post = self.postNet(mel_linear)  + mel_linear
        stop_token = self.stop_linear(spec_feats)
        return mel_post, stop_token
    
    def infer(self, tokens, max_len=1000):
        token_feats = self.encPreNet(tokens)
        assert token_feats.dim() == 3
        inital = torch.zeros((token_feats.shape[0], 1, 80)).to(token_feats.device)
        for idx in range(max_len):
            decoder_feats = self.decPreNet(inital if idx == 0 else mel_post)
            token_feats, _ = self.encoder(token_feats)
            spec_feats = self.decoder(
                                    tgt=decoder_feats,
                                    memory=token_feats
                                    )[0]
            mel_linear = self.mel_linear(spec_feats)
            mel_post = self.postNet(mel_linear)  + mel_linear
            mel_post = torch.cat([inital, mel_post], 1)
        return mel_post[:, 1:, :]





class TextMelCollate:
    
    # TODO: Make this more intuitive, use the pipeline
    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        Arguments
        ---------
        batch: list
            [text_normalized, mel_normalized]
        """
        # TODO: Remove for loops and this dirty hack
        raw_batch = list(batch)
        for i in range(
            len(batch)
        ):  # the pipline return a dictionary wiht one elemnent
            batch[i] = batch[i]['mel_text_pair']

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):

            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        mel_shift_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_shift_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        labels, wavs = [], []
        for i in range(len(ids_sorted_decreasing)):
            idx = ids_sorted_decreasing[i]
            mel = batch[idx][1]
            
            mel_padded[i, :, : mel.size(1)] = mel
            mel_shift_padded[i, :, 1:mel.size(1)] = mel[:, 0:mel.size(1)-1]
            output_lengths[i] = mel.size(1)
            labels.append(raw_batch[idx]['label'])
            wavs.append(raw_batch[idx]['wav'])
        mel_padded = mel_padded.permute(0, 2, 1)
        mel_shift_padded = mel_shift_padded.permute(0, 2, 1)
        return (
            text_padded,
            input_lengths,  
            mel_padded,
            output_lengths,
            mel_shift_padded,
            labels,
            wavs
        )


class Loss(nn.Module):
   
    def __init__(
        self,
        mel_loss_weight
    ):
        super().__init__()

        self.mel_loss = nn.L1Loss()
        self.mel_loss_weight = mel_loss_weight

    def forward(
        self, predictions, targets):
        """Computes the value of the loss function and updates stats
        Arguments
        ---------
        predictions: tuple
            model predictions
        targets: tuple
            ground truth data
        Returns
        -------
        loss: torch.Tensor
            the loss value
        """
        mel_target, mel_length  = targets
        assert len(mel_target.shape) == 3
        mel_out, stop_token = predictions
        
        mel_loss = 0
        for i in range(mel_target.shape[0]):
            if i == 0:
                mel_loss = self.mel_loss(mel_out[i, :mel_length[i], :], mel_target[i, :mel_length[i], :])
            else:
                mel_loss = mel_loss + self.mel_loss(mel_out[i, :mel_length[i], :], mel_target[i, :mel_length[i], :])
        mel_loss = torch.div(mel_loss, len(mel_target))
        return mel_loss*self.mel_loss_weight
    