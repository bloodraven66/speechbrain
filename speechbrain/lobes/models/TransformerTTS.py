import torch
import math
import numpy as np
from torch import nn
import sys
sys.path.append('../../../')
from torch.nn import functional as F
from speechbrain.nnet import CNN
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.embedding import Embedding
from speechbrain.nnet.dropout import Dropout2d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss
from collections import namedtuple

sys.path.append('../../../')
from speechbrain.lobes.models.transformer.Transformer import TransformerEncoder, TransformerDecoder, get_key_padding_mask, get_mel_mask

class PositionalEmbedding(nn.Module):
    """Computation of the positional embeddings.
    Arguments
    ---------
    embed_dim: int
        dimensionality of the embeddings.
    """

    def __init__(self, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.demb = embed_dim
        inv_freq = 1 / (
            10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, mask, dtype, device):
        """Computes the forward pass
        Arguments
        ---------
        seq_len: int
            length of the sequence
        mask: torch.tensor
            mask applied to the positional embeddings
        dtype: str
            dtype of the embeddings
        Returns
        -------
        pos_emb: torch.Tensor
            the tensor with positional embeddings
        """
        pos_seq = torch.arange(seq_len, device=device).to(dtype)

        sinusoid_inp = torch.matmul(
            torch.unsqueeze(pos_seq, -1), torch.unsqueeze(self.inv_freq, 0)
        )
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        if mask is None:
            return pos_emb[None, :, :]
        return pos_emb[None, :, :] * mask

class EncoderPreNet(nn.Module):
    def __init__(self, n_vocab, blank_id, out_channels, dropout, num_layers):
        super().__init__()
        layers = []
        self.phoneme_embedding = Embedding(num_embeddings=n_vocab, embedding_dim=out_channels, blank_id=blank_id)
        for idx in range(num_layers):
            layers.append(CNN.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1))
            layers.append(BatchNorm1d(input_size=out_channels))
            layers.append(nn.ReLU())
            layers.append(Dropout2d(drop_rate=dropout))
        self.layers = Sequential(*layers)
        self.linear = Linear(input_size=out_channels, n_neurons=out_channels)

    def forward(self, x):
        x = self.phoneme_embedding(x)
        x = self.layers(x)
        x = self.linear(x)
        return x

class DecoderPreNet(nn.Module):
    def __init__(self, n_mels, out_channels):
        super().__init__()
        self.fc1 = Linear(input_size=n_mels, n_neurons=out_channels)
        self.fc2 = Linear(input_size=out_channels, n_neurons=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

class PostNet(nn.Module):
    def __init__(self, model_d, n_mels, num_layers):
        super().__init__()
        self.melLinear = Linear(input_size=model_d, n_neurons=n_mels)
        self.stopLinear = Linear(input_size=model_d, n_neurons=1)

        layers = []
        for idx in range(num_layers):
            layers.append(torch.nn.LeakyReLU())
            layers.append(CNN.Conv1d(in_channels=n_mels, out_channels=n_mels, kernel_size=1))
        self.layers = Sequential(*layers)

    def forward(self, x, mask):
        # import matplotlib.pyplot as plt
        # plt.imshow(x[-10].detach().cpu().numpy())
        # plt.savefig('before')
        # plt.clf()
        x_mel = self.melLinear(x) 
        if mask is not None:
            x_mel = x_mel * mask
        
        # res = x_mel.clone()
        # print(x_mel.shape, mask.shape)
        # import matplotlib.pyplot as plt
        # plt.imshow(x_mel[-10].detach().cpu().numpy())
        # plt.savefig('after')
        # plt.clf()
        # exit()
        # return self.layers(x_mel)* mask, x_mel, self.stopLinear(x)
        return x_mel, x_mel, self.stopLinear(x)

class TransformerTTS(nn.Module):
    def __init__(self, pre_net_dropout,
                    pre_net_num_layers,
                    post_net_num_layers,
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
                    enc_ffn_type,
                    dec_ffn_type,
                    n_char,
                    n_mels,
                    padding_idx):
        super().__init__()
        self.enc_num_head = enc_num_head
        self.dec_num_head = dec_num_head
        self.padding_idx = padding_idx

        self.sinusoidal_positional_embed_encoder = PositionalEmbedding(
            enc_d_model
        )
        self.sinusoidal_positional_embed_decoder = PositionalEmbedding(
            dec_d_model
        )


        self.encPreNet = EncoderPreNet(n_vocab=n_char,
                                        blank_id=padding_idx,
                                        out_channels=enc_d_model,
                                        dropout=pre_net_dropout,
                                        num_layers=pre_net_num_layers)

        self.decPreNet = DecoderPreNet(n_mels=n_mels,
                                        out_channels=dec_d_model)

        self.encoder = TransformerEncoder(num_layers=enc_num_layers,
                                        nhead=enc_num_head,
                                        d_ffn=enc_ffn_dim,
                                        d_model=enc_d_model,
                                        kdim=enc_k_dim,
                                        vdim=enc_v_dim,
                                        dropout=enc_dropout,
                                        activation=nn.ReLU,
                                        normalize_before=normalize_before,
                                        ffn_type=enc_ffn_type)

        self.decoder = TransformerDecoder(num_layers=dec_num_layers,
                                        nhead=dec_num_head,
                                        d_ffn=dec_ffn_dim,
                                        d_model=dec_d_model,
                                        kdim=dec_k_dim,
                                        vdim=dec_v_dim,
                                        dropout=dec_dropout,
                                        activation=nn.ReLU,
                                        normalize_before=normalize_before,
                                        ffn_type=dec_ffn_type)

        self.postNet = PostNet(model_d=dec_d_model,
                                n_mels=n_mels,
                                num_layers=post_net_num_layers)

    def forward(self, phonemes, spectogram, mel_lengths, training=True):


        phoneme_feats = self.encPreNet(phonemes)
        
        spectogram = spectogram.transpose(1, 2)
        spec_feats = self.decPreNet(spectogram)
        
        srcmask = get_key_padding_mask(phonemes, pad_idx=self.padding_idx)
        srcmask_inverted = (~srcmask).unsqueeze(-1).float()
        # print(srcmask.shape, srcmask_inverted)
        pos = self.sinusoidal_positional_embed_encoder(
            phoneme_feats.shape[1], srcmask_inverted, phoneme_feats.dtype, phoneme_feats.device
        )
        # print(phoneme_feats.shape, srcmask_inverted.shape, pos.shape)
        phoneme_feats = torch.add(phoneme_feats, pos) * srcmask_inverted
        attn_mask = (
            srcmask.unsqueeze(-1)
            .repeat(self.enc_num_head, 1, phoneme_feats.shape[1])
            .permute(0, 2, 1)
            .bool()
        )
        # print(attn_mask[-2])
        # import matplotlib.pyplot as plt
        # plt.imshow(attn_mask[-2].detach().cpu().numpy())
        # plt.savefig('enc attn')
        # plt.clf()
       

        phoneme_feats, memory = self.encoder(phoneme_feats,
                                            src_mask=attn_mask,
                                            src_key_padding_mask=srcmask)
        

        phoneme_feats = phoneme_feats * srcmask_inverted

        decoder_mel_mask = torch.triu(torch.ones(spec_feats.shape[0],
                                                spectogram.shape[1],
                                                spectogram.shape[1]),
                                                diagonal=1).to(spec_feats.device)
        
        
        mask = get_mel_mask(spec_feats, mel_lengths)
        srcmask_inverted = (~mask).unsqueeze(-1).float()
        pos = self.sinusoidal_positional_embed_encoder(
            spec_feats.shape[1], srcmask_inverted, spec_feats.dtype, spec_feats.device
        )
        spec_feats = torch.add(spec_feats, pos) * srcmask_inverted
        attn_mask = srcmask.unsqueeze(-1).repeat(self.dec_num_head, 1, spec_feats.shape[1]).permute(0, 2, 1)
        tgtattn = (mask.unsqueeze(-1).repeat(1, 1, mask.shape[1]) + decoder_mel_mask).detach()
        tgtattn = torch.clamp(tgtattn, min=0, max=1)
        # import matplotlib.pyplot as plt
        # plt.imshow(mask.detach().cpu().numpy())
        # plt.savefig('mel mask')
        # plt.clf()
        tgtattn = tgtattn.repeat(self.dec_num_head, 1, 1)


        output_mel_feats, sa, multiheadattn = self.decoder(spec_feats, phoneme_feats,
                                                    memory_mask=attn_mask,
                                                    tgt_mask=tgtattn,
                                                    memory_key_padding_mask=srcmask,
                                                    tgt_key_padding_mask=mask)


  
        output_mel_feats = output_mel_feats * srcmask_inverted
        mel_post, mel_linear, stop_token =  self.postNet(output_mel_feats, srcmask_inverted)
        
        return mel_post, mel_linear, stop_token, multiheadattn

    def infer(self, phonemes):
        assert len(phonemes.shape) == 2
        assert phonemes.shape[0] == 1
        phoneme_feats = self.encPreNet(phonemes)
        pos = self.sinusoidal_positional_embed_encoder(
            phoneme_feats.shape[1], None, phoneme_feats.dtype, phoneme_feats.device
        )
        phoneme_feats = torch.add(phoneme_feats, pos)
        phoneme_feats, memory = self.encoder(phoneme_feats,
                                            src_mask=None,
                                            src_key_padding_mask=None)
        spectogram = torch.ones((1, 1, 80)).to(phoneme_feats.device)
        for i in range(600):
            spec_feats = self.decPreNet(spectogram)
            pos = self.sinusoidal_positional_embed_encoder(
            spec_feats.shape[1], None, spec_feats.dtype, spec_feats.device
            )
            spec_feats = torch.add(spec_feats, pos)
            output_mel_feats, sa, multiheadattn = self.decoder(spec_feats, phoneme_feats,
                                                    memory_mask=None,
                                                    tgt_mask=None,
                                                    memory_key_padding_mask=None,
                                                    tgt_key_padding_mask=None)
            mel_post, mel_linear, stop_token =  self.postNet(output_mel_feats, None)
            spectogram = torch.cat([spectogram, mel_post[:, -1:, :]], 1)
        spectogram = spectogram[:, 1:, :]
        return spectogram
class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per step
    Arguments
    ---------
    n_frames_per_step: int
        the number of frames per step
    Returns
    -------
    result: tuple
        a tuple of tensors to be used as inputs/targets
        (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
            len_x
        )
    """

    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step
        # pdb.set_trace()

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

        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):

            text = batch[ids_sorted_decreasing[i]][0]

            # print(text, dur)
            text_padded[i, : text.size(0)] = text
            # print(dur_padded, text_padded)
        # exit()
        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_shifted_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        mel_shifted_padded.zero_()

        output_lengths = torch.LongTensor(len(batch))
        labels, wavs = [], []
        for i in range(len(ids_sorted_decreasing)):
            idx = ids_sorted_decreasing[i]
            mel = batch[idx][1]
            mel_padded[i, :, : mel.size(1)] = mel
            mel_shifted_padded[i, :, 1: mel.size(1)] = mel[:, 1:]
            mel_shifted_padded[i, :, 0] = torch.tensor(1)
            output_lengths[i] = mel.size(1)
            labels.append(raw_batch[idx]['label'])
            wavs.append(raw_batch[idx]['wav'])
        # count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)
        mel_padded = mel_padded.permute(0, 2, 1)

        return (
            text_padded,
            input_lengths,
            mel_padded,
            mel_shifted_padded,
            output_lengths,
            len_x,
            labels,
            wavs
        )
    
    
def mel_spectogram(
    sample_rate,
    hop_length,
    win_length,
    n_fft,
    n_mels,
    f_min,
    f_max,
    power,
    normalized,
    norm,
    mel_scale,
    compression,
    audio,
):
    """calculates MelSpectrogram for a raw audio signal
    Arguments
    ---------
    sample_rate : int
        Sample rate of audio signal.
    hop_length : int
        Length of hop between STFT windows.
    win_length : int
        Window size.
    n_fft : int
        Size of FFT.
    n_mels : int
        Number of mel filterbanks.
    f_min : float
        Minimum frequency.
    f_max : float
        Maximum frequency.
    power : float
        Exponent for the magnitude spectrogram.
    normalized : bool
        Whether to normalize by magnitude after stft.
    norm : str or None
        If "slaney", divide the triangular mel weights by the width of the mel band
    mel_scale : str
        Scale to use: "htk" or "slaney".
    compression : bool
        whether to do dynamic range compression
    audio : torch.tensor
        input audio signal
    """
    from torchaudio import transforms

    audio_to_mel = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=power,
        normalized=normalized,
        norm=norm,
        mel_scale=mel_scale,
    ).to(audio.device)

    mel = audio_to_mel(audio)

    if compression:
        mel = dynamic_range_compression(mel)

    return mel
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """Dynamic range compression for audio signals
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

LossStats = namedtuple(
    "TransformerTTSLoss", "loss mel_loss attn_loss attn_weight"
)

class TTTSLoss(nn.Module):
    def __init__(
        self,
        guided_attention_sigma=None,
        gate_loss_weight=1.0,
        guided_attention_weight=1.0,
        guided_attention_scheduler=None,
        guided_attention_hard_stop=None,
    ):
        super().__init__()
        if guided_attention_weight == 0:
            guided_attention_weight = None
        self.guided_attention_weight = guided_attention_weight
        self.mel_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.guided_attention_loss = GuidedAttentionLoss(
            sigma=guided_attention_sigma
        )
        self.gate_loss_weight = gate_loss_weight
        self.guided_attention_weight = guided_attention_weight
        self.guided_attention_scheduler = guided_attention_scheduler
        self.guided_attention_hard_stop = guided_attention_hard_stop

    def forward(
        self, model_output, targets, epoch
    ):
        mel_target, target_lengths, input_lengths = targets
        mel_target.requires_grad = False
        # gate_target.requires_grad = False
        # gate_target = gate_target.view(-1, 1)

        mel_out_postnet, mel_out, gate_out, alignments = model_output

        # gate_out = gate_out.view(-1, 1)
        for i in range(mel_target.shape[0]):
            if i == 0:
                mel_post_loss = self.mel_loss(mel_out_postnet[i, :target_lengths[i], :], mel_target[i, :target_lengths[i], :])
                mel_lin_loss = self.mel_loss(mel_out[i, :target_lengths[i], :], mel_target[i, :target_lengths[i], :])
            else:
                mel_post_loss = mel_post_loss + self.mel_loss(mel_out_postnet[i, :target_lengths[i], :], mel_target[i, :target_lengths[i], :])
                mel_lin_loss = mel_lin_loss + self.mel_loss(mel_out[i, :target_lengths[i], :], mel_target[i, :target_lengths[i], :])
        mel_post_loss = torch.div(mel_post_loss, len(mel_target))
        mel_lin_loss = torch.div(mel_lin_loss, len(mel_target))
        mel_loss = mel_post_loss + mel_lin_loss
        # gate_loss = self.gate_loss_weight * self.bce_loss(gate_out, gate_target)
        attn_loss, attn_weight = self.get_attention_loss(
            alignments, input_lengths, target_lengths, epoch
        )
        # total_loss = mel_loss + gate_loss + attn_loss
        total_loss = mel_loss + attn_loss
        return LossStats(
            total_loss, mel_loss, attn_loss, attn_weight
        )

    def get_attention_loss(
        self, alignments, input_lengths, target_lengths, epoch,
    ):
        zero_tensor = torch.tensor(0.0, device=alignments[0].device)
        if (
            self.guided_attention_weight is None
            or self.guided_attention_weight == 0
        ):
            attn_weight, attn_loss = zero_tensor, zero_tensor
        else:
            hard_stop_reached = (
                self.guided_attention_hard_stop is not None
                and epoch > self.guided_attention_hard_stop
            )
            if hard_stop_reached:
                attn_weight, attn_loss = zero_tensor, zero_tensor
            else:
                attn_weight = self.guided_attention_weight
                if self.guided_attention_scheduler is not None:
                    _, attn_weight = self.guided_attention_scheduler(epoch)
            attn_weight = torch.tensor(attn_weight, device=alignments[0].device)
            for idx in range(len(alignments)):
                if idx == 0:
                    
                    attn_loss = attn_weight * self.guided_attention_loss(
                        alignments[idx], input_lengths, target_lengths
                    )
                else:
                     attn_loss = attn_loss + attn_weight * self.guided_attention_loss(
                        alignments[idx], input_lengths, target_lengths
                    )
            # attn_loss = attn_loss/len(alignments)
        return attn_loss, attn_weight