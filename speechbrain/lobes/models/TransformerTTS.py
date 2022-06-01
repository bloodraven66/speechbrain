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
sys.path.append('../../../')
from speechbrain.lobes.models.transformer.Transformer import TransformerEncoder, TransformerDecoder, get_key_padding_mask, get_mel_mask

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
            layers.append(CNN.Conv1d(in_channels=n_mels, out_channels=n_mels, kernel_size=1))
        self.layers = Sequential(*layers)

    def forward(self, x):
        x_mel = self.melLinear(x)
        return self.layers(x_mel)+x_mel, x_mel, self.stopLinear(x)

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
                    ffn_type,
                    n_char,
                    n_mels,
                    padding_idx):
        super().__init__()
        self.enc_num_head = enc_num_head
        self.dec_num_head = dec_num_head
        self.padding_idx = padding_idx

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
                                        ffn_type=ffn_type)

        self.decoder = TransformerDecoder(num_layers=dec_num_layers,
                                        nhead=dec_num_head,
                                        d_ffn=dec_ffn_dim,
                                        d_model=dec_d_model,
                                        kdim=dec_k_dim,
                                        vdim=dec_v_dim,
                                        dropout=dec_dropout,
                                        activation=nn.ReLU,
                                        normalize_before=normalize_before,
                                        ffn_type=ffn_type)

        self.postNet = PostNet(model_d=dec_d_model,
                                n_mels=n_mels,
                                num_layers=post_net_num_layers)

    def forward(self, phonemes, spectogram, mel_lengths, training=True):
        phoneme_feats = self.encPreNet(phonemes)
        spec_feats = self.decPreNet(spectogram)

        srcmask = get_key_padding_mask(phonemes, pad_idx=self.padding_idx)
        attn_mask = srcmask.unsqueeze(1).repeat(self.enc_num_head, phoneme_feats.shape[1], 1)

        if not training:
            attn_mask = None
            srcmask = None

        phoneme_feats, memory = self.encoder(phoneme_feats,
                                            src_mask=attn_mask,
                                            src_key_padding_mask=srcmask)

        decoder_mel_mask = torch.triu(torch.ones(spec_feats.shape[0],
                                                spectogram.shape[1],
                                                spectogram.shape[1]),
                                                diagonal=1).to(spec_feats.device)
        if not training:
            attn_mask = None
            srcmask = None
            mask = None
            tgtattn = decoder_mel_mask.gt(0).detach()
        else:
            mask = get_mel_mask(spec_feats, mel_lengths)
            attn_mask = srcmask.unsqueeze(-1).repeat(self.dec_num_head, 1, spec_feats.shape[1]).permute(0, 2, 1)
            tgtattn = (mask.unsqueeze(-1).repeat(1, 1, mask.shape[1]) + decoder_mel_mask).detach()
            tgtattn = torch.clamp(tgtattn, min=0, max=1)

        tgtattn = tgtattn.repeat(self.dec_num_head, 1, 1)
        output_mel_feats, memory, *_ = self.decoder(spec_feats, phoneme_feats,
                                                    memory_mask=attn_mask,
                                                    tgt_mask=tgtattn,
                                                    memory_key_padding_mask=srcmask,
                                                    tgt_key_padding_mask=mask)

        mel_post, mel_linear, stop_token =  self.postNet(output_mel_feats)
        return mel_post, mel_linear, stop_token
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
        mel_padded.zero_()

        output_lengths = torch.LongTensor(len(batch))
        labels, wavs = [], []
        for i in range(len(ids_sorted_decreasing)):
            idx = ids_sorted_decreasing[i]
            mel = batch[idx][1]
            mel_padded[i, :, : mel.size(1)] = mel
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
