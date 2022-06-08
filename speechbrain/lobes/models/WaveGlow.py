# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

"""
Speechbarin implementation of WAVEGLOW: A FLOW-BASED GENERATIVE NETWORK FOR SPEECH SYNTHESIS by
Ryan Prenger, Rafael Valle, Bryan Catanzaro, accepted at ICASSP 2018.

The papers used as reference for this implementation:
 - waveglow (https://arxiv.org/abs/1811.00002)
 - glow (https://arxiv.org/abs/1807.03039)
 - wavenet ()

The codes used as reference for this implementation:
 - https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/waveglow
 [modified wavenet]
 -
 [speechbrain wavenet]
 - https://github.com/openai/glow
 [glow architecture]

Authors:
* Sathvik Udupa 2022
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from torch.autograd import Variable
import torch.nn.functional as F
sys.path.append('../../../')
from speechbrain.nnet import CNN

class WN(nn.Module):
    def __init__(self,
                n_group,
                total_n_group,
                num_layers,
                out_channels,
                kernel_size,
                n_mels,
                ):
        super().__init__()
        self.num_layers = num_layers
        n_mels = n_mels*total_n_group
        in_channels = n_group //2
        self.out_channels = out_channels
        self.first_conv = nn.Conv1d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1)
        self.last_conv = nn.Conv1d(in_channels=out_channels,
                                    out_channels=n_group,
                                    kernel_size=1)
        #reset last layer weights


        self.aud_conv_layers = nn.ModuleList()
        self.mel_conv_layers = nn.ModuleList()
        self.residual_conv_layers = nn.ModuleList()
        for layer in range(num_layers):
            dilation = 2 ** layer
            padding = (dilation * (kernel_size-1))//2
            residual_channels = 2*out_channels if layer < num_layers-1 else out_channels
            aud_conv_layer = nn.Conv1d(in_channels=out_channels,
                                        out_channels=2*out_channels,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        dilation=dilation)
            mel_conv_layer = nn.Conv1d(in_channels=n_mels,
                                        out_channels=2*out_channels,
                                        kernel_size=1)
            residual_conv_layer = nn.Conv1d(in_channels=out_channels,
                                            out_channels=residual_channels,
                                            kernel_size=1)
            self.aud_conv_layers.append(nn.utils.weight_norm(aud_conv_layer, name='weight'))
            self.mel_conv_layers.append(nn.utils.weight_norm(mel_conv_layer, name='weight'))
            self.residual_conv_layers.append(nn.utils.weight_norm(residual_conv_layer))

    def forward(self, audio_feats, mel_feats):
        audio_feats = self.first_conv(audio_feats)
        for layer in range(self.num_layers):
            combined_feats = self.aud_conv_layers[layer](audio_feats) + self.mel_conv_layers[layer](mel_feats)
            combined_feats = torch.tanh(combined_feats[:, :self.out_channels, :]) * torch.sigmoid(combined_feats[:, self.out_channels:, :])
            combined_feats = self.residual_conv_layers[layer](combined_feats)
            if layer < self.num_layers - 1:
                audio_feats = combined_feats[:, :self.out_channels, :] + audio_feats
                combined_feats = combined_feats[:, self.out_channels:, :]
            if layer == 0:
                output_feats = combined_feats
            else:
                output_feats = combined_feats + output_feats
        output_feats = self.last_conv(output_feats)
        return output_feats[:, output_feats.shape[1]//2:, :], output_feats[:, :output_feats.shape[1]//2, :]

# as done in https://arxiv.org/abs/1605.08803 (Density estimation using Real NVP)
class AffineCouplingLayer(nn.Module):
    def __init__(self,
                n_group,
                total_n_group,
                wn_num_layers,
                wn_kernel_size,
                wn_num_channels,
                n_mels):
        super().__init__()
        self.wn = WN(n_group=n_group,
                    total_n_group=total_n_group,
                    num_layers=wn_num_layers,
                    out_channels=wn_num_channels,
                    kernel_size=wn_kernel_size,
                    n_mels=n_mels)

    def split(self, x):
        bs, ts, ch = x.shape
        return torch.split(x, split_size_or_sections=ts//2, dim=1)

    def forward(self, x, mel):
        x_a, x_b = self.split(x)
        log_s, t = self.wn(x_a, mel)
        y_b = torch.mul(torch.exp(log_s), x_b) + t
        y_a = x_a
        return torch.cat([y_a, y_b], dim=1), log_s

    def infer(self, x, mel):
        x_a, x_b = self.split(x)
        log_s, t = self.wn(xA, mel)
        y_b = (x_b - t) / torch.exp(log_s)
        y_a = x_a
        return torch.cat([y_a, y_b], dim=1)

#implemented in https://arxiv.org/pdf/1807.03039.pdf (glow)
class Inverstible1x1Conv(nn.Module):
    def __init__(self,
                num_layers,
                lu_decom,
                w_shape):
        super().__init__()
        self.conv = nn.Conv1d(w_shape, w_shape, 1, bias=False)
        weight = torch.linalg.qr(torch.randn((w_shape, w_shape), dtype=torch.float32))[0]
        if lu_decom:
            A_LU, pivots = weight.lu()
            P, L, U = torch.lu_unpack(A_LU, pivots)
            np_s = torch.diag(np_u)
            np_sign_s = torch.sign(np_s)
            np_log_s = torch.log(torch.abs(np_s))
            np_u = torch.triu(np_u, diaginal=1)
            l_mask = torch.tril(torch.ones((w_shape, w_shape)), -1)
            l = L * l_mask + torch.eye((w_shape, w_shape))
            u = u * l_mask.T + torch.diag(sign_s * torch.exp(log_s))
            w = torch.matmul(p, torch.matmul(l, u))

        else:

            if torch.det(weight) < 0:
                weight[:, -1] = -1 * weight[:, -1]
        weight = weight[:, :, None]
        self.conv.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        z = self.conv(x)
        weight = self.conv.weight.squeeze()
        logdet = torch.logdet(weight)*x.shape[0]*x.shape[1]

        return z, logdet

    def infer(self, x):
        weight = self.conv.weight.squeeze()
        weight_inverse = weight.float().inverse()
        weight_inverse = Variable(weight_inverse[..., None])
        x = F.conv1d(x, weight_inverse, bias=None, stride=1, padding=0)
        return x


class WaveGlow(nn.Module):
    def __init__(self,
                flow_steps,
                early_output_interval,
                early_output_num_channels,
                squeeze_group,
                mel_upsample_kernel_size,
                mel_upsample_kernel_stride,
                n_mels,
                lu_decom,
                num_inv_layers,
                wn_num_layers,
                wn_kernel_size,
                wn_num_channels,
                ):
        super().__init__()
        self.upsample_mel = nn.ConvTranspose1d(n_mels,
                                            n_mels,
                                            mel_upsample_kernel_size,
                                            mel_upsample_kernel_stride)
        inv_1x1_layer, affine_couplin_later = [], []
        inv_w_shape = squeeze_group
        for idx in range(flow_steps):
            if idx % early_output_interval == 0 and idx >0:

                inv_w_shape -= early_output_num_channels
            inv_1x1_layer.append(Inverstible1x1Conv(num_inv_layers, lu_decom, inv_w_shape))
            affine_couplin_later.append(AffineCouplingLayer(n_group=inv_w_shape,
                                                            total_n_group=squeeze_group,
                                                            wn_num_layers=wn_num_layers,
                                                            wn_kernel_size=wn_kernel_size,
                                                            wn_num_channels=wn_num_channels,
                                                            n_mels=n_mels))
        self.inv_1x1_layer = nn.Sequential(*inv_1x1_layer)
        self.affine_couplin_later = nn.Sequential(*affine_couplin_later)
        self.flow_steps = flow_steps
        self.early_output_interval = early_output_interval
        self.early_output_num_channels = early_output_num_channels
        self.squeeze_group = squeeze_group
        self.inv_w_shape = inv_w_shape
    #implemented in https://arxiv.org/pdf/1605.08803.pdf (real nvp)
    def group_audio(self, x):
        x = x.view(x.shape[0], self.squeeze_group, x.shape[1]//self.squeeze_group)
        return x
    def group_mel(self, x):
        bs, n_mels, ts = x.shape
        x = x.view(bs, n_mels, self.squeeze_group,ts//self.squeeze_group)
        x = x.reshape(bs, n_mels* self.squeeze_group, ts//self.squeeze_group)
        return x

    def forward(self, mel, audio):
        assert mel.dim() == 3 and audio.dim() == 2
        bs, ts = audio.shape
        mel = mel.permute(0, 2, 1)
        up_sample_mel = self.upsample_mel(mel)[:, :, :ts]
        z = self.group_audio(audio)
        up_sample_mel = self.group_mel(up_sample_mel)
        z_final, log_det, log_s = [], [], []
        for idx in range(self.flow_steps):

            if idx % self.early_output_interval == 0 and idx != 0:
                z_final.append(z[:, :self.early_output_num_channels, :])
                z = z[:, self.early_output_num_channels:, :]
            z, layer_log_det = self.inv_1x1_layer[idx](z)
            z, layer_log_s = self.affine_couplin_later[idx](z, up_sample_mel)
            log_det.append(layer_log_det)
            log_s.append(layer_log_s)


        z_final.append(z)
        z_final = torch.cat(z_final, dim=1)
        return z_final, log_det, log_s

    def infer(self, mel, sigma=1.0):
            assert mel.dim() == 3
            mel = mel.permute(0, 2, 1)
            mel = self.upsample_mel(mel)
            time_cutoff = self.upsample_mel.kernel_size[0] - self.upsample_mel.stride[0]
            mel = mel[:, :, :-time_cutoff]

            mel = mel.unfold(2, self.squeeze_group, self.squeeze_group).permute(0, 2, 1, 3)
            mel = mel.contiguous().view(mel.size(0), mel.size(1), -1)
            mel = mel.permute(0, 2, 1)

            audio = torch.randn(spect.size(0),
                                self.inv_w_shape,
                                spect.size(2), device=spect.device).to(spect.dtype)
            print(audio.shape, spect.shape)
            audio = torch.autograd.Variable(sigma * audio)

            for k in reversed(range(self.flow_steps)):

                audio = self.affineCouplingLayer[k].infer(audio, spect)

                audio = self.inv1x1layer[k].infer(audio)

                if k % self.early_output_interval == 0 and k != 0:
                    z = torch.randn(spect.size(0), self.early_output_num_channels, spect.size(
                        2), device=spect.device).to(spect.dtype)
                    audio = torch.cat((sigma * z, audio), 1)

            audio = audio.permute(
                0, 2, 1).contiguous().view(
                audio.size(0), -1).data
            return audio

m = WaveGlow(flow_steps=12,
            early_output_interval=4,
            early_output_num_channels=2,
            squeeze_group=8,
            mel_upsample_kernel_size=1024,
            mel_upsample_kernel_stride=256,
            n_mels=80,
            lu_decom=False,
            num_inv_layers=4,
            wn_num_layers=4,
            wn_kernel_size=3,
            wn_num_channels=128,)
# sr = 16000
# nfft = 1024
# hl = 256
# nmel = 80
# torch.manual_seed(0)
#
# aud = torch.ones((1, int(sr*2))) #needs to be %num group
# mel = torch.ones((1, int(sr*2/hl), nmel))
# z_final, log_det, log_s = m(mel, aud)
# print(z_final.shape,[l.shape for l in log_s], [l.shape for l in log_det], )
# print(torch.sum(z_final, dim=2))
# # out = m.infer(mel)
# # print(out.shape, aud.shape)
