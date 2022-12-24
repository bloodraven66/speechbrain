"""
 Recipe for training the TransformerTTS Text-To-Speech model, an end-to-end
 neural text-to-speech (TTS) system introduced in 'Neural Speech Synthesis
 with Transformer Network' paper published in AAAI-19.
 (https://arxiv.org/abs/1809.08895)
 To run this recipe, do the following:
 # python train.py --device=cuda:0 hparams.yaml
 Authors
 * Georges Abous-Rjeili 2021
 * Artem Ploujnikov 2021
 * Yingzhi Wang 2022
 * Sathvik Udupa 2022
"""
import os, sys
import numpy as np
import torch
import logging
sys.path.append("../../../../")
import speechbrain as sb
from speechbrain.utils.text_to_sequence import text_to_sequence
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import scalarize

# sys.path.append("..")

logger = logging.getLogger(__name__)

class TransformerTTSBrain(sb.Brain):

    def on_fit_start(self):
        self.hparams.progress_sample_logger.reset()
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}
        return super().on_fit_start()


    def compute_forward(self, batch, stage):
        inputs, y = batch_to_gpu(batch)
        return self.hparams.model(*inputs)  # 1#2#

    def fit_batch(self, batch):
        # result = super().fit_batch(batch)
        # should_step = self.step % self.grad_accumulation_factor == 0
        # outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        # loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
        # (loss / self.grad_accumulation_factor).backward()
        # if should_step:
        #     if self.check_gradients(loss):
        #         self.optimizer.step()
        #     self.optimizer.zero_grad()
        #     self.optimizer_step += 1
            # self.hparams.noam_annealing(self.optimizer)
        result = super().fit_batch(batch)
        self.hparams.noam_annealing(self.optimizer)
        return result

    def compute_objectives(self, predictions, batch, stage):
        x, y = batch_to_gpu(batch)
        self._remember_sample(x, y, predictions)
        loss_stats = self.hparams.criterion(predictions, y, self.last_epoch)
        self.last_loss_stats[stage] = scalarize(loss_stats)
        return loss_stats.loss

    def _remember_sample(self, inputs, batch, predictions):
        phonemes, mel_shifted_padded, mel_lengths = inputs
        mel_post, mel_linear, stop_token, alignments = predictions
        mel_target, mel_length, phon_len  = batch
        # print(alignments[-1].shape)
        alignments_max = (
            alignments[-1][0]
            .max(dim=-1)
            .values.max(dim=-1)
            .values.unsqueeze(-1)
            .unsqueeze(-1)
        )
        alignments_output = alignments[0][0].T.flip(dims=(1,)) / alignments_max
        self.hparams.progress_sample_logger.remember(
            target=self._get_spectrogram_sample(mel_target),
            output=self._get_spectrogram_sample(mel_linear),
            output_postnet=self._get_spectrogram_sample(mel_post),
            alignments=alignments_output,
            raw_batch=self.hparams.progress_sample_logger.get_batch_sample(
                {
                    "text_padded": phonemes,
                    "input_lengths": phon_len,
                    "mel_target": mel_target,
                    "mel_out": mel_linear,
                    "mel_out_postnet": mel_post,
                    "output_lengths": mel_length,
                    "alignments": alignments,
                }
            ),
        )


    def _get_spectrogram_sample(self, raw):
        sample = raw[0]
        return torch.sqrt(torch.exp(sample))


    def on_stage_end(self, stage, stage_loss, epoch):

        if stage == sb.Stage.VALID:
            # Update learning rate
            lr = self.optimizer.param_groups[-1]["lr"]
            self.last_epoch = epoch

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(  # 1#2#
                stats_meta={"Epoch": epoch, "lr": lr},
                train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                valid_stats=self.last_loss_stats[sb.Stage.VALID],
            )

            # Save the current checkpoint and delete previous checkpoints.
            epoch_metadata = {
                **{"epoch": epoch},
                **self.last_loss_stats[sb.Stage.VALID],
            }
            self.checkpointer.save_and_keep_only(
                meta=epoch_metadata,
                min_keys=["loss"],
                ckpt_predicate=(
                    lambda ckpt: (
                        ckpt.meta["epoch"]
                        % self.hparams.keep_checkpoint_interval
                        != 0
                    )
                )
                if self.hparams.keep_checkpoint_interval is not None
                else None,
            )
            output_progress_sample = (
                self.hparams.progress_samples
                and epoch % self.hparams.progress_samples_interval == 0
            )
            # if output_progress_sample:
            #     self.run_inference_sample()
            #     self.hparams.progress_sample_logger.save(epoch)

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=self.last_loss_stats[sb.Stage.TEST],
            )
            # if self.hparams.progress_samples:
            #     self.run_inference_sample()
            #     self.hparams.progress_sample_logger.save("test")

    # def run_inference_sample(self):
    #     """Produces a sample in inference mode. This is called when producing
    #     samples and can be useful because"""
    #     if self.last_batch is None:
    #         return
    #     inputs, _, _, _, _ = self.last_batch
    #     text_padded, input_lengths, _, _, _ = inputs
    #     mel_out, _, _ = self.hparams.model.infer(
    #         text_padded[:1], input_lengths[:1]
    #     )
    #     self.hparams.progress_sample_logger.remember(
    #         inference_mel_out=self._get_spectrogram_sample(mel_out)
    #     )

def dataio_prepare(hparams):
# Define audio pipeline:\
   
    @sb.utils.data_pipeline.takes("wav", "label")
    @sb.utils.data_pipeline.provides("mel_text_pair")
    def audio_pipeline(wav, label):
        text_seq = torch.IntTensor(
            text_to_sequence(label, hparams["text_cleaners"])
        )
        audio = sb.dataio.dataio.read_audio(wav)
        mel = hparams["mel_spectogram"](audio=audio)

        len_text = len(text_seq)

        return text_seq, mel, len_text

    datasets = {}

    data_info = {
        "train": hparams["train_json"],
        "valid": hparams["valid_json"],
        "test": hparams["test_json"],
    }



    for dataset in hparams["splits"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["mel_text_pair", "wav", "label"],
        )

    return datasets


def batch_to_gpu(batch):
    (
        text_padded,
        input_lengths,
        mel_padded,
        mel_shifted_padded,
        output_lengths,
        len_x,
        labels,
        wavs
    ) = batch
    phonemes = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    max_len = torch.max(input_lengths.data).item()
    spectogram = to_gpu(mel_padded).float()
    mel_shifted_padded = to_gpu(mel_shifted_padded).float()
    mel_lengths = to_gpu(output_lengths).long()
    x = (phonemes, mel_shifted_padded, mel_lengths)
    y = (spectogram, mel_lengths, input_lengths)
    return x, y

def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def criterion(model_output, targets):
    mel_target, mel_length, phon_len  = targets

    assert len(mel_target.shape) == 3
    stop_tokens = [torch.cat([torch.zeros((mel_length[i]-1)), torch.ones((1))]) for i in range(len(mel_length))]
    mel_out, mel_out_postnet, stop_token_out, multiheadattn, sa = model_output

    for i in range(mel_target.shape[0]):
        if i == 0:
            mel_post_loss = torch.nn.MSELoss()(mel_out_postnet[i, :mel_length[i], :], mel_target[i, :mel_length[i], :])
            mel_lin_loss = torch.nn.MSELoss()(mel_out[i, :mel_length[i], :], mel_target[i, :mel_length[i], :])
        else:
            mel_post_loss = mel_post_loss + torch.nn.MSELoss()(mel_out_postnet[i, :mel_length[i], :], mel_target[i, :mel_length[i], :])
            mel_lin_loss = mel_lin_loss + torch.nn.MSELoss()(mel_out[i, :mel_length[i], :], mel_target[i, :mel_length[i], :])
    mel_post_loss = torch.div(mel_post_loss, len(mel_target))
    mel_lin_loss = torch.div(mel_lin_loss, len(mel_target))
    # stop_loss /= len(mel_target)
    return mel_post_loss + mel_lin_loss #+ stop_loss


def main():
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    sb.utils.distributed.ddp_init_group(run_opts)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    sys.path.append("../")
    from ljspeech_prepare import prepare_ljspeech

    sb.utils.distributed.run_on_main(
        prepare_ljspeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams["splits"],
            "split_ratio": hparams["split_ratio"],
            "seed": hparams["seed"],
            "skip_prep": hparams["skip_prep"],
        },
    )
    datasets = dataio_prepare(hparams)

    # Brain class initialization
    transformertts_brain = TransformerTTSBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    # Training
    with torch.autograd.detect_anomaly():
        transformertts_brain.fit(
            transformertts_brain.hparams.epoch_counter,
            datasets["train"],
            datasets["valid"],
            train_loader_kwargs=hparams["train_dataloader_opts"],
            valid_loader_kwargs=hparams["valid_dataloader_opts"],
        )



if __name__ == "__main__":
    main()
