#!/usr/bin/env python3
"""Recipe for training a waveglow vocoder, introduced in
WaveGlow: A flowbased generative network for speech synthesis at ICASSP 2019.
(https://arxiv.org/abs/1811.00002)


To run this recipe, do the following:
> python train.py hparams/train.yaml

Authors
 * Sathvik Udupa 2022
"""

import sys
sys.path.append('../../../../../')
import torch
import copy
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.data_utils import scalarize
import torchaudio
import os


class WaveGlowBrain(sb.Brain):

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics"""
        self.hparams.progress_sample_logger.reset()
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}
        return super().on_fit_start()

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        mel, _ = batch.mel
        y, _ = batch.sig
        return self.hparams.model(mel.transpose(2, 1), y)

    # def init_optimizers(self):
    #     """Called during ``on_fit_start()``, initialize optimizers
    #     after parameters are fully configured (e.g. DDP, jit).
    #     """
    #     if self.opt_class is not None:
    #         opt_class = self.opt_class
    #
    #         self.optimizer = opt_class(self.hparams.model.parameters())
    #
    #         if self.checkpointer is not None:
    #             self.checkpointer.add_recoverable(
    #                 "optimizer", self.optimizer
    #             )


    def compute_objectives(self, predictions, batch, stage):

        return self.criterion(predictions)


    def criterion(self, predictions):
        sigma = self.hparams.sigma
        z_final, log_det, log_s = predictions
        for i, log_s_ in enumerate(log_s):
            if i == 0:
                log_s_total = torch.sum(log_s_)
                log_det_W_total = log_det[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s_)
                log_det_W_total += log_det[i]

        loss = torch.sum(z_final*z_final)/(2*sigma*sigma) - log_s_total - log_det_W_total
        return loss/(z_final.size(0)*z_final.size(1)*z_final.size(2))

    def fit_batch(self, batch):
        result = super().fit_batch(batch)
        return result



    def _remember_sample(self, batch, predictions):

        pass

    def on_stage_end(self, stage, stage_loss, epoch):
        return


    def run_inference_sample(self, name):
        return

    def save_audio(self, name, data, epoch):
        return


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    segment_size = hparams["segment_size"]

    # Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "segment")
    @sb.utils.data_pipeline.provides("mel", "sig")
    def audio_pipeline(wav, segment):
        audio = sb.dataio.dataio.read_audio(wav)
        audio = torch.FloatTensor(audio)
        audio = audio
        if segment:
            if audio.size(0) >= segment_size:
                max_audio_start = audio.size(0) - segment_size
                audio_start = torch.randint(0, max_audio_start, (1,))
                audio = audio[audio_start : audio_start + segment_size]
            else:
                audio = torch.nn.functional.pad(
                    audio, (0, segment_size - audio.size(0)), "constant"
                )

        mel = hparams["mel_spectogram"](audio=audio.squeeze(0))
        return mel, audio

    datasets = {}
    for dataset in hparams["splits"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_json"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["id", "mel", "sig"],
        )

    return datasets


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    sb.utils.distributed.ddp_init_group(run_opts)
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    sys.path.append("../../")
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
    waveglow_brain = WaveGlowBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        )
    if hparams["use_tensorboard"]:
        waveglow_brain.tensorboard_logger = sb.utils.train_logger.TensorboardLogger(
            save_dir=hparams["output_folder"] + "/tensorboard"
        )

    # Training
    waveglow_brain.fit(
        waveglow_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    if "test" in datasets:
        waveglow_brain.evaluate(
            datasets["test"],
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
