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
import torch
import copy
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.data_utils import scalarize
import torchaudio
import os


class WaveGlowBrain(sb.Brain):
    def compute_forward(self, batch, stage):



        return

    def compute_objectives(self, predictions, batch, stage):



        return

    def evaluate_batch(self, batch, stage):

        return

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics
        """
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}
        return super().on_fit_start()

    def init_optimizers(self):
        pass
    def _remember_sample(self, batch, predictions):

        mel, sig = batch
        y_hat, scores_fake, feats_fake, scores_real, feats_real = predictions

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
        audio = audio.unsqueeze(0)
        if segment:
            if audio.size(1) >= segment_size:
                max_audio_start = audio.size(1) - segment_size
                audio_start = torch.randint(0, max_audio_start, (1,))
                audio = audio[:, audio_start : audio_start + segment_size]
            else:
                audio = torch.nn.functional.pad(
                    audio, (0, segment_size - audio.size(1)), "constant"
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
    hifi_gan_brain = HifiGanBrain(
        modules=hparams["modules"],
        opt_class=[
            hparams["opt_class_generator"],
            hparams["opt_class_discriminator"],
            hparams["sch_class_generator"],
            hparams["sch_class_discriminator"],
        ],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if hparams["use_tensorboard"]:
        hifi_gan_brain.tensorboard_logger = sb.utils.train_logger.TensorboardLogger(
            save_dir=hparams["output_folder"] + "/tensorboard"
        )

    # Training
    hifi_gan_brain.fit(
        hifi_gan_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    if "test" in datasets:
        hifi_gan_brain.evaluate(
            datasets["test"],
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
