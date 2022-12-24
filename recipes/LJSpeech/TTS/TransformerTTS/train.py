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
from hyperpyyaml import load_hyperpyyaml

# sys.path.append("..")
from recipes.LJSpeech.TTS.common.utils import PretrainedModelMixin, ProgressSampleImageMixin

logger = logging.getLogger(__name__)

class TransformerTTSBrain(sb.Brain):

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.init_progress_samples()

    def compute_forward(self, batch, stage):
        inputs, y = batch_to_gpu(batch)
        return self.hparams.model(*inputs)  # 1#2#

    def fit_batch(self, batch):
        # result = super().fit_batch(batch)
        should_step = self.step % self.grad_accumulation_factor == 0
        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
        (loss / self.grad_accumulation_factor).backward()
        if should_step:
            if self.check_gradients(loss):
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.optimizer_step += 1
        return loss.detach().cpu()

    def compute_objectives(self, predictions, batch, stage):
        x, y = batch_to_gpu(batch)
        self._remember_sample(y, predictions)
        return criterion(predictions, y)

    def _remember_sample(self, batch, predictions):
        mel_post, mel_linear, stop_token, multiheadattn, sa = predictions
        mel_target, mel_length, phon_len  = batch
        self.multiheadattn = multiheadattn
        self.remember_progress_sample(
                                    target=self._clean_mel(mel_target, mel_length),
                                    pred=self._clean_mel(mel_post, mel_length)
                                    )



    def _clean_mel(self, mel, len, sample_idx=0):
        assert mel.dim() == 3
        return torch.sqrt(torch.exp(mel[sample_idx][:len[sample_idx]]))


    def on_stage_end(self, stage, stage_loss, epoch):

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
            }
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(12,4))
        for i in range(len(self.multiheadattn)):
            ax[i//3][i%3].imshow(self.multiheadattn[i][0].T.detach().cpu().numpy())
        path = os.path.join(
                self.hparams.progress_sample_path, str(epoch), 'mha.png'
            )
        if not os.path.exists(os.path.join(
                self.hparams.progress_sample_path, str(epoch))):
            os.makedirs(os.path.join(
                    self.hparams.progress_sample_path, str(epoch)))
        plt.savefig(path)
        # At the end of validation, we can write
        if stage == sb.Stage.VALID:
            # Update learning rate
            lr = self.optimizer.param_groups[-1]["lr"]

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(  # 1#2#
                stats_meta={"Epoch": epoch, "lr": lr, "steps":self.optimizer_step},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            output_progress_sample = (
                self.hparams.progress_samples
                and epoch % self.hparams.progress_samples_interval == 0
                and epoch > self.hparams.progress_samples_min_rin
            )

            if output_progress_sample:
                print('saving')
                self.save_progress_sample(epoch)

            # Save the current checkpoint and delete previous checkpoints.
            #UNCOMMENT THIS
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["loss"])

        # We also write statistics about test data spectogramto stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
def dataio_prepare(hparams):
# Define audio pipeline:\
    if not os.path.exists("lexicon"):
        with open(hparams["train_data_path"], 'r') as f:
            lines = f.read().split('\n')[:-1]
        char_set = set()
        for l in lines:
            char_set.update(*l.lower().split('|')[1])
        with open("lexicon", 'w') as f:
            f.write('\t'.join(char_set))
    with open('lexicon', 'r') as f:
        lexicon = f.read().split('\t')
    # lexicon.remove(' ')
    input_encoder = hparams.get("input_encoder")
    lexicon = ['@@'] + lexicon
    input_encoder.update_from_iterable(
                lexicon,
                sequence_input=False)
    @sb.utils.data_pipeline.takes("wav", "label")
    @sb.utils.data_pipeline.provides("mel_text_pair")
    def audio_pipeline(wav, label):
        text_seq = input_encoder.encode_sequence_torch(label.lower()).int()
        audio = sb.dataio.dataio.read_audio(wav)
        mel = hparams["mel_spectogram"](audio=audio)

        len_text = len(text_seq)

        return text_seq, mel, len_text

    datasets = {}

    dataset_names = {
                    'train': hparams["train_data_path"],
                    'valid': hparams["valid_data_path"],
                    'test': hparams["test_data_path"]
                    }



    for dataset in dataset_names:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=os.path.join(hparams["save_folder"], dataset+'.csv'),
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
            "train": hparams["train_data_path"],
            "valid": hparams["valid_data_path"],
            "test": hparams["test_data_path"],
            "wavs": hparams["audio_folder"],
            "seed": hparams["seed"],
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
    transformertts_brain.fit(
        transformertts_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    if hparams.get("save_for_pretrained"):
        transformertts_brain.save_for_pretrained()

if __name__ == "__main__":
    main()
