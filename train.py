
import os
import pickle
import torch
import torch.nn as  nn
import numpy as np
import librosa
from librosa.feature import melspectrogram
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import torch
import speechbrain as sb
import sys
import logging
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.text_to_sequence import text_to_sequence
from speechbrain.utils.data_utils import scalarize
import torch.nn as nn
import librosa
from librosa.feature import melspectrogram
logger = logging.getLogger(__name__)

class TTS(sb.Brain):
    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics"""
        self.hparams.progress_sample_logger.reset()
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}
        return super().on_fit_start()

    def compute_forward(self, batch, stage):
        """Computes the forward pass

        Arguments
        ---------
        batch: str
            a single batch
        stage: speechbrain.Stage
            the training stage

        Returns
        -------
        the model output
        """

        _,texts_padded,text_lengths,mels_padded,mel_lengths,_=batch
        return self.modules.model(texts_padded.to(self.device),text_lengths.to(self.device),mels_padded.to(self.device),mel_lengths.to(self.device))

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : torch.Tensor
            The model generated spectrograms and other metrics from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        post_mel_out, mel_out, stop_token_out = predictions
        text,texts_padded,text_lengths,mels_padded,mel_lengths,stop_token_padded=batch
        self.last_batch = batch
        loss_stats = self.hparams.criterion(
          mel_postnet_out = post_mel_out.to(self.device),
          mel_out = mel_out.to(self.device),
          stop_token_out = stop_token_out.to(self.device),
          mel_target = mels_padded.to(self.device),
          stop_token_target = stop_token_padded.to(self.device)
        )
        self._remember_sample(batch, predictions)
        self.last_loss_stats[stage] = scalarize(loss_stats)
        return loss_stats.loss

    def _remember_sample(self, batch, predictions):
        """Remembers samples of spectrograms and the batch for logging purposes

        Arguments
        ---------
        batch: tuple
            a training batch
        predictions: tuple
            predictions (raw output of the Tacotron model)
        """
        post_mel_out, mel_out, stop_token_out = predictions
        text,texts_padded,text_lengths,mels_padded,mel_lengths,stop_token_padded=batch

        self.hparams.progress_sample_logger.remember(
            raw_batch=self.hparams.progress_sample_logger.get_batch_sample(
                {
                    "post_mel_out": post_mel_out,
                    "mel_out": mel_out,
                    "stop_token_out": stop_token_out,
                    "text":text,
                    "texts_padded": texts_padded,
                    "text_lengths": text_lengths,
                    "mels_padded": mels_padded,
                    "mel_lengths": mel_lengths,
                    "stop_token_padded": stop_token_padded
                }
            ),
        )

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        # At the end of validation, we can write
        if stage == sb.Stage.VALID:
            # Update learning rate
            # lr = self.optimizer.param_groups[-1]["lr"]
            # self.last_epoch = epoch
            # old_lr, new_lr = self.hparams.lr_annealing(epoch)
            # sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            lr = self.optimizer.param_groups[-1]["lr"]
            self.last_epoch = epoch

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch": epoch, "lr":  lr},
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
            if output_progress_sample:

                self.hparams.progress_sample_logger.save(epoch)

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=self.last_loss_stats[sb.Stage.TEST],
            )
            if self.hparams.progress_samples:
                self.hparams.progress_sample_logger.save("test")


def dataio_prepare(hparams):
    @sb.utils.data_pipeline.takes("wav", "label")
    @sb.utils.data_pipeline.provides("text","mel")
    def audio_pipeline(wav, label):
        text_seq = hparams["audio_feature_extractor"].text_to_seq(text=label)
        waveform, sample_rate = torchaudio.load(wav, normalize=True)
        text=text_seq
        mel = hparams["audio_feature_extractor"].GenerateToMelSpectogram(wav=waveform)
        return text,mel

    datasets = {}
    for dataset in hparams["splits"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_json"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["text","mel"],
        )
    return datasets

if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    sys.path.append("./")

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

    tts_brain = TTS(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )


    tts_brain.fit(
        tts_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    if "test" in datasets:
        tts_brain.evaluate(
            datasets["test"],
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
