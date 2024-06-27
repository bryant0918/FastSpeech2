import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder, vocoder_infer
from utils.tools import to_device, log, synth_one_sample, synth_one_sample_pretrain
from model import FastSpeech2Loss
from dataset import PreTrainDataset, TrainDataset


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def evaluate(model, step, configs, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = TrainDataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    word_step = train_config["step"]["word_step"]
    step = 1
    
    # Evaluation
    loss_sums = [0 for _ in range(9)]
    for batches in loader:
        for batch in batches:
            batch = to_device(batch, device)
            with torch.no_grad():
                # realign pitch energy and duration here for target use batched for source below
                realigned_p = realign_p_e_d(batch[16], batch[17])
                realigned_e = realign_p_e_d(batch[16], batch[18])
                realigned_d = realign_p_e_d(batch[16], batch[19])
                realigned_d = custom_round(realigned_d)

                # Forward pass: Src to Tgt
                input = (batch[4],) + batch[12:15] + batch[8:11] + batch[15:17] + (realigned_p, realigned_e, realigned_d, batch[-1])
                output_tgt = model(*(input))

                log_duration_targets = torch.log(realigned_d.float() + 1)

                # For calculating Word Loss
                if step % word_step == 0:
                    mels = [output_tgt[1][i, :output_tgt[9][i]].transpose(0,1) for i in range(batch_size)]
                    wav_predictions = vocoder_infer(
                        mels,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    loss_input = (batch[2],) + (batch[8],) + (realigned_p, realigned_e, log_duration_targets)
                    loss_predictions = output_tgt + (wav_predictions,)
                else:
                    loss_input = (None, batch[8]) + (realigned_p, realigned_e, log_duration_targets)
                    loss_predictions = output_tgt + (None,)
                
                # Calculate loss for Src to Tgt
                losses_src_to_tgt = Loss(loss_input, loss_predictions, "to_tgt")
                total_loss_src_to_tgt = losses_src_to_tgt[0]
                print("total_loss_src_to_tgt: ", total_loss_src_to_tgt)
                
                alignments = flip_mapping(batch[16], batch[5].shape[1])

                d_src = realigned_d
                # realign p,e,d targets back to src space
                realigned_p = realign_p_e_d(alignments, output_tgt[2])
                realigned_e = realign_p_e_d(alignments, output_tgt[3])
                realigned_log_d = realign_p_e_d(alignments, output_tgt[4])
                realigned_d = torch.clamp(torch.exp(realigned_log_d) - 1, min=0)
                realigned_d = custom_round(realigned_d)

                # Forward pass: Tgt to Src (so tgt is now src and src is now tgt)
                output_src = model(langs=batch[11], texts=batch[5], text_lens=batch[6], max_text_len=batch[7],
                                   mels=output_tgt[1], mel_lens=output_tgt[9], max_mel_len=batch[10],
                                   speaker_embs=batch[15], alignments=alignments, p_targets=realigned_p, 
                                   e_targets=realigned_e, d_targets=realigned_d, d_src=d_src)

                # For calculating Word Loss
                if step % word_step == 0:
                    mels = [output_src[1][i, :output_src[9][i]].transpose(0,1) for i in range(batch_size)]
                    wav_predictions = vocoder_infer(
                        mels,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    loss_input = (batch[1],) + (batch[8],) + (realigned_p, realigned_e, realigned_log_d)
                    loss_predictions = output_src[:10] + (output_tgt[10],) + (output_src[11],) + (wav_predictions,)
                else:
                    loss_input = (None, batch[8]) + (realigned_p, realigned_e, realigned_log_d)
                    loss_predictions = output_src[:10] + (output_tgt[10],) + (output_src[11],) + (None,)

                # Calculate loss for Tgt to Src
                losses_tgt_to_src = Loss(loss_input, loss_predictions, "to_src")
                total_loss_tgt_to_src = losses_tgt_to_src[0]

                # Combine the losses
                total_loss = (total_loss_src_to_tgt + total_loss_tgt_to_src) / 2


                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])
                
            step += 1

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, Prosody Loss: {:.4f}, Word Loss: {:.4f}, Full Duration Loss: {:.4f}".format(
                *([step] + [l for l in loss_means])
    )

    if logger is not None:
        targets = (batch[0],) +(batch[7],) + batch[11:]
        predictions = (output[1],) + output[8:10]
        targets = (batch[0],) +(batch[8],) + (realigned_p, realigned_e, realigned_d)
        predictions = (output[1],) + output[8:10]
        
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            tgt_targets,
            src_targets,
            predicted_tgt,
            predicted_src,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message

def evaluate_pretrain(model, step, configs, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = PreTrainDataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    word_step = train_config["step"]["word_step"]
    step = 1
    # Evaluation
    loss_sums = [0 for _ in range(9)]
    for batches in loader:
        for batch in batches:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                input = batch[3:11] + (None,) + batch[11:]
                output = model(*(input))

                log_duration_targets = torch.log(batch[-1] + 1)
                if step % word_step == 0:
                    # Get predicted audio
                    mels = [output[1][i, :output[9][i]].transpose(0,1) for i in range(batch_size)]
                    wav_predictions = vocoder_infer(
                        mels,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    loss_input = (batch[1],) + batch[7:9] + batch[11:13] + (log_duration_targets,)
                    loss_predictions = output + (wav_predictions,)
                else:
                    loss_input = (None,) + batch[7:9] + batch[11:13] + (log_duration_targets,)
                    loss_predictions = output + (None,)
                
                # Call Loss
                losses = Loss(loss_input, loss_predictions, "to_src")

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])
                
            step += 1

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, Prosody Loss: {:.4f}, Word Loss: {:.4f}, Full Duration Loss: {:.4f}".format(
                *([step] + [l for l in loss_means])
    )

    if logger is not None:
        targets = (batch[0],) +(batch[7],) + batch[11:]
        predictions = (output[1],) + output[8:10]
        
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample_pretrain(
            targets,
            predictions,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs)
    print(message)