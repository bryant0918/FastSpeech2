import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder, vocoder_infer
from utils.tools import to_device, log, synth_one_sample, synth_one_sample_pretrain, flip_mapping, realign_p_e_d, custom_round
from utils.training import loop
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
                # Forward
                losses_src_to_tgt, losses_tgt_to_src, output_tgt, output_src = loop(preprocess_config, model_config, batch, model, Loss, vocoder, step, word_step)
                               
                losses = [(l1.item() + l2.item())/2 for l1, l2 in zip(losses_src_to_tgt, losses_tgt_to_src)]

                for i in range(len(losses)):
                    loss_sums[i] += losses[i] * len(batch[0])
                
            step += 1

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, Prosody Loss: {:.4f}, Word Loss: {:.4f}, Full Duration Loss: {:.4f}".format(
                *([step] + [l for l in loss_means])
    )

    if logger is not None:
        # Want to see all 3 mels
        src_gt = (batch[0], batch[6]) + batch[8:10] + batch[17:20]
        tgt_targets = (batch[8], batch[13]) + batch[20:]
        src_targets = (batch[8],) + batch[17:20]
        predicted_tgt = (output_tgt[1],) + output_tgt[8:10]
        predicted_src = (output_src[1],) + output_src[8:10]

        fig, tgt_wav_prediction, src_wav_prediction, wav_reconstruction, tag = synth_one_sample(
            src_gt, 
            tgt_targets, src_targets,
            predicted_tgt, predicted_src,
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
            audio=tgt_wav_prediction,
            sampling_rate=sampling_rate,
            tag="Training/step_{}_{}_tgt_synthesized".format(step, tag),
        )
        log(
            logger,
            audio=src_wav_prediction,
            sampling_rate=sampling_rate,
            tag="Training/step_{}_{}_src_synthesized".format(step, tag),
        )
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Training/step_{}_{}_reconstructed".format(step, tag),
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