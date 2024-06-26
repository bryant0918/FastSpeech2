import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder, vocoder_infer
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import PreTrainDataset, TrainDataset


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def evaluate(model, step, configs, logger=None, vocoder=None, pretrain=False):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    if pretrain:
        dataset = PreTrainDataset(
            "val.txt", preprocess_config, train_config, sort=False, drop_last=False
        )
    else:
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
    loss_sums = [0 for _ in range(8)]
    for batches in loader:
        for batch in batches:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                if pretrain:
                    input = batch[3:10] + (None,) + batch[10:]
                else:
                    input = batch[10:13] + batch[7:10] + batch[13:15] + (realigned_p, realigned_e, realigned_d, batch[-1])
                output = model(*(input))

                if step % word_step == 0:
                    # Get predicted audio
                    mels = [output[1][i, :output[9][i]].transpose(0,1) for i in range(batch_size)]
                    wav_predictions = vocoder_infer(
                        mels,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    loss_input = (batch[1],) + (batch[6],) + batch[10:]
                    loss_predictions = output + (wav_predictions,)
                else:
                    loss_input = (None, batch[6]) + batch[10:]
                    loss_predictions = output + (None,)
                
                # Cal Loss
                if pretrain:
                    losses = Loss(loss_input, loss_predictions, "to_src")
                else:
                    loss_input = (batch[7],) + (realigned_p, realigned_e, realigned_d)
                    losses = Loss(loss_input, output, "to_tgt")

                    # TODO: Implement to-src loss and realignment above

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])
                
            step += 1

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, Prosody Loss: {:.4f}, Word Loss: {:.4f}".format(
                *([step] + [l for l in loss_means])
    )

    if logger is not None:
        if pretrain:
            targets = (batch[0],) +(batch[6],) + batch[10:]
            predictions = (output[1],) + output[8:10]
        else:
            raise NotImplementedError
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
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