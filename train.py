import argparse
import os

import numpy as np
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num, vocoder_infer
from utils.tools import to_device, log, synth_one_sample, flip_mapping, realign_p_e_d, custom_round
from model import FastSpeech2Loss
from dataset import Dataset, TrainDataset

from evaluate import evaluate

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device", device)


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = TrainDataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 1  # Set this larger than 1 (4) to enable sorting in Dataset
    # assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    print(next(model.parameters()).device)    
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)
    print("Vocoder Loaded")

    # Init loggerc
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    word_step = train_config["step"]["word_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    torch.autograd.set_detect_anomaly(True)

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)

        for batches in loader:
            for batch in batches:
                batch = to_device(batch, device)
                # batch = (ids, raw_texts, raw_translations, speakers, text_langs, texts, src_lens, max_text_lens, mels, mel_lens,
                #           max_mel_lens, translation_langs, translations, translation_lens, max_translation_len, speaker_embeddings, 
                #           alignments, pitches, energies, durations)

                if step == 5:
                    raise NotImplementedError

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
                    loss_input = (batch[2],) + batch[8:10] + (realigned_p, realigned_e, log_duration_targets)
                    loss_predictions = output_tgt + (wav_predictions,)
                else:
                    loss_input = (None,) + batch[8:10] + (realigned_p, realigned_e, log_duration_targets)
                    loss_predictions = output_tgt + (None,)
                
                # Calculate loss for Src to Tgt
                losses_src_to_tgt = Loss(loss_input, loss_predictions, "to_tgt")
                total_loss_src_to_tgt = losses_src_to_tgt[0]
                
                alignments = flip_mapping(batch[16], batch[5].shape[1])
                                
                # output_tgt[4] is log_d_predictions
                # print("d_rounded: ", output_tgt[5])
                # print("log_d_predictions: ", output_tgt[4])
                # d_src = torch.clamp(torch.round(torch.exp(output_tgt[4]) - 1).long(), min=0)
                d_src = realigned_d
                
                # print("batdch[-1]", batch[-1])

                # realign p,e,d targets back to src space
                re_realigned_p = realign_p_e_d(alignments, output_tgt[2])
                re_realigned_e = realign_p_e_d(alignments, output_tgt[3])
                realigned_log_d = realign_p_e_d(alignments, output_tgt[4])
                re_realigned_d = torch.clamp(torch.exp(realigned_log_d) - 1, min=0)
                
                re_realigned_d = custom_round(re_realigned_d)

                # # Forward pass: Tgt to Src (so tgt is now src and src is now tgt)
                output_src = model(langs=batch[11], texts=batch[5], text_lens=batch[6], max_text_len=batch[7],
                                   mels=output_tgt[1], mel_lens=output_tgt[9], max_mel_len=batch[10],
                                   speaker_embs=batch[15], alignments=alignments, p_targets=re_realigned_p, 
                                   e_targets=re_realigned_e, d_targets=re_realigned_d, d_src=d_src)

                # For calculating Word Loss
                if step % word_step == 0:
                    mels = [output_src[1][i, :output_src[9][i]].transpose(0,1) for i in range(batch_size)]
                    wav_predictions = vocoder_infer(
                        mels,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    loss_input = (batch[1],) + batch[8:10] + (re_realigned_p, re_realigned_e, realigned_log_d)
                    loss_predictions = output_src[:10] + (output_tgt[10],) + (output_src[11],) + (wav_predictions,)
                else:
                    loss_input = (None,) + batch[8:10] + (re_realigned_p, re_realigned_e, realigned_log_d)
                    loss_predictions = output_src[:10] + (output_tgt[10],) + (output_src[11],) + (None,)
                
                # # Calculate loss for Tgt to Src
                losses_tgt_to_src = Loss(loss_input, loss_predictions, "to_src")
                total_loss_tgt_to_src = losses_tgt_to_src[0]

                # Combine the losses
                total_loss = (total_loss_src_to_tgt + total_loss_tgt_to_src) / 2

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()

                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [(l1.item() + l2.item())/2 for l1, l2 in zip(losses_src_to_tgt, losses_tgt_to_src)]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, Prosody Loss: {:.4f}, Word Loss: {:.4f}, Full Duration Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    # Want to see all 3 mels
                    src_gt = (batch[0], batch[6]) + (batch[8:10]) + batch[17:]
                    tgt_targets = (batch[8], batch[13]) + (realigned_p, realigned_e, realigned_d)
                    src_targets = (batch[8],) + batch[17:]
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
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=tgt_wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_tgt_synthesized".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=src_wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_src_synthesized".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    for name, param in model.named_parameters():
                        if 'beta' in name:
                            print(name, param)
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
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

    main(args, configs)

