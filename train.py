import argparse
import os

import numpy as np
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample, flip_mapping, realign_p_e_d
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

    # # Debugging dataset
    # print("loader: ", loader)
    # for batches in loader:
    #     for batch in batches:
    #         batch = to_device(batch, device)
    #         print("batch", batch[8], batch[9])
    #         input = batch[3:]
    #         print("input", input[6], input[7])

    #         raise NotImplementedError


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

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    torch.autograd.set_detect_anomaly(True)

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)

        for batches in loader:
            for batch in batches:
                batch = to_device(batch, device)
                # batch = (ids, raw_texts, raw_translations, speakers, texts, src_lens, max_text_lens, mels, mel_lens,
                #          max_mel_lens, translations, translation_lens, max(translation_lens), speaker_embeddings, alignments, pitches, energies,
                #          durations)

                if step == 4:
                    raise NotImplementedError
                # print()
                # print(f"texts shape: {batch[4].shape}")
                # print(f"src_lens shape: {batch[5].shape}")
                # print(f"max_src_len: {batch[6]}{type(batch[6])}")
                # print(f"mels shape: {batch[7].shape}")
                # print(f"mel_lens shape: {batch[8].shape}")
                # print(f"max_mel_len: {batch[9]}{type(batch[9])}")
                # print(f"translations shape: {batch[10].shape}")
                # print(f"translation_lens shape: {batch[11].shape}{batch[11]}")
                # print(f"max_translation_len: {batch[12]}{type(batch[12])}")
                # print(f"speaker_embs shape: {batch[13].shape}")
                # print(f"alignments shape: {batch[14].shape}")
                # print(f"pitches shape: {batch[15].shape}")   
                # print(f"energies shape: {batch[16].shape}")
                # print(f"durations shape: {batch[17].shape}")
                # print()

                # flip mapping
                # realign pitch energy and duration here for target use batched for source below
                realigned_p = realign_p_e_d(batch[14], batch[15])
                realigned_e = realign_p_e_d(batch[14], batch[16])
                realigned_d = realign_p_e_d(batch[14], batch[17])
                # print("Realigned pitches: ", realigned_p.shape)
                # print("Realigned energies: ", realigned_e.shape)
                # print("Realigned durations: ", realigned_d.shape)

                # print("PITCH", realigned_p)


                # Forward
                if batch is None:
                    print("Batch is None")
                    
                input = batch[10:13] + batch[7:10] + batch[13:15] + (realigned_p, realigned_e, realigned_d, batch[-1])
                # input = batch[4:10] + batch[13:]

                # try:
                #     output = model(*(batch[3:]))
                #     print("Good")
                #     if step == 5:
                #         raise NotImplementedError
                # except:
                #     print("Bad")
                #     print(batch[1])
                #     step += 1
                #     if step >= 5:
                #         raise NotImplementedError
                #     continue

                # Forward pass: Src to Tgt
                # input_english = batch['english_audio']
                output_tgt = model(*(input))
                
                # Calculate loss for Src to Tgt
                
                print("\nCalculating Loss for SRC to TGT")
                loss_input = (batch[7],) + (realigned_p, realigned_e, realigned_d)
                losses_src_to_tgt = Loss(loss_input, output_tgt, "to_tgt")
                total_loss_src_to_tgt = losses_src_to_tgt[0]
                print("total_loss_src_to_tgt: ", total_loss_src_to_tgt)


                # output_tgt = (output, postnet_output, p_predictions, e_predictions, log_d_predictions, d_rounded, src_masks,
                #               mel_masks, src_lens, mel_lens, adjusted_e_tgt, )
                # We need:
                # (speakers, texts, src_lens, max_src_len, mels=None, mel_lens=None, max_mel_len=None,
                # translations=None, translation_lens=None, max_translation_len=None, speaker_embs=None,
                # alignments=None, p_targets=None, e_targets=None, d_targets=None, prev_e=None, p_control=1.0,
                # e_control=1.0, d_control=1.0,)
                # I'll do:
                # mel_lens = output_tgt[9]
                # max_mel_len = max(mel_lens)  # TODO: this is only for batch though...
                # mels_tgt = output_tgt[1]
                # alignments = reverse_alignments()
                # (speakers=batch[3], texts=translations, src_lens=translation_lens, max_src_len=max_translation_len,
                #  mels=mels_tgt, mel_lens=mel_lens, max_mel_len=max(mel_lens),
                #  translations=None, translation_lens=None, max_translation_lens=None, speaker_embs=batch[13],
                #  alignments=alignments, p_targets=output_tgt[2], e_targets=output_tgt[3], d_targets=output_tgt[4],
                #  prev_e=None, p_control=1.0, e_control=1.0, d_control=1.0,)
                
                alignments = flip_mapping(batch[14])

                # reverse_input = (batch[3], batch[10], batch[11], batch[12], output_tgt[1], output_tgt[9], max(output_tgt[9]),
                                #  None, None, None, batch[13], alignments, output_tgt[2], output_tgt[3], output_tgt[4], None, 1.0, 1.0, 1.0)
                
                max_mel_len = np.int64(max(output_tgt[9]).cpu().numpy())

                print()
                print(f"texts shape: {batch[4].shape}")
                # print(f"src_lens shape: {batch[5].shape}")
                # print(f"max_src_len: {batch[6]}{type(batch[12])}")
                # print(f"mels shape: {output_tgt[1].shape}")
                # print(f"mel_lens shape: {output_tgt[9].shape}")
                # print(f"max_mel_len: {max_mel_len}{type(max_mel_len)}")
                # print(f"speaker_embs shape: {batch[13].shape}")
                print(f"alignments shape: {alignments.shape}")
                print(f"p_targets shape: {output_tgt[2].shape}")  # Pitches energies and durations should be same size as texts
                print(f"e_targets shape: {output_tgt[3].shape}")
                print(f"d_targets shape: {output_tgt[4].shape}")
                # print()
                # print("Pitches: ", batch[15])

                # realign p,e,d targets back to src space
                realigned_p = realign_p_e_d(alignments, output_tgt[2])
                realigned_e = realign_p_e_d(alignments, output_tgt[3])
                realigned_d = realign_p_e_d(alignments, output_tgt[4])
                print("Realigned pitches: ", realigned_p.shape)
                print("Realigned energies: ", realigned_e.shape)
                print("Realigned durations: ", realigned_d.shape)
                print()

                # # Forward pass: Tgt to Src (so tgt is now src and src is now tgt)
                output_src = model(texts=batch[4], src_lens=batch[5], max_src_len=batch[6],
                                   mels=output_tgt[1], mel_lens=output_tgt[9], max_mel_len=max_mel_len,
                                   speaker_embs=batch[13], alignments=alignments, p_targets=realigned_p, 
                                   e_targets=realigned_e, d_targets=realigned_d, d_src=output_tgt[4])
                
                # # Calculate loss for Tgt to Src
                print("\nCalculating Loss for TGT to SRC")
                losses_tgt_to_src = Loss(batch, output_src, "to_src")
                total_loss_tgt_to_src = losses_tgt_to_src[0]

                # Combine the losses
                total_loss = (total_loss_src_to_tgt + total_loss_tgt_to_src) / 2
                print("Total Loss: ", total_loss)
                # # Cal Loss
                # losses = Loss(batch, output)
                # total_loss = losses[0]

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
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
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
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
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
