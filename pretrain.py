import argparse
import os

import numpy as np
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import multiprocessing as mp

from utils.model import get_model, get_vocoder, get_param_num, vocoder_infer, get_discriminator
from utils.tools import to_device, log, synth_one_sample_pretrain
from utils.training import pretrain_loop
from model import FastSpeech2Loss, Discriminator
from dataset import PreTrainDataset

from evaluate import evaluate_pretrain

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device", device)

def main(args, configs):
    print("Prepare training ...")

    preprocess_config, preprocess_config2, model_config, train_config = configs

    # Get dataset
    dataset = PreTrainDataset("train.txt", preprocess_config, preprocess_config2, train_config, sort=True, drop_last=True)
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 1  # Set this larger than 1 (4) to enable sorting in Dataset
    # assert batch_size * group_size < len(dataset)
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        # sampler=sampler,
    )

    # Prepare model
    model, optimizer = get_model(args, configs[1:], device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config, train_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Prepare discriminator
    discriminator, d_optimizer, d_scheduler = get_discriminator(args, configs[1:], device, train=True)
    discriminator = nn.DataParallel(discriminator)
    criterion_d = nn.BCELoss()
    discriminator_params = get_param_num(discriminator)
    print("Number of Discriminator Parameters:", discriminator_params)
    print("Total Parameters:", num_param + discriminator_params)
    
    # Load vocoder
    vocoder = get_vocoder(model_config, device)

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
    discriminator_step = train_config["step"]["discriminator_step"]
    warm_up_step = train_config['optimizer']['warm_up_step']

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    torch.autograd.set_detect_anomaly(True)

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)

        for batches in loader:
            try:
                for batch in batches:
                    batch = to_device(batch, device)
                    # batch = (ids, raw_texts, speakers, langs, texts, src_lens, max_src_len, mels, mel_lens, max_mel_len, 
                    #           speaker_embeddings, pitches, energies, durations)
                    
                    # if step == 15010:
                    #     raise Exception("Stop")

                    # Forward
                    losses, output, d_loss = pretrain_loop(preprocess_config, model_config, batch, model, Loss, discriminator, criterion_d,
                                                            vocoder, step, word_step, device, True, d_optimizer, discriminator_step, warm_up_step)

                    # Backward
                    total_loss = losses[0] / grad_acc_step
                    total_loss.backward()

                    d_scheduler.step()

                    if step % grad_acc_step == 0:
                        # Clipping gradients to avoid gradient explosion
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                        # Update weights
                        optimizer.step_and_update_lr()
                        optimizer.zero_grad()

                    if step % log_step == 0:
                        losses = [l.item() for l in losses]
                        losses.append(d_loss)
                        message1 = "Step {}/{}, ".format(step, total_step)
                        message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, Prosody Loss: {:.4f}, Word Loss: {:.4f}, Full Duration Loss: {:.4f}, G Loss: {:.4f}, Prosody Reg: {:.4f}, D Loss: {:.4f}".format(
                            *losses
                        )

                        with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                            f.write(message1 + message2 + "\n")

                        outer_bar.write(message1 + message2)

                        log(train_logger, step, losses=losses, lr=optimizer.get_lr())

                    if step % synth_step == 0:
                        targets = (batch[0],) +(batch[7],) + batch[11:]
                        predictions = (output[1],) + output[8:10]
                        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample_pretrain(
                            targets,
                            predictions,
                            vocoder,
                            model_config,
                            preprocess_config,
                        )
                        log(
                            train_logger,
                            step,
                            fig=fig,
                            tag="Training/step_{}_{}".format(step, tag),
                        )
                        sampling_rate = preprocess_config["preprocessing"]["audio"][
                            "sampling_rate"
                        ]
                        log(
                            train_logger,
                            step,
                            audio=wav_reconstruction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_reconstructed".format(step, tag),
                        )
                        log(
                            train_logger,
                            step,
                            audio=wav_prediction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_synthesized".format(step, tag),
                        )

                    if step % val_step == 0:
                        model.eval()
                        discriminator.eval()
                        message = evaluate_pretrain(model, discriminator, step, configs, val_logger, vocoder)
                        with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                            f.write(message + "\n")
                        outer_bar.write(message)

                        model.train()
                        discriminator.train()

                    if step % save_step == 0:
                        print("Saving checkpoints")
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

                        torch.save(
                            {
                                "discriminator": discriminator.state_dict(),
                                "optimizer": d_optimizer.state_dict(),
                            },
                            os.path.join(
                                train_config["path"]["ckpt_path"],
                                "disc_{}.pth.tar".format(step),
                            ),
                        )

                    if step == total_step:
                        quit()
                    step += 1
                    outer_bar.update(1)
            except KeyboardInterrupt:
                print("Training interrupted -- Saving checkpoints")
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

                torch.save(
                    {
                        "discriminator": discriminator.state_dict(),
                        "optimizer": d_optimizer.state_dict(),
                    },
                    os.path.join(
                        train_config["path"]["ckpt_path"],
                        "disc_{}.pth.tar".format(step),
                    ),
                )
                raise
            except Exception as e:
                print("Training interrupted by other exception.")
                print(e)
                raise e

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("-p", "--preprocess_config", type=str,required=True, 
                        help="path to preprocess.yaml")
    parser.add_argument("-p2", "--preprocess_config2", type=str,required=False, 
                        help="path to second preprocess.yaml for other language")
    parser.add_argument("-m", "--model_config", type=str, required=True, help="path to model.yaml")
    parser.add_argument("-t", "--train_config", type=str, required=True, help="path to train.yaml")
    parser.add_argument("-w", "--num_workers", type=int, default=4, help="number of cpu workers for dataloader")
    args = parser.parse_args()

    # Read Configs
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    preprocess2_config = yaml.load(open(args.preprocess_config2, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, preprocess2_config, model_config, train_config)

    main(args, configs)