import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.model import get_model, get_vocoder, get_param_num, get_discriminator
from utils.tools import to_device, log, synth_one_sample_pretrain
from utils.training import pretrain_loop
from model import FastSpeech2Loss
from dataset import PreTrainDataset

from evaluate import evaluate_pretrain

def setup(rank, world_size, ip):
    os.environ['MASTER_ADDR'] = ip      # Multi-Node (Cluster): Use the IP address of the master node when GPUs are distributed across multiple machines.
    os.environ['MASTER_PORT'] = '12355' # Ensure this port is free
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, args, configs, world_size):
    print("Prepare training ...")
    setup(rank, world_size, args.ip)

    preprocess_config, preprocess_config2, model_config, train_config = configs

    # Get dataset
    dataset = PreTrainDataset("train.txt", preprocess_config, preprocess_config2, train_config, sort=True, drop_last=True)
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 1  # Set this larger than 1 (4) to enable sorting in Dataset
    # assert batch_size * group_size < len(dataset)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        # shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        sampler=sampler,
    )

    # Prepare model
    model, optimizer = get_model(args, configs[1:], rank, train=True)
    model = DDP(model, device_ids=[rank])  # find_unused_parameters=True
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config, train_config).to(rank)
    print("Number of FastSpeech2 Parameters:", num_param)

    # indices_to_print = [0, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108]
    # indices_to_print = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106]
    # named_parameters = list(model.named_parameters())
    # for idx in indices_to_print:
    #     print(f"Name: {named_parameters[idx][0]}, Shape: {named_parameters[idx][1].shape}")

    # Prepare discriminator
    discriminator, d_optimizer, d_scheduler = get_discriminator(args, configs[1:], rank, train=True)
    discriminator = DDP(discriminator, device_ids=[rank])
    criterion_d = nn.BCELoss()
    discriminator_params = get_param_num(discriminator)
    print("Number of Discriminator Parameters:", discriminator_params)
    print("Total Parameters:", num_param + discriminator_params)
    
    # Load vocoder
    vocoder = get_vocoder(model_config, f"cuda:{rank}")

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

    # Only use while debugging
    # torch.autograd.set_detect_anomaly(True)
    import time
    start = time.time()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)

        for batches in loader:
            try:
                for batch in batches:
                    batch = to_device(batch, rank)
                    # batch = (ids, raw_texts, speakers, langs, texts, src_lens, max_src_len, mels, mel_lens, max_mel_len, 
                    #           speaker_embeddings, pitches, energies, durations)
                    
                    if step == 200:
                        print("Time taken for 200 steps: ", time.time() - start)
                        raise Exception("Stop")

                    # Forward
                    losses, output, d_loss = pretrain_loop(preprocess_config, model_config, batch, model, Loss, discriminator, criterion_d,
                                                            vocoder, step, word_step, rank, True, d_optimizer, discriminator_step, warm_up_step)

                    # Backward
                    total_loss = losses[0] / grad_acc_step

                    # If mels is None then set requires grad for prosody extractor parameters to False
                    
                    total_loss.backward()

                    # print("linear3 weight grad", model.module.prosody_predictor.linear3.weight.grad)  # Should not be None
                    # print("linear3 bias grad", model.module.prosody_predictor.linear3.bias.grad) 

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
                        cleanup()
                        quit()
                    step += 1
                    outer_bar.update(1)
            except KeyboardInterrupt:
                if step > 20:
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
                cleanup()
                raise
            except Exception as e:
                print("Training interrupted by other exception.")
                print(e)
                cleanup()
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
    parser.add_argument("--ip", type=str, default="localhost", help="IP address of master node")
    args = parser.parse_args()

    # Read Configs
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    preprocess2_config = yaml.load(open(args.preprocess_config2, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, preprocess2_config, model_config, train_config)

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(args, configs, world_size,), nprocs=world_size, join=True)


# sed -i "s/'localhost'/'192.222.52.202'/" pretrain_ddp.py


# -- Process 7 terminated with the following error: Traceback (most recent call last): File "/opt/conda/envs/Emotiv/lib/python3.10/site-packages/torch/multiprocessing/spawn.py", line 75, in _wrap fn(i, *args) File "/FastSpeech2/pretrain_ddp.py", line 32, in main setup(rank, world_size) File "/FastSpeech2/pretrain_ddp.py", line 25, in setup dist.init_process_group("nccl", rank=rank, world_size=world_size) File "/opt/conda/envs/Emotiv/lib/python3.10/site-packages/torch/distributed/c10d_logger.py", line 75, in wrapper return func(*args, **kwargs) File "/opt/conda/envs/Emotiv/lib/python3.10/site-packages/torch/distributed/c10d_logger.py", line 89, in wrapper func_return = func(*args, **kwargs) File "/opt/conda/envs/Emotiv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py", line 1305, in init_process_group store, rank, world_size = next(rendezvous_iterator) File "/opt/conda/envs/Emotiv/lib/python3.10/site-packages/torch/distributed/rendezvous.py", line 246, in _env_rendezvous_handler store = _create_c10d_store(master_addr, master_port, rank, world_size, timeout, use_libuv) File "/opt/conda/envs/Emotiv/lib/python3.10/site-packages/torch/distributed/rendezvous.py", line 174, in _create_c10d_store return TCPStore( torch.distributed.DistNetworkError: Connection reset by peer