import os
import json

import torch
import numpy as np

import hifigan
import bigvgan
from model import ScheduledOptim, FastSpeech2Pros, ProsLearner, Discriminator


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    
    pretrain = train_config['pretrain']

    if "synthesizer" in model_config:
        if model_config["synthesizer"]["model"] == "FastSpeech2":
            model = FastSpeech2(preprocess_config, model_config).to(device)
        elif model_config["synthesizer"]["model"] == "FastSpeech2Pros":
            model = FastSpeech2Pros(preprocess_config, model_config, pretrain).to(device)
    elif "pros_learner" in model_config:
        model = ProsLearner(preprocess_config, model_config).to(device)
    else:
        raise ValueError("Model type not supported. Check model config.")

    if not pretrain:
        if args.from_pretrained_ckpt:
            pretrain_dir = train_config["path"]["ckpt_path"].replace("train", "pretrain")
            ckpt_path = os.path.join(pretrain_dir,
                                     "{}.pth.tar".format(args.from_pretrained_ckpt))
            
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
        
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if not pretrain:
            if args.from_pretrained_ckpt:
                scheduled_optim.load_state_dict(ckpt["optimizer"])
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_discriminator(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    pretrain = train_config['pretrain']

    model = Discriminator(preprocess_config).to(device)

    if not pretrain:
        if args.from_pretrained_ckpt:
            pretrain_dir = train_config["path"]["ckpt_path"].replace("train", "pretrain")
            ckpt_path = os.path.join(pretrain_dir,
                                     "disc_{}.pth.tar".format(args.from_pretrained_ckpt))
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt["discriminator"])

    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "disc_{}.pth.tar".format(args.restore_step),
        )
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            # Add 'module.' prefix to each key
            modified_ckpt = {key.replace('module.', ''): value for key, value in ckpt['discriminator'].items()}
            try:
                model.load_state_dict(ckpt["discriminator"])
            except:
                model.load_state_dict(modified_ckpt)

    if train:
        warm_up_step = train_config['optimizer']['warm_up_step']
        d_initial_lr, d_max_lr = train_config["d_optimizer"]["d_initial_lr"], train_config["d_optimizer"]["d_max_lr"]
        betas = train_config["d_optimizer"]["betas"]
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=d_initial_lr,
            betas=betas,
        )
        if not pretrain:
            if args.from_pretrained_ckpt and os.path.exists(ckpt_path):
                optimizer.load_state_dict(ckpt["optimizer"])
        if args.restore_step and os.path.exists(ckpt_path):
            optimizer.load_state_dict(ckpt["optimizer"])

        # Define a lambda function to increase the learning rate up to 10,000 steps and then decrease
        def lr_lambda(step):
            if step < warm_up_step:
                # Linearly increase the learning rate to max_lr
                return d_initial_lr + (d_max_lr - d_initial_lr) * (step / warm_up_step)
            elif step < warm_up_step*10:
                # Linearly decrease the learning rate back to initial_lr
                return d_max_lr - (d_max_lr - d_initial_lr) * ((step - warm_up_step) / (warm_up_step*10 - warm_up_step))
            else:
                # After total_steps, maintain the initial learning rate
                return d_initial_lr
            
        # Define the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        model.train()
        return model, optimizer, scheduler

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)

    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar", map_location=device)
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar", map_location=device)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    elif name == "BigVGAN":
        with open("bigvgan/config.json", "r") as f:
            config = json.load(f)
        config = bigvgan.AttrDict(config)
        vocoder = bigvgan.Generator(config)
        ckpt = torch.load("bigvgan/g_05000000", map_location=device)

        vocoder.load_state_dict(ckpt['generator'])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]

    if isinstance(mels, list):
        wavs = []
        for mel in mels:
            if mel.shape[1] == 0:
                wavs.append(None)
                continue
            if name == "MelGAN":
                wav = vocoder.inverse(mel / np.log(10))
            elif name == "HiFi-GAN":
                wav = vocoder(mel).squeeze(1).squeeze(0)
            elif name == "BigVGAN":
                wav = vocoder(mel.unsqueeze(0)).squeeze(1).squeeze(0)
            # wav = wav * preprocess_config["preprocessing"]["audio"]["max_wav_value"]     
            wavs.append(wav)
        if len(wavs) == 0:
            return None

    else:
        with torch.no_grad():
            if name == "MelGAN":
                wavs = vocoder.inverse(mels / np.log(10))
            elif name == "HiFi-GAN":
                if mels.shape[2] == 0:
                    return None
                wavs = vocoder(mels).squeeze(1)

        wavs = (
            wavs.cpu().numpy()
            * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
        ).astype("int16")
        
        wavs = [wav for wav in wavs if np.shape(wav)[0] > 0]

        for i in range(len(mels)):
            if lengths is not None:
                wavs[i] = wavs[i][: lengths[i]]

    return wavs


