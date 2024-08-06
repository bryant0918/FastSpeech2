import torch

from utils.tools import flip_mapping, realign_p_e_d, custom_round
from utils.model import vocoder_infer

import time


def loop(preprocess_config, model_config, batch, model, Loss, discriminator, criterion_d, 
         vocoder, step, word_step, device, training=False, d_optimizer=None, discriminator_step=1, warm_up_step=4000):
    batch_size = len(batch[0])

    # Custom feature selection to not over-rely on input audio
    mels = batch[8][:batch_size//2]
    input = (training, batch[4]) + batch[12:15] + (mels,) + batch[9:11] + batch[15:17] + batch[20:] + batch[19:]

    output_tgt = model(*(input))

    # Train Discriminator on only 1 of the two fakes
    first_fake = torch.bernoulli(torch.tensor(0.5))
    d_loss = torch.tensor(0.0, device=device)
    if first_fake:
        if step % discriminator_step == 0 or step >= warm_up_step:
            # Experiment Label Smoothing
            real_labels = torch.ones(batch_size, 1, device=device) * .95
            fake_labels = torch.zeros(batch_size, 1, device=device) + .05 * torch.ones(batch_size, 1, device=device)

            # Train Discriminator with real data
            # real_data_noisy = batch[7] + 0.1 * torch.randn_like(batch[7])
            real_outputs = discriminator(batch[8])
            d_loss_real = criterion_d(real_outputs, real_labels)

            # Train Discriminator with fake data (generated by TTS model)
            fake_outputs = discriminator(output_tgt[1].detach())  # detach to avoid backpropagating through the Generator
            d_loss_fake = criterion_d(fake_outputs, fake_labels)
            d_loss = d_loss_real + d_loss_fake
        
            if training:
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
    
    pred_generated = discriminator(output_tgt[1])

    log_duration_targets = torch.log(batch[-1].float() + 1)
    
    # For calculating Word Loss
    mel_size = max(batch_size//8, 1)
    if step % word_step == 0:
        # mels = [output_tgt[1][i, :output_tgt[9][i]].transpose(0,1) for i in range(mel_size)]
        mels = output_tgt[1][:mel_size].transpose(1,2)
        wav_predictions = vocoder_infer(
            mels,
            vocoder,
            model_config,
            preprocess_config,
            lengths=output_tgt[9][:mel_size],
            for_loss = True
        )
        loss_input = (batch[2][:mel_size],) + batch[8:10] + batch[20:22] + (log_duration_targets,)
        loss_predictions = output_tgt + (wav_predictions, pred_generated)
    else:
        loss_input = (None,) + batch[8:10] + batch[20:22] + (log_duration_targets,)
        loss_predictions = output_tgt + (None, pred_generated)
    
    # Calculate loss for Src to Tgt
    losses_src_to_tgt = Loss(loss_input, loss_predictions, "to_tgt", word_step)
    
    alignments = flip_mapping(batch[16], batch[5].shape[1])

    # realign p,e,d targets back to src space
    # re_realigned_p = realign_p_e_d(alignments, output_tgt[2])
    # re_realigned_e = realign_p_e_d(alignments, output_tgt[3])
    realigned_log_d = realign_p_e_d(alignments, output_tgt[4])
    # re_realigned_d = torch.clamp(torch.exp(realigned_log_d) - 1, min=0)
    # re_realigned_d = custom_round(re_realigned_d)

    # Changed to using rounded_d instead of log_d (when d_targets=True then this will just be realigned_d)
    re_realigned_d = custom_round(realign_p_e_d(alignments, output_tgt[5]))
    re_realigned_p = realign_p_e_d(alignments, batch[-3])
    re_realigned_e = realign_p_e_d(alignments, batch[-2])

    # print("source duration", len(batch[-4][0]), batch[-4][0][:15], sum(batch[-4][0]).item())
    # print("source log_duration", len(torch.log(batch[-4].float() + 1)[0]), torch.log(batch[-4].float() + 1)[0][:20])
    # print("source realigned_log_d", len(realigned_log_d[0]), realigned_log_d[0][:20])
    # print("source re_realigned_d: ", len(re_realigned_d[0]), re_realigned_d[0][:15], sum(re_realigned_d[0]).item())
    # realigned_rounded_d = custom_round(realign_p_e_d(alignments, output_tgt[5]))
    # print("source re_realigned_d_rounded: ", len(realigned_rounded_d[0]), realigned_rounded_d[0][:20], sum(realigned_rounded_d[0]).item())

    # re_realigned_d_no_round = realign_p_e_d(alignments, realign_p_e_d(batch[16], batch[19]))
    # print("source re_realigned_d_no_round: ", len(re_realigned_d_no_round[0]), re_realigned_d_no_round[0][:15], sum(re_realigned_d_no_round[0]).item())

    # Forward pass: Tgt to Src (so tgt is now src and src is now tgt)
    mels = output_tgt[1][:batch_size//2]
    output_src = model(training=training, langs=batch[11], texts=batch[5], text_lens=batch[6], max_text_len=batch[7],
                        mels=mels, mel_lens=output_tgt[9], max_mel_len=batch[10],
                        speaker_embs=batch[15], alignments=alignments, p_targets=re_realigned_p, 
                        e_targets=re_realigned_e, d_targets=re_realigned_d, d_src=batch[-1])

    if not first_fake:
        if step % discriminator_step == 0 or step >= warm_up_step:
            # Experiment Label Smoothing
            real_labels = torch.ones(batch_size, 1, device=device) * .95
            fake_labels = torch.zeros(batch_size, 1, device=device) + .05 * torch.ones(batch_size, 1, device=device)

            # Train Discriminator with real data
            # real_data_noisy = batch[7] + 0.1 * torch.randn_like(batch[7])
            real_outputs = discriminator(batch[8])
            d_loss_real = criterion_d(real_outputs, real_labels)

            # Train Discriminator with fake data (generated by TTS model)
            fake_outputs = discriminator(output_src[1].detach())  # detach to avoid backpropagating through the Generator
            d_loss_fake = criterion_d(fake_outputs, fake_labels)
            d_loss = d_loss_real + d_loss_fake
        
            if training:
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

    pred_generated = discriminator(output_src[1])

    # For calculating Word Loss
    if step % word_step == 0:
        # mels = [output_src[1][i, :output_src[9][i]].transpose(0,1) for i in range(mel_size)]
        mels = output_src[1][:mel_size].transpose(1,2)
        
        wav_predictions = vocoder_infer(
            mels,
            vocoder,
            model_config,
            preprocess_config,
            lengths=output_src[9][:mel_size],
            for_loss = True
        )
        loss_input = (batch[1][:mel_size],) + batch[8:10] + (re_realigned_p, re_realigned_e, realigned_log_d)
        loss_predictions = output_src[:10] + (output_tgt[10],) + output_src[11:] + (wav_predictions, pred_generated)
    else:
        loss_input = (None,) + batch[8:10]+ (re_realigned_p, re_realigned_e, realigned_log_d)
        loss_predictions = output_src[:10] + (output_tgt[10],) + output_src[11:] + (None, pred_generated)

    # Calculate loss for Tgt to Src
    losses_tgt_to_src = Loss(loss_input, loss_predictions, "to_src", word_step)

    return losses_src_to_tgt, losses_tgt_to_src, output_tgt, output_src, d_loss.item()

def pretrain_loop(preprocess_config, model_config, batch, model, Loss, discriminator, criterion_d,
                    vocoder, step, word_step, device, training=False,
                    d_optimizer=None, discriminator_step=1, warm_up_step=4000):
    batch_size = len(batch[0])

    # Custom feature selection to not over-rely on input audio
    mels = batch[7][:batch_size//2]
    input = (training,) + batch[3:7] + (mels,) + batch[8:10] + (batch[11], None) + batch[12:]
  
    start0 = time.time()
    # Generator Forward pass: Src to Src
    output = model(*(input))
    # print("Time for forward function: ", time.time() - start0)

    d_loss = torch.tensor(0.0, device=device)
    if step % discriminator_step == 0 or step >= warm_up_step:
        # Experiment Label Smoothing
        real_labels = .95 * torch.ones(batch_size, 1, device=device) 
        fake_labels = .05 * torch.ones(batch_size, 1, device=device)

        # Train Discriminator with real data
        # real_data_noisy = batch[7] + 0.1 * torch.randn_like(batch[7])
        real_input = batch[7].clone().detach()
        real_outputs = discriminator(real_input)
        d_loss_real = criterion_d(real_outputs, real_labels)

        # Train Discriminator with fake data (generated by TTS model)
        fake_outputs = discriminator(output[1].detach())  # detach to avoid backpropagating through the Generator
        
        d_loss_fake = criterion_d(fake_outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake

        if training:
            d_loss.backward()
            d_optimizer.step()
            d_optimizer.zero_grad()
    
    pred_generated = discriminator(output[1])

    log_duration_targets = torch.log(batch[-1] + 1)
    # For calculating Word Loss
    if step % word_step == 0:
        mel_size = max(batch_size//8, 1)
        # mels = [output[1][i, :output[9][i]].transpose(0,1) for i in range(mel_size)]
        mels = output[1][:mel_size].transpose(1,2)
        
        wav_predictions = vocoder_infer(
            mels,
            vocoder,
            model_config,
            preprocess_config,
            lengths=output[9][:mel_size],
            for_loss = True
        )
        loss_input = (batch[1][:mel_size],) + batch[7:9] + (batch[10],) + batch[12:14] + (log_duration_targets,)
        loss_predictions = output + (wav_predictions, pred_generated)
    else:
        loss_input = (None,) + batch[7:9] + (batch[10],) + batch[12:14] + (log_duration_targets,)
        loss_predictions = output + (None, pred_generated)
    
    start1 = time.time()
    # Calculate loss for Src to Src
    losses = Loss(loss_input, loss_predictions, "to_src", word_step)
    # print("Time for loss function: ", time.time() - start1)
    
    return losses, output, d_loss.item()