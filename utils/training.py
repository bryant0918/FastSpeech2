import torch

from utils.tools import flip_mapping, realign_p_e_d, custom_round
from utils.model import vocoder_infer



def loop(preprocess_config, model_config, batch, model, Loss, vocoder, step, word_step):
    # Forward pass: Src to Tgt
    input = (batch[4],) + batch[12:15] + batch[8:11] + batch[15:17] + batch[20:] + (batch[19],)
    output_tgt = model(*(input))

    log_duration_targets = torch.log(batch[-1].float() + 1)
    # For calculating Word Loss
    if step % word_step == 0:
        mels = [output_tgt[1][i, :output_tgt[9][i]].transpose(0,1) for i in range(batch_size)]
        wav_predictions = vocoder_infer(
            mels,
            vocoder,
            model_config,
            preprocess_config,
        )
        loss_input = (batch[2],) + batch[8:10] + batch[20:22] + (log_duration_targets,)
        loss_predictions = output_tgt + (wav_predictions,)
    else:
        loss_input = (None,) + batch[8:10] + batch[20:22] + (log_duration_targets,)
        loss_predictions = output_tgt + (None,)
    
    # Calculate loss for Src to Tgt
    losses_src_to_tgt = Loss(loss_input, loss_predictions, "to_tgt")
    
    alignments = flip_mapping(batch[16], batch[5].shape[1])

    # realign p,e,d targets back to src space
    re_realigned_p = realign_p_e_d(alignments, output_tgt[2])
    re_realigned_e = realign_p_e_d(alignments, output_tgt[3])
    realigned_log_d = realign_p_e_d(alignments, output_tgt[4])
    re_realigned_d = torch.clamp(torch.exp(realigned_log_d) - 1, min=0)
    re_realigned_d = custom_round(re_realigned_d)

    # Forward pass: Tgt to Src (so tgt is now src and src is now tgt)
    output_src = model(langs=batch[11], texts=batch[5], text_lens=batch[6], max_text_len=batch[7],
                        mels=output_tgt[1], mel_lens=output_tgt[9], max_mel_len=batch[10],
                        speaker_embs=batch[15], alignments=alignments, p_targets=re_realigned_p, 
                        e_targets=re_realigned_e, d_targets=re_realigned_d, d_src=batch[-1])

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
        loss_input = (None,) + batch[8:10]+ (re_realigned_p, re_realigned_e, realigned_log_d)
        loss_predictions = output_src[:10] + (output_tgt[10],) + (output_src[11],) + (None,)

    # Calculate loss for Tgt to Src
    losses_tgt_to_src = Loss(loss_input, loss_predictions, "to_src")
    return losses_src_to_tgt, losses_tgt_to_src, output_tgt, output_src
