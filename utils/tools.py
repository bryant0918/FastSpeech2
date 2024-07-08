import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt


matplotlib.use("Agg")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def to_device(data, device):
    # For Pros Full training
    if len(data) == 23:
        (ids, raw_texts, raw_translations, speakers, src_langs, texts, text_lens, max_text_lens, mels, mel_lens,
        max_mel_lens, tgt_langs, translations, translation_lens, max_translation_len, speaker_embeddings, alignments, 
        pitches, energies, durations, realigned_p, realigned_e, realigned_d) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        text_langs = torch.from_numpy(src_langs).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(text_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        translation_langs = torch.from_numpy(tgt_langs).long().to(device)
        translations = torch.from_numpy(translations).long().to(device)
        translation_lens = torch.from_numpy(translation_lens).to(device)

        speaker_embeddings = np.array(speaker_embeddings)
        speaker_embeddings = torch.from_numpy(speaker_embeddings).to(device)

        alignments = alignments.to(device)
        pitches = pitches.to(device)
        energies = energies.to(device)
        durations = durations.to(device)
        realigned_p = realigned_p.to(device)
        realigned_e = realigned_e.to(device)
        realigned_d = realigned_d.to(device)

        return (ids, raw_texts, raw_translations, speakers, text_langs, texts, src_lens, max_text_lens, mels, mel_lens,
                max_mel_lens, translation_langs, translations, translation_lens, max_translation_len, speaker_embeddings, 
                alignments, pitches, energies, durations, realigned_p, realigned_e, realigned_d)

    # For Pros Pretraining
    if len(data) == 14:
        (ids, raw_texts, speakers, text_langs, texts, src_lens, max_src_len, mels, mel_lens, max_mel_len, speaker_embeddings,
        pitches, energies, durations) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        text_langs = torch.from_numpy(text_langs).long().to(device)

        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)

        speaker_embeddings = torch.from_numpy(np.array(speaker_embeddings)).to(device)

        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)

        return (ids, raw_texts, speakers, text_langs, texts, src_lens, max_src_len, mels, mel_lens, max_mel_len, speaker_embeddings,
                pitches, energies, durations)

    # For Pros Synth
    if len(data) == 8:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len, speaker_embs, mels) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        if len(speaker_embs) > 1:
            speaker_embs = torch.cat(speaker_embs, dim=0).to(device)
        else:
            speaker_embs = speaker_embs.to(device)
        if len(mels) > 1:
            print(len(mels))
            mels = torch.cat(mels, dim=0).to(device)
        else:
            mels = mels.to(device)

        return ids, raw_texts, speakers, texts, src_lens, max_src_len, speaker_embs, mels

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return ids, raw_texts, speakers, texts, src_lens, max_src_len



def log(
    logger, step=None, losses=None, fig=None, lr=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)
        logger.add_scalar("Loss/prosody_loss", losses[6], step)
        logger.add_scalar("Loss/word_loss", losses[7], step)
        logger.add_scalar("Loss/full_duration_loss", losses[8], step)

    if lr is not None:
        logger.add_scalar("Learning_rate", lr, step)

    if fig is not None:
        logger.add_figure(tag, fig, global_step=step)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            global_step=step,
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(src_gt, tgt_targets, src_targets, predicted_tgt, predicted_src, vocoder, model_config, preprocess_config):    
    (   basename, 
        src_gt_len,
        src_gt_mel,
        src_gt_mel_lens,
        src_gt_pitch,
        src_gt_energy,
        src_gt_duration,
    ) = src_gt
    
    (   
        tgt_mel_target, 
        tgt_len,
        tgt_pitch_target, 
        tgt_energy_target, 
        tgt_duration_target,
    ) = tgt_targets

    (   
        src_mel_target, 
        src_pitch_target, 
        src_energy_target, 
        src_duration_target,
    ) = src_targets

    (   tgt_mel_prediction,
        tgt_src_len,
        tgt_mel_len,
    ) = predicted_tgt

    (   src_mel_prediction,
        src_src_len,
        src_mel_len,
    ) = predicted_src
    
    src_gt_len = src_gt_len[0].item()
    src_gt_mel_len = src_gt_mel_lens[0].item()
    tgt_len = tgt_len[0].item()
    tgt_mel_len = tgt_mel_len[0].item()
    src_mel_len = src_mel_len[0].item()

    src_gt_mel = src_gt_mel[0, :src_gt_mel_len].detach().transpose(0, 1)

    tgt_mel_prediction = tgt_mel_prediction[0, :tgt_mel_len].detach().transpose(0, 1)
    src_mel_prediction = src_mel_prediction[0, :src_mel_len].detach().transpose(0, 1)
    
    src_gt_duration = src_gt_duration[0, :src_gt_len].detach().cpu().numpy()
    tgt_duration_target = tgt_duration_target[0, :tgt_len].detach().cpu().numpy()

    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        src_gt_pitch = src_gt_pitch[0, :src_gt_len].detach().cpu().numpy()
        src_gt_pitch = expand(src_gt_pitch, src_gt_duration)

        tgt_pitch_target = tgt_pitch_target[0, :tgt_len].detach().cpu().numpy()
        tgt_pitch_target = expand(tgt_pitch_target, tgt_duration_target)

    else:
        src_gt_pitch = src_gt_pitch[0, :src_gt_mel_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        src_gt_energy = src_gt_energy[0, :src_gt_len].detach().cpu().numpy()
        src_gt_energy = expand(src_gt_energy, src_gt_duration)

        tgt_energy_target = tgt_energy_target[0, :tgt_len].detach().cpu().numpy()
        tgt_energy_target = expand(tgt_energy_target, tgt_duration_target)

    else:
        src_gt_energy = src_gt_energy[0, :src_gt_mel_len].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    fig = plot_mel(
        [
            (tgt_mel_prediction.cpu().numpy(), tgt_pitch_target, tgt_energy_target),
            (src_mel_prediction.cpu().numpy(), src_gt_pitch, src_gt_energy),
            (src_gt_mel.cpu().numpy(), src_gt_pitch, src_gt_energy),
            
        ],
        stats,
        ["Synthetized TGT Spectrogram", "Synthesized SRC Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer
        tgt_wav_prediction = vocoder_infer(
            tgt_mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]

        src_wav_prediction = vocoder_infer(
            src_mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]

        wav_reconstruction = vocoder_infer(
            src_gt_mel.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        tgt_wav_prediction = src_wav_prediction = wav_reconstruction = None

    return fig, tgt_wav_prediction, src_wav_prediction, wav_reconstruction, basename


def synth_one_sample_pretrain(targets, predictions, vocoder, model_config, preprocess_config):    
    (   basename, 
        mel_target, 
        pitch_target, 
        energy_target, 
        duration_target,
    ) = targets

    (   mel_prediction,
        src_len,
        mel_len,
    ) = predictions
    
    src_len = src_len[0].item()
    mel_len = mel_len[0].item()

    mel_target = mel_target[0, :mel_len].detach().transpose(0, 1)

    mel_prediction = mel_prediction[0, :mel_len].detach().transpose(0, 1)
    
    duration = duration_target[0, :src_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = pitch_target[0, :src_len].detach().cpu().numpy()
        pitch = expand(pitch, duration)
    else:
        pitch = pitch_target[0, :mel_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = energy_target[0, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = energy_target[0, :mel_len].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    # # To Test BigVGAN vs HIFI-GAN ## BigVGAN Not better
    # id = torch.randint(0,10,(1,)).item()
    # with open(f'output/result/LJSpeech/pretrain/mel_prediction{id}.npy', 'wb') as f:
    #     np.save(f, mel_prediction.cpu().numpy())
    # with open(f'output/result/LJSpeech/pretrain/mel_target{id}.npy', 'wb') as f:
    #     np.save(f, mel_target.cpu().numpy())
    
    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]

        # id = torch.randint(0,10,(1,)).item()
        # # Save audio file
        # wavfile.write(f'output/result/LJSpeech/pretrain/wav_reconstruction{id}.wav', preprocess_config["preprocessing"]["audio"]["sampling_rate"], wav_reconstruction)
        # wavfile.write(f'output/result/LJSpeech/pretrain/wav_prediction{id}.wav', preprocess_config["preprocessing"]["audio"]["sampling_rate"], wav_prediction)

    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path):

    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[8][i].item()
        mel_len = predictions[9][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[5][i, :src_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions[2][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions[3][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[3][i, :mel_len].detach().cpu().numpy()

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(path, "{}.png".format(basename)))
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_inhomogeneous_2D(inputs, PAD=0):
    def pad_array(array, max_array_length, max_element_length, PAD):
        padded_array = []
        for element in array:
            padded_element = np.pad(
                element, (0, max_element_length - len(element)), mode="constant", constant_values=PAD
            )
            padded_array.append(padded_element)
        
        while len(padded_array) < max_array_length:
            padded_array.append([PAD] * max_element_length)
        
        return np.array(padded_array)
    
    max_array_length = max(len(array) for array in inputs)
    max_element_length = max(len(element) for array in inputs for element in array)
    
    padded = np.array([pad_array(array, max_array_length, max_element_length, PAD) for array in inputs])
    
    return padded


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def flip_mapping(tgt_to_src_mappings, src_seq_len):

        batch = []
        for tgt_to_src_mapping in tgt_to_src_mappings:
            # Initialize a list of lists to store the target to source mappings
            src_to_tgt_mapping = [[] for _ in range(src_seq_len - 1)]
            
            # Iterate through each source index and its corresponding target indices
            for tgt_idx, src_indices in enumerate(tgt_to_src_mapping):
                for src_idx in src_indices:
                    if src_idx > 0:
                        src_to_tgt_mapping[src_idx].append(tgt_idx)

            batch.append(src_to_tgt_mapping)

        src_to_tgt_mappings_padded = pad_inhomogeneous_2D(batch)
        src_to_tgt_mappings_padded = torch.from_numpy(src_to_tgt_mappings_padded).int().to(device)
                
        return src_to_tgt_mappings_padded


def realign_p_e_d(alignments, p_e_d):
        new_ped = torch.zeros(p_e_d.size(0), len(alignments[0])+1, device=p_e_d.device)
        for b, alignment in enumerate(alignments):
            for j, src_indices in enumerate(alignment):
                new_ped[b][j+1] = torch.mean(torch.tensor([p_e_d[b][i] for i in src_indices], dtype=torch.float32))
        return new_ped


def custom_round(x):
    # Round x in (0, .5] up to 1, keep 0 as is
    mask = (x <= 0.5) & (x > 0)
    x[mask] = torch.ceil(x[mask])
    # Add pos/neg eps randomely so half the .5's round up and half down
    eps = (torch.rand_like(x) - .5)/100
    return torch.round(x + eps).int()