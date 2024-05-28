import os
import random
import json
import pickle

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import epitran
from itertools import chain

from text.ipadict import db
from text import _es_punctuations
import audio as Audio


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "alignments", "phone")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i
            for j, wav_name in enumerate(os.listdir(os.path.join(self.in_dir, speaker))):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                tg_path = os.path.join(self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename))
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename)
                    if ret is None:
                        continue
                    else:
                        info, pitch, energy, n = ret
                    out.append(info)
                else:
                    print("TextGrid not found: {}".format(tg_path))
                    continue

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

                n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            print("Pitch_scaler", pitch_scaler)
            print(pitch_scaler.mean_)
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]
        print("Length of out: ", len(out), out)

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}_src.lab".format(basename))
        translation_path = os.path.join(self.in_dir, speaker, "{}_tgt.lab".format(basename))
        word_alignment_path = os.path.join(self.out_dir, "alignments", "word", "{}-word_alignment-{}.npy".format(speaker, basename))

        tg_path = os.path.join(self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename))

        # Get src time alignments
        textgrid = tgt.io.read_textgrid(tg_path)

        src_phones, duration, start, end = self.get_alignment(textgrid)

        # To flatten phones
        flat_phones = list(chain.from_iterable(src_phones))
        text = "{" + " ".join(flat_phones) + "}"
        if start >= end:
            return None

        # Read raw translation text
        if not os.path.exists(translation_path):
            return None
        with open(translation_path, "r") as f:
            raw_translation = f.readline().strip("\n")

        # Clean translation text # TODO: change punctuation based on language and add more cleaners if not already done.
        # also currently cleaning in ljspeech.py
        raw_translation = raw_translation.translate(str.maketrans('', '', _es_punctuations))

        epi = epitran.Epitran('spa-Latn') # TODO: Change based on language

        tgt_phones = epi.transliterate(raw_translation)
        tgt_phones = tgt_phones.split()

        if len(tgt_phones) != len(raw_translation.split()): # TODO: Figure this out
            # So far only loses singular 'h' like for middle initial because h is silent in spanish
            # Can check and insert myself.
            print(f"{basename} Length of raw translation does not equal length of phonemes! ",
                  raw_translation, tgt_phones, "Continuing...")
            return None

        tgt_phones = [[char for char in word] for word in tgt_phones]

        # Flatten tgt_phones to save to train.txt, val.txt
        flat_phones = list(chain.from_iterable(tgt_phones))

        translation = "{" + " ".join(flat_phones) + "}"

        # Open word alignment file
        word_alignments = np.load(word_alignment_path)

        # Get src tgt phone alignments
        phone_alignments = self.get_phoneme_alignment(word_alignments, src_phones, tgt_phones)

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[int(self.sampling_rate * start) : int(self.sampling_rate * end)].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")
        if phone_alignments is IndexError:
            print(text_path)
            print("raw text", raw_text)
            raise IndexError

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "mel", mel_filename), mel_spectrogram.T)

        phone_alignment_filename = "{}-phone_alignment-{}".format(speaker, basename)
        # np.save(os.path.join(self.out_dir, "alignments", "phone", phone_alignment_filename), phone_alignments)
        # Saving the dictionary
        with open(os.path.join(self.out_dir, "alignments", "phone", phone_alignment_filename + '.pkl'), 'wb') as f:
            pickle.dump(phone_alignments, f)

        return (
            "|".join([basename, speaker, text, raw_text, translation, raw_translation]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, textgrid):
        phones_tier = textgrid.get_tier_by_name("phones")
        words_tier = textgrid.get_tier_by_name("words")
        word_end_times = [w.end_time for w in words_tier._objects]

        sil_phones = ["sil", "sp"]  # Not 'spn'

        all_phones, word_phones, durations = [], [], []
        start_time, end_time = 0, 0
        end_idx, word_idx = 0, 0
        num_phones, num_words = 0, 0

        for t in phones_tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if all_phones == [] and word_phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                if p == "spn" and words_tier.intervals[word_idx].text == "<unk>":
                    # For spoken noise
                    word_phones.append(p)
                    num_phones += 1
                elif p == "spn" and words_tier.intervals[word_idx].text != "<unk>":
                    if not isinstance(all_phones[-1], list):
                        all_phones[-1] = p
                    else:
                        all_phones.append(p)
                    num_phones += 1
                    num_words += 1

                # For ordinary phones
                else:
                    word_phones.append(p)
                    num_phones += 1

                if word_end_times[word_idx] == e:
                    all_phones.append(word_phones)
                    word_phones = []
                    end_time = e
                    end_idx = num_phones
                    num_words += 1

                    if word_idx == len(words_tier.intervals) - 1:  # That was the last word
                        break

                    word_idx += 1

            else:  # For silent phones
                all_phones.append(p)
                num_phones += 1
                num_words += 1

            durations.append(int(np.round(e * self.sampling_rate / self.hop_length) -
                                 np.round(s * self.sampling_rate / self.hop_length)))
            # durations.append(int(np.round(e * 22050 / 256) - np.round(s * 22050 / 256)))

        # Trim tailing silences
        phones = all_phones[:num_words]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def get_phoneme_alignment(self, word_alignments, src_phones, tgt_phones):
        phone_alignments = {}

        # print("src_phones", src_phones)
        # print("tgt_phones", tgt_phones)
        # print("word_alignments", word_alignments)
        # print("cumsums", np.cumsum([len(src_phone) for src_phone in src_phones]))
        src_phone_cumsums = np.cumsum([len(src_phone) for src_phone in src_phones])
        # print("cumsums", np.cumsum(tgt_phones))
        # To flatten phones
        flat_src_phones = list(chain.from_iterable(src_phones))
        flat_tgt_phones = list(chain.from_iterable(tgt_phones))

        flat_phone_alignments = []

        for word_alignment in word_alignments:
            i, j = word_alignment[0], word_alignment[1]

            if i == 0:
                flat_src_phones_idx = 0
            else:
                flat_src_phones_idx = src_phone_cumsums[i-1]

            if j == 0:
                flat_tgt_phones_idx = 0
            else:
                flat_tgt_phones_idx = len(tgt_phones[j-1])

            try:
                src_word_phones = src_phones[i]
                tgt_word_phones = tgt_phones[j]
            except IndexError:
                print("src_phones", src_phones)
                print("tgt_phones", tgt_phones)
                print("word_alignment", word_alignment)
                print("Word alignments", word_alignments)
                print("i, j", i, j)

                return IndexError


            phone_weight = len(src_word_phones) / len(tgt_word_phones)
            phone_alignment, flat_phone_alignment = [], []
            current_src_phone = 0
            phone_accumulations = 0
            tgt_phone = 0
            the_word = []

            while tgt_phone < len(tgt_word_phones):
                if (1-phone_accumulations) > phone_weight:   # Use all of the phone_weight left
                    phone_accumulations += phone_weight
                    if current_src_phone not in phone_alignment:
                        phone_alignment.append(current_src_phone)
                        flat_phone_alignment.append(flat_src_phones_idx)
                    # Reset
                    phone_weight = len(src_word_phones) / len(tgt_word_phones)
                    tgt_phone += 1
                    flat_tgt_phones_idx += 1
                    the_word.append(phone_alignment)
                    flat_phone_alignments.append(flat_phone_alignment)
                    phone_alignment = []
                    flat_phone_alignment = []

                elif phone_weight == (1-phone_accumulations):   # Use all of the phone_weight left
                    phone_alignment.append(current_src_phone)
                    flat_phone_alignment.append(flat_src_phones_idx)
                    phone_accumulations = 0
                    current_src_phone += 1
                    flat_src_phones_idx += 1
                    tgt_phone += 1
                    flat_tgt_phones_idx += 1
                    phone_weight = len(src_word_phones) / len(tgt_word_phones)
                    the_word.append(phone_alignment)
                    flat_phone_alignments.append(flat_phone_alignment)
                    phone_alignment = []
                    flat_phone_alignment = []

                else:   # Phone weight > what's available --> Use part of the phone_weight
                    phone_alignment.append(current_src_phone)
                    flat_phone_alignment.append(flat_src_phones_idx)
                    current_src_phone += 1
                    flat_src_phones_idx += 1
                    phone_weight = phone_weight - (1 - phone_accumulations)
                    phone_accumulations = 0

            if i not in phone_alignments:
                phone_alignments[i] = {}

            if j not in phone_alignments[i]:
                phone_alignments[i][j] = {}

            phone_alignments[i][j] = {k: corresponding_tgt_phones for k, corresponding_tgt_phones in enumerate(the_word)}

        # TODO: Get rid of phone_alignments?

        return flat_phone_alignments

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value

