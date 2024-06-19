import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text
from text.cleaners import spanish_cleaners

from deep_translator import GoogleTranslator
from simalign import SentenceAligner


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    preprocessed_dir = config["path"]["preprocessed_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    speaker = "LJSpeech"

    os.makedirs((os.path.join(preprocessed_dir, "alignments", "word")), exist_ok=True)

    word_aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="m")

    with open(os.path.join(in_dir, "metadata.csv"), encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("|")
            base_name = parts[0]
            out_translation_path = os.path.join(out_dir, speaker, "{}_tgt.lab".format(base_name))

            text = parts[2]
            text = _clean_text(text, cleaners)

            # TODO: Get cleaners for translation language also (and handle api connection errors)
            translation = GoogleTranslator(source='auto', target='es').translate(text)
            translation = spanish_cleaners(translation)

            wav_path = os.path.join(in_dir, "wavs", "{}.wav".format(base_name))
            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                try:
                    wav, _ = librosa.load(wav_path, sr=sampling_rate)
                except:
                    print("Skipped: ", wav_path)
                    continue
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(os.path.join(out_dir, speaker, "{}_src.lab".format(base_name)), "w") as f1:
                    f1.write(text)

                with open(out_translation_path, "w") as f1:
                    f1.write(translation)

                alignments = word_aligner.get_word_aligns(text.split(), translation.split())
                alignment = alignments["mwmf"]

                with open(os.path.join(preprocessed_dir, "alignments", "word", "{}-word_alignment-{}.npy".format(speaker, base_name)), "wb") as f1:
                    np.save(f1, np.array(alignment))
            
            