import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Process, Queue


from text import _clean_text
from text.cleaners import english_cleaners

from deep_translator import GoogleTranslator, DeeplTranslator, MyMemoryTranslator
from simalign import SentenceAligner
from api_keys import deepl, good_proxies

from transformers import MarianMTModel, MarianTokenizer
import torch


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


def translate(text, tokenizer, model):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Generate translation using the model
    translated = model.generate(**inputs)
    # Decode the translated text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    preprocessed_dir = config["path"]["preprocessed_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    os.makedirs((os.path.join(preprocessed_dir, "alignments", "word")), exist_ok=True)
    os.makedirs((os.path.join(preprocessed_dir, "speaker_emb")), exist_ok=True)

    word_aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="m")
    # translator = GoogleTranslator(source='es', target='en', proxies=good_proxies)
    # deepl_translator = DeeplTranslator(api_key = deepl, source='es', target='en', proxies=good_proxies)
    # mmt2en_translator = MyMemoryTranslator(source='es-ES', target='en-US', proxies=good_proxies)

    # Load the tokenizer and model
    model_name = 'Helsinki-NLP/opus-mt-es-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # args = (in_dir, out_dir, preprocessed_dir, sampling_rate, max_wav_value, cleaners, word_aligner)

    def process_line(line):
        # (in_dir, out_dir, preprocessed_dir, sampling_rate, max_wav_value, cleaners, word_aligner) = args
        parts = line.strip().split("|")
        audio_path = parts[0]
        speaker = parts[1]
        text = ''.join(parts[2:])
        # audio_path, speaker, text = line.strip().split("|")
        base_name, ext = os.path.splitext(os.path.basename(audio_path))
        out_translation_path = os.path.join(out_dir, speaker, "{}_tgt.lab".format(base_name))
        
        # Skip if file exists (pick up where you left off)
        if os.path.exists(out_translation_path):
            return
        
        if not text:
            return
        text = _clean_text(text, cleaners)

        translation = translate(text, tokenizer, model)

        translation = english_cleaners(translation)
        
        if not ext:
            audio_path += ".wav"
        wav_path = os.path.join(in_dir, audio_path)
        # print("Wav path: ", wav_path)
        if os.path.exists(wav_path):
            os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
            try:
                wav, _ = librosa.load(wav_path, sr=sampling_rate)
            except:
                print("Skipped: ", wav_path)
                return

            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                sampling_rate,
                wav.astype(np.int16),
            )
            with open(os.path.join(out_dir, speaker, "{}.lab".format(base_name)), "w") as f1:
                f1.write(text)

            with open(out_translation_path, "w") as f1:
                f1.write(translation)

            alignments = word_aligner.get_word_aligns(text.split(), translation.split())
            alignment = alignments["mwmf"]

            with open(os.path.join(preprocessed_dir, "alignments", "word", "{}-word_alignment-{}.npy".format(speaker, base_name)), "wb") as f1:
                np.save(f1, np.array(alignment))

    print("Preparing alignments...")
    with open(os.path.join(in_dir, "metadata.csv"), encoding="utf-8") as f:
        lines = f.readlines()

    # Use ThreadPoolExecutor to process lines in parallel
    with ThreadPoolExecutor(max_workers=21) as executor:
        list(tqdm(executor.map(process_line, lines), total=len(lines)))


