import os
import subprocess

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text
from text.cleaners import spanish_cleaners

from deep_translator import GoogleTranslator
from simalign import SentenceAligner
from transformers import MarianMTModel, MarianTokenizer
from api_keys import my_proxy


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
    word_aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="m")
    translator  = GoogleTranslator(source='en', target='es', proxies=my_proxy)
    google_translate = True

    # Start vpn for proxy
    script_path = "/home/ditto/Datasets/vpn/cyberghost/start_vpn.sh"
    subprocess.run([script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Load the tokenizer and model
    model_name = 'Helsinki-NLP/opus-mt-es-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Get list of files of failed speech restoration
    dataset_path = os.path.dirname(in_dir)
    sub_dataset_name = os.path.basename(in_dir)
    with open(os.path.join(dataset_path, "libritts_r_failed_speech_restoration_examples", f"{sub_dataset_name}_bad_sample_list.txt"), "r") as f:
        bad_sample_list = [line.strip() for line in f]
    
    for speaker in tqdm(os.listdir(in_dir)):
        for chapter in os.listdir(os.path.join(in_dir, speaker)):
            for file_name in os.listdir(os.path.join(in_dir, speaker, chapter)):
                if file_name[-4:] != ".wav":
                    continue
                base_name = file_name[:-4]

                alignments_path = os.path.join(preprocessed_dir, "alignments", "word", "{}-word_alignment-{}.npy".format(speaker, base_name))
                if os.path.exists(alignments_path):
                        continue

                text_path = os.path.join(in_dir, speaker, chapter, "{}.normalized.txt".format(base_name))
                if not os.path.exists(text_path):
                    text_path = os.path.join(in_dir, speaker, chapter, "{}.original.txt".format(base_name))
                    if not os.path.exists(text_path):
                        text_path = os.path.join(os.path.dirname(dataset_path), "LibriTTS", sub_dataset_name, speaker, chapter, "{}.normalized.txt".format(base_name))

                out_translation_path = os.path.join(out_dir, speaker, "{}_tgt.lab".format(base_name))

                # Get good wav path from right place
                check_path = os.path.join('./', sub_dataset_name, speaker, chapter, file_name)
                try:
                    bad_sample_list.remove(check_path)
                    wav_path = os.path.join(os.path.dirname(dataset_path), "LibriTTS", sub_dataset_name, speaker, chapter, "{}.wav".format(base_name))
                except ValueError:
                    wav_path = os.path.join(
                        in_dir, speaker, chapter, "{}.wav".format(base_name)
                    )
                
                with open(text_path) as f:
                    text = f.readline().strip("\n")
                text = _clean_text(text, cleaners)
                
                # Use google translate then restart container.
                if google_translate:
                    try:
                        translation = translator.translate(text)
                    except:
                        subprocess.run([script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        try:
                            translation = translator.translate(text)
                        except:
                            google_translate = False
                            translation = translate(text, tokenizer, model)
                else:
                    translation = translate(text, tokenizer, model)
                
                translation = spanish_cleaners(translation)

                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sr=sampling_rate)
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

                with open(alignments_path, "wb") as f1:
                    np.save(f1, np.array(alignment))