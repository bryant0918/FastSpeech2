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

def split_into_chunks(lst, n):
    """Splits a list into n approximately equal parts."""
    chunk_size = len(lst) // n
    remainder = len(lst) % n
    chunks = []

    for i in range(n):
        start = i * chunk_size + min(i, remainder)
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])

    return chunks

def load_model(cpu_id, queue, response_queue):
    # Bind process to cpu_id if necessary
    # Load the tokenizer and model
    model_name = 'Helsinki-NLP/opus-mt-es-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    model_str = "Model loaded on CPU {}".format(cpu_id)
    while True:
        data = queue.get()  # Wait for data
        if data == "STOP":
            break
        # Process data with model
        translation = translate(data, tokenizer, model)
        result = "Processed by " + model_str
        print(result)
        response_queue.put(translation)  # Send translation back

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
    # model_name = 'Helsinki-NLP/opus-mt-es-en'
    # tokenizer = MarianTokenizer.from_pretrained(model_name)
    # model = MarianMTModel.from_pretrained(model_name)

    # args = (in_dir, out_dir, preprocessed_dir, sampling_rate, max_wav_value, cleaners, word_aligner)

    def process_line(line, cpu_id, model_queues, response_queues):
        # (in_dir, out_dir, preprocessed_dir, sampling_rate, max_wav_value, cleaners, word_aligner) = args
        audio_path, speaker, text = line.strip().split("|")
        base_name, ext = os.path.splitext(os.path.basename(audio_path))
        out_translation_path = os.path.join(out_dir, speaker, "{}_tgt.lab".format(base_name))
        
        # Skip if file exists (pick up where you left off)
        if os.path.exists(out_translation_path):
            return
        
        if not text:
            return
        text = _clean_text(text, cleaners)

        model_id = cpu_id % 4
        model_queues[model_id].put(text)
        # translation = translate(text, tokenizer, model)

        translation = response_queues[model_id].get()

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
    # with ThreadPoolExecutor(max_workers=21) as executor:
    #     list(tqdm(executor.map(process_line, lines), total=len(lines)))

    model_queues = [Queue() for _ in range(4)]  # Queues for model processes
    response_queues = [Queue() for _ in range(4)]
            
    # Start model processes
    model_processes = [Process(target=load_model, args=(i, model_queues[i], response_queues[i])) for i in range(4)]
    for p in model_processes:
        p.start()

    def process_chunk(chunk, cpu_id, model_queues, response_queues):
        print("In process chunk")
        for line in tqdm(chunk):
            process_line(line, cpu_id, model_queues, response_queues)
        return "Chunk processed"
    
    # Start processing processes
    chunks = split_into_chunks(lines, 4)
    print("Split chunks")
    processing_processes = [Process(target=process_chunk, args=(chunks[i], i, model_queues, response_queues)) for i in range(4, 8)]
    for p in processing_processes:
        p.start()

    # Join processes for cleanup
    for p in processing_processes:
        p.join()

    # Stop model processes
    for q in model_queues:
        q.put("STOP")
    for p in model_processes:
        p.join()

    # print(len(chunks))
    # print([len(chunk) for chunk in chunks])
    # args = (in_dir, out_dir, preprocessed_dir, sampling_rate, max_wav_value, cleaners, word_aligner)
    # with ProcessPoolExecutor(max_workers=21) as executor:
    #     futures = [executor.submit(process_chunk, chunk, args) for chunk in chunks]
    #     for future in futures:
    #         result = future.result()  # This blocks until the future is complete
    #         print(result) 


