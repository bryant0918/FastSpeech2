import torch


def test_translation():
    import os

    translation_path = os.path.join(self.in_dir, speaker, "{}_tgt.lab".format(basename))

    if not os.path.exists(translation_path):
        return None
    with open(translation_path, "r") as f:
        raw_translation = f.readline().strip("\n")

    return raw_translation

def test_marian_translte():
    from transformers import MarianMTModel, MarianTokenizer
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'    

    # Load the tokenizer and model
    model_name = 'Helsinki-NLP/opus-mt-es-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)

    def translate(text, tokenizer, model):
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        # Generate translation using the model
        translated = model.generate(**inputs)
        # Decode the translated text
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text

    # Example usage
    spanish_text = "Hola, ¿cómo estás?"
    start = time.time()
    translated_text = translate(spanish_text, tokenizer, model)
    print("Time taken: ", time.time() - start)
    print(f"Translated text: {translated_text}")

def test_google_translate():
    from deep_translator import GoogleTranslator

    text = "printing in the only sense with which we are at present concerned differs from most if not from all the arts and crafts represented in the exhibition"
    text = "and the aggregate amount of debts sued for was eightyone thousand seven hundred ninetyone pounds"

    # TODO: Get cleaners for translation language also (and handle api connection errors)
    translation = GoogleTranslator(source='auto', target='es').translate(text)
    # translation = _clean_text(translation, cleaners)

    print(translation)
    from text.cleaners import remove_punctuation, english_cleaners
    translation = remove_punctuation(translation)
    print(translation)

    text = "En Inglaterra, por esta época, Caslon hizo un intento en particular, quien comenzó su negocio en Londres como fundidor tipográfico en mil setecientos veinte y tenía cuarenta y cinco manzanas."
    english = GoogleTranslator(source='es', target='en').translate(translation)
    print(english)
    english = english_cleaners(english)
    print(english)


    return translation

def test_translators():
    from text.cleaners import remove_punctuation, english_cleaners
    import time, os
    from api_keys import chatgpt, deepl, qcri
    from deep_translator import (GoogleTranslator,
                                ChatGptTranslator,
                                MicrosoftTranslator,
                                PonsTranslator,
                                LingueeTranslator,
                                MyMemoryTranslator,
                                YandexTranslator,
                                PapagoTranslator,
                                DeeplTranslator,
                                QcriTranslator,
                                LibreTranslator,
                                single_detection,
                                batch_detection)
    

    """
    What we've learned:
    - translate_batch is NOT faster than translate sequentially
    - GoogleTranslator is baseline
    - ChatGptTranslator requires api key and funds
    - MicrosoftTranslator requires api key, can get free trial but requires credit card
    - PonsTranslator only allows up to 50 characters
    - LingueeTranslator only allows up to 50 characters
    - MyMemoryTranslator is slower
    - YandexTranslator requires api key, supposedly can get free trial but couldn't find, just ask for card
    - PapagoTranslator requires api key and funds
    - DeeplTranslator requires api key, has free tier but is slower
    - QcriTranslator requires api key, free, slow, en-es server not always up
    - LibreTranslator requires api key and funds

    Google translate remains to be the best option.
    Use DeepL after maximum requests to google api.
    use MyMemory as last resort.
    """
    
    proxies_example = {
    'http': '82.136.72.102:80',
    'http':'200.108.190.38:999',
    'http':'113.160.155.121:19132',
    'http':'116.63.129.202:6000',
    'http':'191.37.208.1:8081',
    'http':'103.158.27.94:8090',
    'http':'113.161.187.190:8080',
    }
    proxies_example = {
    'socks4':'103.88.169.6:3629',
    'socks4':'138.201.21.228:48164',
    }
    google2es = GoogleTranslator(source='en', target='es', proxies=proxies_example)
    google2en = GoogleTranslator(source='es', target='en')

    chatgpt2es = ChatGptTranslator(api_key=chatgpt, source='en', target='es')

    # micro2es = MicrosoftTranslator(source='en', target='es')

    pons2es = PonsTranslator(source='english', target='spanish')

    linguee2es = LingueeTranslator(source='english', target='spanish')

    mmt2es = MyMemoryTranslator(source='en-US', target='es-ES')
    mmt2en = MyMemoryTranslator(source='es-ES', target='en-US')

    # yandex2es = YandexTranslator(source='en', target='es')

    # papago2es = PapagoTranslator(source='en', target='es')

    deepl2es = DeeplTranslator(api_key = deepl, source='en', target='es')
    deepl2en = DeeplTranslator(api_key = deepl, source='es', target='en')

    qcri2en = QcriTranslator(api_key = qcri, source='Spanish', target='English')

    # libre2es = LibreTranslator(source='en', target='es')

    translators2es = [google2es]
    translators2en = [mmt2en, deepl2en]

    en_text = "printing in the only sense with which we are at present concerned differs from most if not from all the arts and crafts represented in the exhibition"
    en_text2 = "and the aggregate amount of debts sued for was eightyone thousand seven hundred ninetyone pounds"
    es_text = "En Inglaterra, por esta época, Caslon hizo un intento en particular, quien comenzó su negocio en Londres como fundidor tipográfico en mil setecientos veinte y tenía cuarenta y cinco manzanas."

    in_dir = "/home/ditto/Datasets/Speech/es"
    with open(os.path.join(in_dir, "metadata.csv"), encoding="utf-8") as f:
        lines = f.readlines()

    # audio_path, speaker, text = line.strip().split("|")
    batch_text = [line.strip().split("|")[2] for line in lines][:4]

    for translator in translators2es:
        print()
        print("Testing: ", translator.__class__.__name__)
        start = time.time()
        # translation = translator.translate(en_text)
        translation = translator.translate(source="Spanish", target='English', text=en_text, domain='general')
        time1 = time.time() - start
        print("Time taken: ", time1)
        print(translation)
        translation = remove_punctuation(translation)
        print(translation)

        start = time.time()
        # translation = translator.translate(en_text)
        translation = translator.translate(source='en', target='es', text=en_text, domain='general')
        time2 = time.time() - start
        print("Time taken: ", time2)
        print(translation)
        translation = remove_punctuation(translation)
        print(translation)

        print("Total time: ", time1 + time2)

        # batch_text = [en_text, en_text2]
        start = time.time()
        # translations = translator.translate_batch(batch_text)
        translations = translator.translate_batch(batch_text, domain='general-fast')
        time3 = time.time() - start
        print("Batch Time taken: ", time3)
        print(translations)

    for translator in translators2en:
        print()
        print("Testing: ", translator.__class__.__name__)
        start = time.time()
        # translation = translator.translate(en_text)
        translation = translator.translate(source='es', target='en', domain='general', text=es_text)
        time1 = time.time() - start
        print("Time taken: ", time1)
        print(translation, type(translation))
        translation = english_cleaners(translation)
        print(translation)

        start = time.time()
        # translation = translator.translate(en_text)
        translation = translator.translate(source='es', target='en', domain='general', text=es_text)
        time2 = time.time() - start
        print("Time taken: ", time2)
        print(translation)
        translation = remove_punctuation(translation)
        print(translation)

        print("Total time: ", time1 + time2)

        # batch_text = [en_text, en_text2]
        start = time.time()
        # translations = translator.translate_batch(batch_text)
        translations = translator.translate_batch(batch_text, domain='general')
        time3 = time.time() - start
        print("Batch Time taken: ", time3)
        print(translations)

    import subprocess
    import json
    # command = ["echo", "Hello, World!"]
    # process = subprocess.run(command, capture_output=True, text=True)

    # Define the curl command
    curl_command = """
    curl -X POST http://localhost:11434/api/generate -d '{{
    "model": "mistral",
    "prompt":"Translate this sentence into spanish without adding any extra context in your response: {}"
    }}'
    """.format(en_text)

    # Execute the curl command
    start = time.time()
    result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)
    full_response = ""

    for line in result.stdout.strip().split('\n'):
        data = json.loads(line)
        full_response += data["response"]
        if data.get("done", True):  # Using .get() to avoid KeyError if 'done' is missing
            break
    
    print("Time taken: ", time.time() - start)

    print(full_response)

def test_inv_melspec():
    import librosa
    from audio.stft import TacotronSTFT
    from audio.audio_processing import griffin_lim
    import time
    import yaml
    from audio.tools import get_mel_from_wav, inv_mel_spec
    import torch
    import numpy as np

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    preprocess_config = "config/LJSpeech/preprocess.yaml"
    config = yaml.load(open(preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = "config/LJSpeech/model.yaml"
    model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)

    audio_path = "raw_data/LJSpeech/LJSpeech/LJ001-0004.wav"
    mel_path = "preprocessed_data/LJSpeech/mel/LJSpeech-mel-LJ001-0004.npy"

    STFT = TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )
    
    wav, _ = librosa.load(audio_path)
    wav = torch.from_numpy(wav)

    mel,energy = get_mel_from_wav(wav, STFT)
    mel = torch.from_numpy(mel).to(device)
    print("mel shape: ", mel.shape)

    melspec = torch.from_numpy(np.load(mel_path)).permute(1,0).to(device)
    print("melspec shape: ", melspec.shape)

    start = time.time()
    audio1 = inv_mel_spec(mel, "output/test1.wav", STFT.to(device))
    print("Time taken: ", time.time() - start)
    # audio2 = inv_mel_spec(melspec, "output/test2.wav", STFT)

    from utils.model import get_vocoder, vocoder_infer
    from scipy.io.wavfile import write

    # Load vocoder
    vocoder = get_vocoder(model_config, device)
    print("Vocoder Loaded")

    start = time.time()
    wav_reconstruction = vocoder_infer(
        mel.unsqueeze(0),
        vocoder,
        model_config,
        config,
    )[0]
    
    print("Time taken: ", time.time() - start)
    write("output/test1.wav", config["preprocessing"]["audio"]["sampling_rate"], wav_reconstruction)

def test_whisper_STT():
    import whisper
    import librosa
    import time
    import os
    import numpy as np

    audio_path = "demo/LJSpeech/LJ001-0012_ground-truth.wav"
    model = whisper.load_model("small").to('cuda')

    files = os.listdir("demo/LJSpeech")
    files.sort()
    times = []
    for file in files:
        if file.endswith(".wav"):
            audio_path = os.path.join("demo/LJSpeech", file)
            
            wav, _ = librosa.load(audio_path)
            wav = torch.from_numpy(wav).to('cuda')
            start = time.time()
            text = model.transcribe(wav)
            times.append(time.time() - start)
            # print("\n", file, text['text'])

    print("Average time: ", np.mean(times))
    print("Total Time: ", np.sum(times))    
    # model = whisper.load_model("base").to('cuda')
    # wav = torch.randn(120000).to(torch.float32).to("cuda")
    # start = time.time()
    # text = model.transcribe(wav)

def test_whisperX():
    from whisperX import whisperx
    import os
    import numpy as np
    from utils.tools import pad_1D
    import torch
    import time

    device = "cuda" 
    batch_size = 8 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("small", device, compute_type=compute_type)

    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)
    files = os.listdir("demo/LJSpeech")
    files.sort()
    
    times = []
    for file in files:
        if file.endswith(".wav"):
            audio_path = os.path.join("demo/LJSpeech", file)
            audio = whisperx.load_audio(audio_path)
            print("\naudio shape: ", np.shape(audio))
            print("wav dtype:", audio.dtype)

            audio = torch.from_numpy(audio).to(device)
            print("Audio shape: ", audio.shape)
            print("audio dtype:", audio.dtype)

            start = time.time()
            result = model.transcribe(audio, batch_size=batch_size, language='en')
            times.append(time.time() - start)
            # print(result["segments"][0]['text']) # before alignment
            
    print("Average time: ", np.mean(times))
    print("Total Time: ", np.sum(times))
    
def test_insanely_fast_whisper():
    import torch
    from transformers import pipeline
    from transformers.utils import is_flash_attn_2_available
    import time

    print("is_flash_attn_2_available: ", is_flash_attn_2_available())

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
        torch_dtype=torch.float16,
        device="cuda:0", # or mps for Mac devices
        model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
    )

    start = time.time()
    outputs = pipe(
        "demo/LJSpeech/LJ001-0012_ground-truth.wav",
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=False,
    )
    print("Time taken: ", time.time() - start)
    print(outputs)

    # Use Base environment to test since needs torch >= 2.1.1
    # But slower than whisperx anyway (and regular whisper)

def test_g2p():
    import time
    start = time.time()
    from g2p_en import G2p
    import bisect
    from text.cmudict import CMUDict
    print("Time taken", time.time() - start)

    lexicon_path = "lexicon/librispeech-lexicon.txt"

    g2p = G2p()
    text = "pandas"
    start = time.time()
    phones = g2p(text)
    print("Time taken: ", time.time() - start)
    print(phones)

    start = time.time()
    with open(lexicon_path, "r") as f:
        lexicon = f.readlines()

    new_line = f"{text.upper()}\t{' '.join(phones)}\n"

    bisect.insort(lexicon, new_line)

    with open(lexicon_path, "w") as f:
        f.writelines(lexicon)
    print("Time taken", time.time() - start)

    text = "pandas"
    cmu = CMUDict(lexicon_path)
    phones = cmu.lookup(text)[0].split(' ')
    print("phones: ", phones)

def new_train_val_file():
    filepaths = ["preprocessed_data/LJSpeech/train.txt",
                    "preprocessed_data/LJSpeech/val.txt"]
    for filepath in filepaths:
        with open(filepath, "r") as f:
            lines = f.readlines()

        # Open the file again (or a new file) to write the modified contents
        with open(filepath, 'w') as file:
            for line in lines:
                # Prepend "en|es|" to each line
                file.write(f'en|es|{line}')

def custom_round(x):
    # Round x in (0, .5] up to 1, keep 0 as is
    mask = (x <= 0.5) & (x > 0)
    # Add pos/neg eps randomely so half the .5's round up and half down
    eps = (torch.rand_like(x) - .5)/100
    x[mask] = torch.ceil(x[mask])
    return torch.round(x + eps)

def test_speaker_emb():
    import numpy as np
    import pickle

    speaker_emb_path_mean = "preprocessed_data/Spanish/speaker_emb/F001/F001.pkl"       
    speaker_emb_path_indiv = "preprocessed_data/Spanish/speaker_emb/F001/TEDX_F_001_SPA_0001.pkl"
    
    with open(speaker_emb_path_mean, 'rb') as f:
        emb_dict = pickle.load(f)
    mean_embedding = torch.from_numpy(emb_dict["mean"])

    with open(speaker_emb_path_indiv, 'rb') as f:
        emb_dict = pickle.load(f)
    indiv_embedding = torch.from_numpy(emb_dict["default"])

    embedding = np.mean([mean_embedding, indiv_embedding], axis=0)
    print("Embedding shape:", np.shape(embedding))

def test_vocoder():
    import time
    from utils.model import get_vocoder, vocoder_infer
    from scipy.io.wavfile import write
    import json
    import numpy as np
    import hifigan
    import bigvgan
    import yaml

    device = torch.device("cuda")

    model_config = "config/LJSpeech/model.yaml"
    model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
    preprocess_config = "config/LJSpeech/preprocess.yaml"
    preprocess_config = yaml.load(open(preprocess_config, "r"), Loader=yaml.FullLoader)

    with open("hifigan/config.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    hifi = hifigan.Generator(config)
    ckpt = torch.load("hifigan/generator_universal.pth.tar", map_location=device)
    hifi.load_state_dict(ckpt["generator"])
    hifi.eval()
    hifi.remove_weight_norm()
    hifi.to(device)

    with open("bigvgan/config.json", "r") as f:
        config = json.load(f)
    config = bigvgan.AttrDict(config)
    bigv = bigvgan.Generator(config)
    ckpt = torch.load("bigvgan/g_05000000", map_location=device)
    bigv.load_state_dict(ckpt['generator'])
    bigv.eval()
    bigv.remove_weight_norm()
    bigv.to(device)

    hifi_times, bigv_times = [], []

    for i in range(10):
        mel_target_path = f'output/result/LJSpeech/pretrain/mel_target{i}.npy'
        mel_prediction_path = f'output/result/LJSpeech/pretrain/mel_prediction{i}.npy'
        hifigan_target_path = f'output/result/LJSpeech/pretrain/hifigan_target{i}.wav'
        hifigan_prediction_path = f'output/result/LJSpeech/pretrain/hifigan_prediction{i}.wav'
        bigvgan_target_path = f'output/result/LJSpeech/pretrain/bigvgan_target{i}.wav'
        bigvgan_prediction_path = f'output/result/LJSpeech/pretrain/bigvgan_prediction{i}.wav'

        mel_target = torch.from_numpy(np.load(mel_target_path)).to(device)
        mel_prediction = torch.from_numpy(np.load(mel_prediction_path)).to(device)

        start = time.time()
        hifi_target = vocoder_infer(
            mel_target.unsqueeze(0),
            hifi,
            model_config,
            preprocess_config,
        )[0]
        hifi_times.append(time.time() - start)
        write(hifigan_target_path, 22050, hifi_target)

        start = time.time()
        hifi_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            hifi,
            model_config,
            preprocess_config,
        )[0]
        hifi_times.append(time.time() - start)
        write(hifigan_prediction_path, 22050, hifi_prediction)

        start = time.time()
        bigv_target = vocoder_infer(
            mel_target.unsqueeze(0),
            bigv,
            model_config,
            preprocess_config,
        )[0]
        bigv_times.append(time.time() - start)
        write(bigvgan_target_path, 22050, bigv_target)

        start = time.time()
        bigv_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            bigv,
            model_config,
            preprocess_config,
        )[0]
        bigv_times.append(time.time() - start)
        write(bigvgan_prediction_path, 22050, bigv_prediction)
    
    print("Done")

def test_miipher():
    from miipher.dataset.preprocess_for_infer import PreprocessForInfer
    from miipher.lightning_module import MiipherLightningModule
    from lightning_vocoders.models.hifigan.xvector_lightning_module import HiFiGANXvectorLightningModule
    import torch
    import torchaudio
    import hydra

    miipher_path = "miipher/miipher_v2.ckpt"
    miipher = MiipherLightningModule.load_from_checkpoint(miipher_path,map_location='cpu')
    vocoder = HiFiGANXvectorLightningModule.load_from_checkpoint("miipher/vocoder_finetuned_v2.ckpt",map_location='cpu')
    xvector_model = hydra.utils.instantiate(vocoder.cfg.data.xvector.model)
    xvector_model = xvector_model.to('cpu')
    preprocessor = PreprocessForInfer(miipher.cfg)
    preprocessor.cfg.preprocess.text2phone_model.is_cuda=False
    
    @torch.inference_mode()
    def main(wav_path, output_path, transcript=None, lang_code=None, phones=None):
        wav,sr =torchaudio.load(wav_path)
        wav = wav[0].unsqueeze(0)
        batch = preprocessor.process(
            'test',
            (torch.tensor(wav),sr),
            word_segmented_text=transcript,
            lang_code=lang_code,
            phoneme_text=phones
        )

        

        miipher.feature_extractor(batch)
        (
            phone_feature,
            speaker_feature,
            degraded_ssl_feature,
            _,
        ) = miipher.feature_extractor(batch)
        

        cleaned_ssl_feature, _ = miipher(phone_feature,speaker_feature,degraded_ssl_feature)
        vocoder_xvector = xvector_model.encode_batch(batch['degraded_wav_16k'].view(1,-1).cpu()).squeeze(1)
        cleaned_wav = vocoder.generator_forward({"input_feature": cleaned_ssl_feature, "xvector": vocoder_xvector})[0].T
        # with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fp:
        #     torchaudio.save(fp,cleaned_wav.view(1,-1), sample_rate=22050,format='wav')
        #     return fp.name
        
        torchaudio.save(output_path,cleaned_wav.view(1,-1), sample_rate=22050,format='wav')
    
    audio_file = "/home/ditto/Ditto/FastSpeech2/raw_data/Spanish/M043/TEDX_M_043_SPA_0070.wav"
    output_path = "output/result/test_miipher.wav"
    phones = "i e s o s e p a ɾ a m i s e s e p w e ð e r e ð u s i ɾ e n c e e n u n d̪ e s p e r t̪ a ɾ d̪ e l a k o n θ j e n θ j a u m a n a ɡ ɾ a θ j a s"
    
    main(audio_file, output_path, phones=phones)
    
    # # Load the model and feature extractor
    # model_name = "miipher"
    # model = AutoModelForAudioFrameClassification.from_pretrained(model_name)
    # feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    # audio_file = "/home/ditto/Ditto/FastSpeech2/raw_data/Spanish/F001/TEDX_F_001_SPA_0001.wav"
    # output_path = "output/result/test_miipher.wav"

    # waveform, sample_rate = torchaudio.load(audio_file)

    # inputs = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt")

    # inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # with torch.no_grad():
    #     outputs = model(**inputs)
    # enhanced_waveform = outputs['logits'].cpu().numpy()
    
    # torchaudio.save(output_path, enhanced_waveform, sample_rate)

def test_realign_ped():
    durations = torch.tensor([ 0,  4,  5,  4,  4,  5,  2, 16, 14,  7, 10,  7,  5,  3,  6,  6,  6,  5,
                              6, 15, 10, 15,  5, 14, 20, 10,  3,  6,  3,  4, 10, 10,  7,  3,  5,  4,
                              9,  5,  4,  9,  7,  5,  7,  2,  9,  4,  4, 12,  6, 12,  5,  4,  4,  6,
                              4,  3,  7,  3,  3, 10,  8,  8,  4,  7,  9,  7,  2,  9,  6,  3,  3, 10,
                              5, 11,  3,  7,  9, 11,  8,  3,  3,  7,  3,  9, 10,  8,  8,  3, 22,  0,
                              0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], device='cuda:0')
    
    dur_list = [ 0,  4,  5,  4,  4,  5,  2, 16, 14,  7, 10,  7,  5,  3,  6,  6,  6,  5,
                              6, 15, 10, 15,  5, 14, 20, 10,  3,  6,  3,  4, 10, 10,  7,  3,  5,  4,
                              9,  5,  4,  9,  7,  5,  7,  2,  9,  4,  4, 12,  6, 12,  5,  4,  4,  6,
                              4,  3,  7,  3,  3, 10,  8,  8,  4,  7,  9,  7,  2,  9,  6,  3,  3, 10,
                              5, 11,  3,  7,  9, 11,  8,  3,  3,  7,  3,  9, 10,  8,  8,  3, 22,  0,
                              0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    print("22 at: ", dur_list.index(22))

    alignments = torch.tensor([[ 0,  0,  0,  0,  0],
                               [ 1,  0,  0,  0,  0],
                               [ 2,  0,  0,  0,  0],
                               [ 3,  0,  0,  0,  0],
                               [ 4,  0,  0,  0,  0],
                               [ 5,  0,  0,  0,  0],
                               [ 5,  6,  0,  0,  0],
                               [ 6,  0,  0,  0,  0],
                               [ 7,  0,  0,  0,  0],
                               [ 7,  0,  0,  0,  0],
                               [ 7,  0,  0,  0,  0],
                               [ 7,  8,  0,  0,  0],
                               [ 8,  0,  0,  0,  0],
                               [ 8,  0,  0,  0,  0],
                               [ 8,  9,  0,  0,  0],
                               [ 9,  0,  0,  0,  0],
                               [ 9,  0,  0,  0,  0],
                               [ 9, 10,  0,  0,  0],
                               [10,  0,  0,  0,  0],
                               [10,  0,  0,  0,  0],
                               [17,  0,  0,  0,  0],
                               [18,  0,  0,  0,  0],
                               [11,  0,  0,  0,  0],
                               [11, 12,  0,  0,  0],
                               [12, 13,  0,  0,  0],
                               [13,  0,  0,  0,  0],
                               [14,  0,  0,  0,  0],
                               [14,  0,  0,  0,  0],
                               [14, 15,  0,  0,  0],
                               [15,  0,  0,  0,  0],
                               [15,  0,  0,  0,  0],
                               [15, 16,  0,  0,  0],
                               [16,  0,  0,  0,  0],
                               [16,  0,  0,  0,  0],
                               [19,  0,  0,  0,  0],
                               [20,  0,  0,  0,  0],
                               [ 0,  0,  0,  0,  0],
                               [ 0,  0,  0,  0,  0],
                               [21,  0,  0,  0,  0],
                               [21, 22,  0,  0,  0],
                               [22,  0,  0,  0,  0],
                               [22, 23,  0,  0,  0],
                               [23,  0,  0,  0,  0],
                               [24,  0,  0,  0,  0],
                               [24, 25,  0,  0,  0],
                               [25, 26,  0,  0,  0],
                               [26,  0,  0,  0,  0],
                               [27,  0,  0,  0,  0],
                               [28,  0,  0,  0,  0],
                               [29, 44, 45,  0,  0],
                               [29, 30, 45, 46,  0],
                               [30, 31, 46, 47,  0],
                               [31, 32, 47, 48, 49],
                               [32, 33, 49, 50,  0],
                               [33, 34, 50, 51,  0],
                               [34, 51, 52, 53,  0],
                               [38, 39, 40,  0,  0],
                               [41, 42, 43,  0,  0],
                               [ 0,  0,  0,  0,  0],
                               [ 0,  0,  0,  0,  0],
                               [35,  0,  0,  0,  0],
                               [35, 36,  0,  0,  0],
                               [36, 37,  0,  0,  0],
                               [37,  0,  0,  0,  0],
                               [44, 45,  0,  0,  0],
                               [45, 46,  0,  0,  0],
                               [47, 48,  0,  0,  0],
                               [48, 49,  0,  0,  0],
                               [50, 51,  0,  0,  0],
                               [51, 52,  0,  0,  0],
                               [53,  0,  0,  0,  0],
                               [54,  0,  0,  0,  0],
                               [55,  0,  0,  0,  0],
                               [56,  0,  0,  0,  0],
                               [56, 57,  0,  0,  0],
                               [57,  0,  0,  0,  0],
                               [58, 83,  0,  0,  0],
                               [59, 83, 84,  0,  0],
                               [60, 84, 85,  0,  0],
                               [61, 85,  0,  0,  0],
                               [62, 85, 86,  0,  0],
                               [63, 86, 87,  0,  0],
                               [64, 87,  0,  0,  0],
                               [65,  0,  0,  0,  0],
                               [66,  0,  0,  0,  0],
                               [67, 73,  0,  0,  0],
                               [67, 68, 73, 74,  0],
                               [68, 69, 74, 75,  0],
                               [69, 70, 75,  0,  0],
                               [70, 71, 75, 76,  0],
                               [71, 72, 76, 77,  0],
                               [72, 77, 78,  0,  0],
                               [67,  0,  0,  0,  0],
                               [67, 68,  0,  0,  0],
                               [68,  0,  0,  0,  0],
                               [68, 69,  0,  0,  0],
                               [69, 70,  0,  0,  0],
                               [70,  0,  0,  0,  0],
                               [70, 71,  0,  0,  0],
                               [71, 72,  0,  0,  0],
                               [72,  0,  0,  0,  0],
                               [78,  0,  0,  0,  0],
                               [79,  0,  0,  0,  0],
                               [80, 81,  0,  0,  0],
                               [81, 82,  0,  0,  0],
                               [83,  0,  0,  0,  0],
                               [83, 84,  0,  0,  0],
                               [84, 85,  0,  0,  0],
                               [85,  0,  0,  0,  0],
                               [85, 86,  0,  0,  0],
                               [86, 87,  0,  0,  0],
                               [87,  0,  0,  0,  0]], device='cuda:0', dtype=torch.int32)

    durations = durations.unsqueeze(0)
    alignments = alignments.unsqueeze(0)

    # def realign_p_e_d(alignments, p_e_d):
    #     new_ped = torch.zeros(p_e_d.size(0), len(alignments[0])+1, device=p_e_d.device)
    #     for b, alignment in enumerate(alignments):
    #         for j, src_indices in enumerate(alignment):
    #             # Filter out zeros (when alignment is padded with zero, here the src_idx is zero and durations[0] is always 0) 
    #             non_zero_p_e_d = [p_e_d[b][i] for i in src_indices if p_e_d[b][i] != 0]
    #             if non_zero_p_e_d:  # Check if the list is not empty
    #                 average_p_e_d = sum(non_zero_p_e_d) / len(non_zero_p_e_d)
    #             else:
    #                 average_p_e_d = 0  # Avoid division by zero if all durations are zero
    #             print("average_p_e_d: ", average_p_e_d)
    #             new_ped[b][j] = average_p_e_d
    #     return new_ped
    
    def realign_p_e_d(alignments, p_e_d):
        # Initialize new_ped with zeros_like to ensure it's on the same device and has the same dtype
        new_ped = torch.zeros(p_e_d.size(0), len(alignments[0])+1, device=p_e_d.device)
        for b, alignment in enumerate(alignments):
            for j, src_indices in enumerate(alignment):
                # Use torch operations to maintain the computation graph
                non_zero_p_e_d = p_e_d[b, src_indices][p_e_d[b, src_indices] != 0]
                if non_zero_p_e_d.nelement() > 0:  # Check if there are non-zero elements
                    average_p_e_d = non_zero_p_e_d.float().mean()
                else:
                    average_p_e_d = torch.tensor(0, device=p_e_d.device, dtype=p_e_d.dtype)
                # Use indexing to assign values to maintain the computation graph
                new_ped[b, j] = average_p_e_d
        return new_ped

    def realign_d(alignments, durations):
        new_d = torch.zeros(durations.size(0), len(alignments[0])+1, device=durations.device)
        for b, alignment in enumerate(alignments):
            for j, src_indices in enumerate(alignment):

                # Filter out zeros (when alignment is padded with zero, here the src_idx is zero and durations[0] is always 0) 
                non_zero_durations = [durations[b][i] for i in src_indices if durations[b][i] != 0]
                if non_zero_durations:  # Check if the list is not empty
                    average_duration = sum(non_zero_durations) / len(non_zero_durations)
                else:
                    average_duration = 0  # Avoid division by zero if all durations are zero
                new_d[b][j] = average_duration
        return new_d
    
    def custom_round(x):
        # Round x in (0, .5] up to 1, keep 0 as is
        mask = (x <= 0.5) & (x > 0)
        x[mask] = torch.ceil(x[mask])
        # Add pos/neg eps randomely so half the .5's round up and half down
        eps = (torch.rand_like(x) - .5)/100
        return torch.round(x + eps).int()
    
    realigned_d = realign_p_e_d(alignments, durations)
    # realigned_d = realign_d(alignments, durations)

    print("duration sum: ", torch.sum(durations))
    print("realigned_d: ", torch.sum(realigned_d), realigned_d)
    print("rounded realigened_d: ", torch.sum(custom_round(realigned_d)), custom_round(realigned_d))
    
def test_npc():
    from model.modules import NPC
    import yaml
    
    model_config = "config/Tiny/model.yaml"
    model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)

    # Test Alexander-H-Liu's NPC Module
    x = torch.randn(4, 87, 256)

    print("x shape", x.shape)
    
    npc = NPC(model_config)

    print(sum(param.numel() for param in npc.parameters()))

    pred, _ = npc(x)

    print(pred.shape)
    loss = torch.nn.functional.l1_loss(pred, x)
    print(loss)

def test_pros_loss():
    from model.loss import ProsLoss

    prosody_loss = ProsLoss()

    log_pi = -torch.abs(torch.randn(2,83,8)).to(torch.float64) 
    mu = torch.randn(2,83,8,256).to(torch.float64) * .0001
    sigma = torch.abs(torch.randn(2,83,8,256)).to(torch.float64) * .0001
    
    x = (log_pi, mu, sigma)
    y = torch.randn(2, 83, 256).to(torch.float64) * .0001

    print(torch.norm(y, p=2))

    print("log_pi", torch.min(log_pi).item(), torch.max(log_pi).item())
    print("mu", torch.min(mu).item(), torch.max(mu).item())
    print("sigma", torch.min(sigma).item(), torch.max(sigma).item())
    print("y", torch.min(y).item(), torch.max(y).item())

    mask = torch.ones(2, 83).to(torch.bool)
    mask[:, 80:] = 0

    loss = prosody_loss(x, y, mask)

    print("\nloss", loss.item())

def anaylze_p_e_d():
    import numpy as np

    og_pitch = "preprocessed_data/LJSpeech/pitch/LJSpeech-pitch-LJ019-0137.npy"
    aug_pitch = "preprocessed_data/LJSpeech/pitch/augmented-pitch-LJ031-0020_a.npy"
    aug_pitch = "preprocessed_data/LJSpeech/pitch/augmented-pitch-LJ032-0001_a.npy"

    og_energy = "preprocessed_data/LJSpeech/energy/LJSpeech-energy-LJ032-0001.npy"
    aug_energy = "preprocessed_data/LJSpeech/energy/augmented-energy-LJ032-0001_a.npy"

    og_pitch = np.load(og_pitch)
    aug_pitch = np.load(aug_pitch)

    og_energy = np.load(og_energy)
    aug_energy = np.load(aug_energy)

    print(np.min(og_pitch), np.max(og_pitch), np.shape(og_pitch))
    print(np.min(aug_pitch), np.max(aug_pitch), np.shape(aug_pitch))

    print(np.min(og_energy), np.max(og_energy), np.shape(og_energy))
    print(np.min(aug_energy), np.max(aug_energy), np.shape(aug_energy))

    # 138.92622993901406 296.77412518844494 (15,)
    # -1.2443200884455081 5.16465252707064 (15,)

def test_plot():
    from utils.tools import plot_mel, expand
    from matplotlib import pyplot as plt
    import numpy as np
    import os
    import json

    basename = "LJ040-0143"

    og_pitch = f"preprocessed_data/LJSpeech/pitch/LJSpeech-pitch-{basename}.npy"
    aug_pitch = f"preprocessed_data/LJSpeech/pitch/augmented-pitch-{basename}_a.npy"

    og_energy = f"preprocessed_data/LJSpeech/energy/LJSpeech-energy-{basename}.npy"
    aug_energy = f"preprocessed_data/LJSpeech/energy/augmented-energy-{basename}_a.npy"

    og_duration = f"preprocessed_data/LJSpeech/duration/LJSpeech-duration-{basename}.npy"
    aug_duration = f"preprocessed_data/LJSpeech/duration/augmented-duration-{basename}_a.npy"

    og_pitch = np.load(og_pitch)
    aug_pitch = np.load(aug_pitch)

    og_energy = np.load(og_energy)
    aug_energy = np.load(aug_energy)

    og_duration = np.load(og_duration)
    aug_duration = np.load(aug_duration)

    og_pitch = expand(og_pitch, og_duration)
    aug_pitch = expand(aug_pitch, aug_duration)

    og_energy = expand(og_energy, og_duration)
    aug_energy = expand(aug_energy, aug_duration)

    og_mel = f"preprocessed_data/LJSpeech/mel/LJSpeech-mel-{basename}.npy"
    aug_mel = f"preprocessed_data/LJSpeech/mel/augmented-mel-{basename}_a.npy"
    
    og_mel = np.load(og_mel).T
    aug_mel = np.load(aug_mel).T

    with open(
        "preprocessed_data/LJSpeech/stats.json"
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]


    fig = plot_mel(
        [
            (og_mel, og_pitch, og_energy),
            (aug_mel, aug_pitch, aug_energy),
        ],
        stats,
        ["Original Spectrogram", "Augmented Spectrogram"],
    )

    fig.savefig(f'output/result/LJSpeech/{basename}.png')

def test_dataloader():
    import time
    import yaml
    import numpy as np
    from torch.utils.data import DataLoader
    from dataset import PreTrainDataset, TrainDataset

    preprocess_config = "config/LJSpeech/preprocess.yaml"
    preprocess_config2 = "config/LJSpeech/preprocess_es.yaml"
    model_config = "config/LJSpeech/model.yaml"
    train_config = "config/LJSpeech/train.yaml"
    num_workers = 16

    preprocess_config = yaml.load(open(preprocess_config, "r"), Loader=yaml.FullLoader)
    preprocess_config2 = yaml.load(open(preprocess_config2, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)


    # dataset = PreTrainDataset("train.txt", preprocess_config, preprocess_config2, train_config, sort=True, drop_last=True)
    dataset = TrainDataset("train.txt", preprocess_config, preprocess_config2, train_config, sort=True, drop_last=True)

    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 1  # Set this larger than 1 (4) to enable sorting in Dataset

    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        # sampler=sampler,
    )

    print(len(loader))
    times = []
    start = time.time()
    for i, batches in enumerate(loader):
        times.append(time.time() - start)
        start = time.time()
        
        if i == 100:
            break

    # len_batches = [len(batches) for batches in loader]
    print(np.mean(times), np.std(times))                #  0.0022732298650001005 0.007514691563871479
    # Training with reverse_alignments and num_workers=0 : 0.14733001973369333 0.1094607383560366
                                                        #  0.01245397388344944 0.04326651956277088


    return

def test_DPP():
    # from pretrain_ddp import main
    from train_ddp import main

    import yaml
    import argparse
    import torch.multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--from_pretrained_ckpt", type=int)
    parser.add_argument("-p", "--preprocess_config", type=str,required=False, 
                        help="path to preprocess.yaml")
    parser.add_argument("-p2", "--preprocess_config2", type=str,required=False, 
                        help="path to second preprocess.yaml for other language")
    parser.add_argument("-m", "--model_config", type=str, required=False, help="path to model.yaml")
    parser.add_argument("-t", "--train_config", type=str, required=False, help="path to train.yaml")
    parser.add_argument("-w", "--num_workers", type=int, default=12, help="number of cpu workers for dataloader")
    args = parser.parse_args()

    args.restore_step = 0 if args.restore_step is None else args.restore_step

    preprocess_config = "config/LJSpeech/preprocess.yaml"
    preprocess_config2 = "config/LJSpeech/preprocess_es.yaml"
    model_config = "config/LJSpeech/model.yaml"
    # train_config = "config/LJSpeech/pretrain.yaml"
    train_config = "config/LJSpeech/train.yaml"

    preprocess_config = yaml.load(open(preprocess_config, "r"), Loader=yaml.FullLoader)
    preprocess2_config = yaml.load(open(preprocess_config2, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)

    configs = (preprocess_config, preprocess2_config, model_config, train_config)

    world_size = torch.cuda.device_count()
    print("World size: ", world_size)
    print("num workers: ", args.num_workers)
    mp.spawn(main, args=(args, configs, world_size,), nprocs=world_size, join=True)

def profile_data_loading():
    import cProfile
    import pstats
    from torch.utils.data import DataLoader
    from dataset import PreTrainDataset
    import yaml
    import psutil
    import time

    preprocess_config = "config/LJSpeech/preprocess.yaml"
    preprocess_config2 = "config/LJSpeech/preprocess_es.yaml"
    model_config = "config/LJSpeech/model.yaml"
    train_config = "config/LJSpeech/train.yaml"

    preprocess_config = yaml.load(open(preprocess_config, "r"), Loader=yaml.FullLoader)
    preprocess_config2 = yaml.load(open(preprocess_config2, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)

    dataset = PreTrainDataset("train.txt", preprocess_config, preprocess_config2, train_config, sort=True, drop_last=True)

    train_loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn, num_workers=0)

    # Capture initial disk I/O stats
    io_before = psutil.disk_io_counters()

    profiler = cProfile.Profile()
    profiler.enable()
    
    for i, data in enumerate(train_loader):
        time.sleep(.1)  # Introduce a small delay
        if i >= 50:  # Profile more batches
            break
    
  
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)  # Print the top 10 results

    # Capture final disk I/O stats
    io_after = psutil.disk_io_counters()

    # Calculate the difference
    read_bytes = io_after.read_bytes - io_before.read_bytes
    write_bytes = io_after.write_bytes - io_before.write_bytes

    print(f"Read bytes: {read_bytes}")
    print(f"Write bytes: {write_bytes}")

def custom_cyclic_lr():
    import torch
    from torch.optim.lr_scheduler import _LRScheduler
    import matplotlib.pyplot as plt
    import numpy as np

    class CyclicDecayLR(_LRScheduler):
        def __init__(self, optimizer, A, gamma, freq, lambd, max_lr, min_lr, last_epoch=-1):
            self.A = A
            self.gamma = gamma
            self.freq = freq
            self.lambd = lambd
            self.max_lr = max_lr
            self.min_lr = min_lr
            self.last_epoch = last_epoch

            # Set initial_lr for each param group
            for group in optimizer.param_groups:
                if 'initial_lr' not in group:
                    group['initial_lr'] = group['lr']

            super(CyclicDecayLR, self).__init__(optimizer, last_epoch)

        def get_lr(self):
            lr = self.A * np.exp(-self.gamma * self.last_epoch) * np.sin(self.last_epoch * self.freq) + \
                np.exp(-self.lambd * self.last_epoch) * self.max_lr + self.min_lr
            return [lr for _ in self.optimizer.param_groups]
            # return self.A*np.exp(-self.gamma*self.last_epoch) * np.cos(self.freq*self.last_epoch) + np.exp(-self.lambd*self.last_epoch)*self.max_lr + self.min_lr

    # Example usage Steps:
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters())
    # scheduler = CyclicDecayLR(optimizer, .0002, .0001, .002, .00005, .002, .00002)
    scheduler = CyclicDecayLR(optimizer, .0008, .0001, .001, .00005, .002, .00002, -1)

    domain = 900000
    lrs = []
    for epoch in range(0,domain,10):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_lr())

    plt.plot(lrs)
    plt.savefig("output/result/cyclic_lr_steps2.png")

    # Example
    # model = torch.nn.Linear(10, 2)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
    # scheduler = CyclicDecayLR(optimizer, .0002, .02, 2, .03, .002, .00002)

    # domain = 100
    # lrs = []
    # for epoch in range(0,domain):
    #     optimizer.step()
    #     scheduler.step()
    #     lrs.append(scheduler.get_lr())

    # plt.plot(lrs)
    # plt.savefig("output/result/cyclic_lr_epochs.png")



if __name__ == "__main__":
    # test_whisper_STT()
    # test_whisperX()
    # test_npc()
    custom_cyclic_lr()
    pass
