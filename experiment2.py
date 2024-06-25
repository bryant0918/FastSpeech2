
def test_translation():
    import os

    translation_path = os.path.join(self.in_dir, speaker, "{}_tgt.lab".format(basename))

    if not os.path.exists(translation_path):
        return None
    with open(translation_path, "r") as f:
        raw_translation = f.readline().strip("\n")

    return raw_translation

def test_google_translate():
    from deep_translator import GoogleTranslator

    text = "printing in the only sense with which we are at present concerned differs from most if not from all the arts and crafts represented in the exhibition"
    
    text = "and the aggregate amount of debts sued for was eightyone thousand seven hundred ninetyone pounds"

    # TODO: Get cleaners for translation language also (and handle api connection errors)
    translation = GoogleTranslator(source='auto', target='es').translate(text)
    # translation = _clean_text(translation, cleaners)

    print(translation)
    from text.cleaners import remove_punctuation
    translation = remove_punctuation(translation)
    print(translation)

    return translation

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
    import torch
    import time

    audio_path = "raw_data/LJSpeech/LJSpeech/LJ001-0004.wav"
    wav, _ = librosa.load(audio_path)
    wav = torch.from_numpy(wav)

    model = whisper.load_model("base").to('cpu')

    text = model.transcribe(wav)

    print(text)
    audio_path = "output/test2.wav"
    wav, _ = librosa.load(audio_path)
    wav = torch.from_numpy(wav).to('cpu')
    print("wav shape", wav.shape, wav.dtype, wav.device)
    
    start = time.time()
    text = model.transcribe(wav)
    print("Time taken: ", time.time() - start)
    print(text)
    print(text['text'])

    model = whisper.load_model("base").to('cuda')
    wav = torch.randn(120000).to(torch.float32).to("cuda")
    start = time.time()
    text = model.transcribe(wav)
    print("Time taken: ", time.time() - start)


if __name__ == "__main__":
    # test_inv_melspec()
    test_whisper_STT()

    pass
