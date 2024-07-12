import torch


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
    from text.cleaners import remove_punctuation, english_cleaners
    translation = remove_punctuation(translation)
    print(translation)

    text = "En Inglaterra, por esta época, Caslon hizo un intento en particular, quien comenzó su negocio en Londres como fundidor tipográfico en mil setecientos veinte y tenía cuarenta y cinco manzanas."
    english = GoogleTranslator(source='es', target='en').translate(translation)
    print(english)
    english = english_cleaners(english)
    print(english)


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
    import time

    audio_path = "output/result/LJSpeech/pretrain/hifigan_target9.wav"
    wav, _ = librosa.load(audio_path)
    wav = torch.from_numpy(wav)
    model = whisper.load_model("base").to('cpu')
    text = model.transcribe(wav)
    print(text)

    audio_path = "output/result/LJSpeech/pretrain/hifigan_prediction9.wav"
    wav, _ = librosa.load(audio_path)
    wav = torch.from_numpy(wav).to('cpu')
    
    start = time.time()
    text = model.transcribe(wav)
    print("Time taken: ", time.time() - start)
    print(text)
    print(text['text'])

    # model = whisper.load_model("base").to('cuda')
    # wav = torch.randn(120000).to(torch.float32).to("cuda")
    # start = time.time()
    # text = model.transcribe(wav)

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
    


if __name__ == "__main__":
    test_realign_ped()
    pass
