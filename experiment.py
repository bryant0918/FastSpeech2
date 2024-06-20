from model import FastSpeech2Pros
import torch
import yaml
import numpy as np
from synthesize import preprocess_english
from transformer.Models import BaseProsodyPredictor, ProsodyPredictor, ProsodyExtractor
import pickle
from audio.tools import get_mel_from_wav
import audio as Audio
import librosa
import time
from preprocessor.preprocessor import Preprocessor
import tgt


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Device:", device)

preprocess_config = "config/LJSpeech/preprocess.yaml"
model_config = "config/LJSpeech/model.yaml"

# Read Config
preprocess_config = yaml.load(open(preprocess_config, "r"), Loader=yaml.FullLoader)
model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)

# ASENAME LJ016-0008                                                                                                                                                    | 0/6288 [00:00<?, ?it/s]
# BASENAME LJ007-0050
basename = "LJ007-0050"


"""Test reverse alignment"""
test_reverse_alignment = False
if test_reverse_alignment:
    def flip_mapping(src_to_tgt_mapping):
        # Find the maximum target index to determine the size of the new mapping
        max_tgt_idx = int(src_to_tgt_mapping.max())
        
        # Initialize a list of lists to store the target to source mappings
        tgt_to_src_mapping = [[] for _ in range(max_tgt_idx + 1)]
        
        # Iterate through each source index and its corresponding target indices
        for src_idx, tgt_indices in enumerate(src_to_tgt_mapping):
            for tgt_idx in tgt_indices:
                if tgt_idx > 0:
                    tgt_to_src_mapping[tgt_idx].append(src_idx)
        
        # Convert lists to tensors and pad to have uniform shape
        max_len = max(len(tgt_indices) for tgt_indices in tgt_to_src_mapping)
        tgt_to_src_mapping_padded = torch.zeros((len(tgt_to_src_mapping), max_len), dtype=torch.long)
        for i, tgt_indices in enumerate(tgt_to_src_mapping):
            tgt_to_src_mapping_padded[i, :len(tgt_indices)] = torch.tensor(tgt_indices, dtype=torch.long)
        
        return tgt_to_src_mapping_padded

    # Example tensor
    src_to_tgt_mapping = torch.tensor([[0,0,0],[1,2,0],[0,3,4],[0,0,0]])

    # Flip the mapping
    tgt_to_src_mapping = flip_mapping(src_to_tgt_mapping)

    print(tgt_to_src_mapping)

"""Test Whisper AI for ASR"""
test_whisper = False
if test_whisper: 
    import whisper
    import torch.nn.functional as F
    import torchaudio.functional as audio_F

    model = whisper.load_model("base")

    audio = whisper.load_audio("raw_data/LJSpeech/LJSpeech/LJ001-0004.wav")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    print("Their Mel shape: ", mel.size())  # Requires [80, 3000]

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    print("Result: ", result.text)

    mel_path = "preprocessed_data/LJSpeech/mel/LJSpeech-mel-LJ001-0004.npy"
    melspec = torch.from_numpy(np.load(mel_path)).to(device)

    # melspec = melspec.unsqueeze(0).unsqueeze(0)   # To get dimension [1,1,W:X, H:80]
    print("Mel shape: ", melspec.size())

    melspec = melspec.permute(1,0)
    print("Mel shape: ", melspec.size())

    # Padding to achieve size [80, 3000]
    target_size = 3000
    current_size = melspec.size(1)  # Size of the second dimension (145)
    padding_size = target_size - current_size  # Padding needed (3000 - 145)

    # Apply padding
    padded_tensor = F.pad(melspec, (0, padding_size))

    # Compute the Log-Mel Spectrogram
    # log_mel_spectrogram = torch.log(padded_tensor + 1e-9)  # Adding a small value to avoid log(0)
    # Whisper transform (uses base 10 and normalizes)
    log_spec = torch.clamp(padded_tensor, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    # Verify the size
    print(log_spec.size())  # Should print torch.Size([80, 3000])


    # detect the spoken language
    _, probs = model.detect_language(log_spec)
    print(f"Detected language: {max(probs, key=probs.get)}")

    options = whisper.DecodingOptions()
    result = whisper.decode(model, log_spec, options)

    # print the recognized text
    print(result.text)

    print("options: ", options)

    get_mel = True
    if get_mel:
        config = preprocess_config
        print()
        STFT = Audio.stft.TacotronSTFT(
            400, #n_fft
            160, #hop_length
            400, #win_length
            config["preprocessing"]["mel"]["n_mel_channels"],
            1600, #sampling_rate
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

        wav_path = "raw_data/LJSpeech/LJSpeech/LJ001-0004.wav"
        sampling_rate = 1600

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = torch.from_numpy(wav.astype(np.float32)).to(device)

        # melspec, energy = get_mel_from_wav(wav, STFT)

        N_FFT = 400
        HOP_LENGTH = 160
        n_mels = 80

        # mel_spec = librosa.feature.melspectrogram(
        #     y=wav,
        #     sr=sampling_rate,
        #     n_fft=N_FFT,
        #     hop_length=HOP_LENGTH,
        #     n_mels=n_mels
        # )
        # print("Mel shape: ", np.shape(mel_spec))

        window = torch.hann_window(N_FFT).to(device)
        stft = torch.stft(wav, N_FFT, HOP_LENGTH, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        
        # import os
        # filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
        # with np.load(filters_path, allow_pickle=False) as f:
        #     filters = torch.from_numpy(f[f"mel_{n_mels}"]).to(device)
        # filters = mel_filters(wav.device, n_mels)

        mel_filters_librosa = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=N_FFT,
            n_mels=n_mels,
            fmin=0.0,
            fmax=sampling_rate / 2.0,
            norm="slaney",
            htk=True,
        )

        mel_filters = audio_F.melscale_fbanks(
            int(N_FFT // 2 + 1),
            n_mels=n_mels,
            f_min=0.0,
            f_max=sampling_rate / 2.0,
            sample_rate=sampling_rate,
            norm="slaney",
        ).to(device).T

        mel_filters_librosa = torch.from_numpy(mel_filters_librosa).to(device)
        
        mel_spec = mel_filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        # log_spec = torch.from_numpy(log_spec).to(device)

        print("log_spec shape: ", log_spec.size())

        # melspec = melspec.permute(1,0)
        # print("Mel shape: ", melspec.size())

        # # Padding to achieve size [80, 3000]
        # target_size = 3000
        # current_size = np.shape(mel_spec)[1]  # Size of the second dimension (145)
        # padding_size = target_size - current_size  # Padding needed (3000 - 145)

        # # Apply padding
        # padded_tensor = np.pad(mel_spec, ((0, 0), (0, padding_size)), mode='constant')
        # Padding to achieve size [80, 3000]
        target_size = 3000
        current_size = log_spec.size(1)  # Size of the second dimension (145)
        padding_size = target_size - current_size  # Padding needed (3000 - 145)

        # Apply padding
        padded_tensor = F.pad(log_spec, (0, padding_size))
        # Convert to log scale (dB)
        # log_mel_spec = librosa.power_to_db(padded_tensor, ref=np.max)
        # print("Log Mel shape: ", log_mel_spec.shape)

        # Compute the Log-Mel Spectrogram
        # log_mel_spectrogram = torch.log(padded_tensor + 1e-9)  # Adding a small value to avoid log(0)
        # Whisper transform (uses base 10 and normalizes)
        # padded_tensor = torch.from_numpy(padded_tensor).to(device)
        log_spec = torch.clamp(padded_tensor, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        # Verify the size
        # log_mel_spec = torch.from_numpy(log_mel_spec).to(device)
        print(log_spec.size())  # Should print torch.Size([80, 3000])


        # detect the spoken language
        _, probs = model.detect_language(log_spec)
        print(f"Detected language: {max(probs, key=probs.get)}")

        options = whisper.DecodingOptions()
        result = whisper.decode(model, log_spec, options)

        # print the recognized text
        print(result.text)

        print("options: ", options)


"""Debug segmentatin fault"""  # Fixed by export LD_LIBRARY_PATH=""
seg_fault = False
if seg_fault:
    import torch.nn as nn

    # Assuming output is your tensor and model definition
    output = torch.randn(16, 256, 102).cuda()  # Example tensor on CUDA

    # Print output details
    print("output", output.shape, output.device, output.dtype)

    # Assuming self.w_1 is a nn.Conv1d layer
    class ExampleModel(nn.Module):
        def __init__(self):
            super(ExampleModel, self).__init__()
            self.w_1 = nn.Conv1d(in_channels=256, out_channels=1024, kernel_size=9, stride=1, padding=4)
        
        def forward(self, x):
            return self.w_1(x)

    # Create an instance of the model
    model = ExampleModel()
    model = model.cuda()  # Move model to CUDA if not already there

    # Print weights of self.w_1
    print("Weights of self.w_1:", model.w_1.weight.size())

    # Attempt to forward pass
    try:
        output_after_w1 = model.w_1(output)
        print("Output shape after self.w_1:", output_after_w1.shape)
    except Exception as e:
        print("Error during forward pass:", e)

"""Test phoneme realignment"""
# Switch phone embeddings order to match target language through alignment model
# aligned_split_phones = split_phones * alignments
phone_realignment = False
if phone_realignment:
    import epitran
    phone_alignment_path = "preprocessed_data/LJSpeech/alignments/phone/LJSpeech-phone_alignment-LJ001-0048.pkl"
    src_text_path = "raw_data/LJSpeech/LJSpeech/LJ001-0048_src.lab"
    tgt_text_path = "raw_data/LJSpeech/LJSpeech/LJ001-0048_tgt.lab"
    mel_path = "preprocessed_data/LJSpeech/mel/LJSpeech-mel-LJ001-0048.npy"
    durations_path = "preprocessed_data/LJSpeech/duration/LJSpeech-duration-LJ001-0048.npy"

    src_phones = "{HH IH1 Z L EH1 T ER0 IH0 Z AE1 D M ER0 AH0 B L IY0 K L IH1 R AE1 N D R EH1 G Y AH0 L ER0 sp B AH1 T " \
                 "sp AE1 T L IY1 S T EH1 Z B Y UW1 T AH0 F AH0 L AE1 Z EH1 N IY0 AH1 DH ER0 R OW1 M AH0 N T AY1 P}"

    with open(phone_alignment_path, 'rb') as f:
        phone_alignments = pickle.load(f)

    with open(src_text_path, 'r') as f:
        src_text = f.read()

    with open(tgt_text_path, 'r') as f:
        tgt_text = f.read()

    epi = epitran.Epitran('spa-Latn')  # TODO: Change based on language

    tgt_phones = epi.transliterate(tgt_text)
    tgt_phones = tgt_phones.split()

    d_targets = torch.from_numpy(np.load(durations_path)).unsqueeze(0).to(device)
    # print("duration shape: ", d_targets.size())

    get_split_phones = False
    if get_split_phones:

        mels = torch.from_numpy(np.load(mel_path)).unsqueeze(0).unsqueeze(0).to(device)
        print("Mel shape: ", mels.size())

        # print("mel_lengths", mel_lens)
        # mel_masks = (get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None)

        # prosody extractor
        prosody_extractor = ProsodyExtractor(1, 128, 8).to(device)
        e_src = prosody_extractor(mels)  # e is [batch_size, melspec H, melspec W, 128]

        print("e_src shape: ", e_src.size())
        # Split phone embeddings by phone
        # [batch_size (list), phoneme_sequence_length (list), melspec H (tensor), melspec W (tensor), 128 (tensor)]
        split_phones = prosody_extractor.split_phones(e_src, d_targets, device)

        print("split_phones", len(split_phones))
        torch.save(split_phones, "preprocessed_data/Bryant/split_phones.pt")
        split_src_phones = split_phones
    else:
        split_src_phones = torch.load("preprocessed_data/Bryant/split_phones.pt")

    # print(type(split_phones), len(split_phones))
    print(type(split_src_phones[0]), [len(phone) for phone in split_src_phones])
    # print([split_phones[0][i].size() for i in range(len(split_phones[0]))])
    print("d_targets", d_targets)
    print()

    # Get prosody prediction  ( I may want 256 output to be able to concat)
    prosody_predictor = ProsodyPredictor(256, 128, 4, 8).to(device)
    h_si = torch.rand([2, 88, 256], device=device)
    h_sd = torch.rand([2, 88, 256], device=device)

    e = prosody_predictor(h_sd, h_si)
    tgt_samp = prosody_predictor.sample2(e)
    print("Target Sample shape: ", tgt_samp.size())
    # print(tgt_samp[0, 61])
    # print(tgt_samp[0, 87])

    beta = 0.1
    new_e = torch.zeros(2, 88, 128, device=device)
    sum_results = torch.zeros(2, 88, 128, device=device)  # Tensor to accumulate results
    counts = torch.zeros(2, 88, device=device)  # Tensor to keep count of how many times each index is updated

    # print("Phone aligments", len(phone_alignments), phone_alignments[0])

    from utils.tools import pad_inhomogeneous_2D
    phone_alignments2 = phone_alignments[:50]

    phone_alignments = pad_inhomogeneous_2D([phone_alignments, phone_alignments2])
    phone_alignments = torch.from_numpy(phone_alignments).int().to(device)
    print("Phone aligments", phone_alignments.shape)
    print(len(split_src_phones))
    split_src_phones = [split_src_phones[0], split_src_phones[0]]

    
    print("phone_alignments", phone_alignments.shape)
    for b in range(len(phone_alignments)):
        for j in range(len(phone_alignments[b])):
            for i in phone_alignments[b][j]:
                # Reshape B to be broadcastable to A's shape
                B_broadcasted = tgt_samp[b][j].unsqueeze(0).unsqueeze(0)

                # Compute the weighted combination
                result = (1 - beta) * B_broadcasted + beta * split_src_phones[b][i]
                new_mean = result.mean(dim=(0, 1))  # [128]

                # Accumulate the new_mean for each index
                # sum_results[b][:, i] += new_mean
                new_e[b, i] += new_mean
                counts[b, i] += 1

        # Average the accumulated results
        for i in range(88):
            if counts[b][i] > 0:
                new_e[b, i] /= counts[b][i]
    print("New e shape: ", new_e[0, 61])
    print("New e shape: ", new_e.size(), new_e[0, 84])

    print(len(phone_alignments), phone_alignments)
    print()

"""Test TextGrid"""
test_textgrid = True
if test_textgrid:
    import tgt
    from itertools import chain
    import os

    print("\n TESTING TEXTGRID GET ALIGNMENT")
        
    def get_alignment(textgrid):
        phones_tier = textgrid.get_tier_by_name("phones")
        words_tier = textgrid.get_tier_by_name("words")
        word_end_times = [w.end_time for w in words_tier._objects]

        sil_phones = ["sil", "sp"]   # Not 'spn'

        all_phones, word_phones, durations = [], [], []
        start_time, end_time = 0, 0
        end_idx, word_idx = 0, 0
        num_phones, num_words = 0, 0

        for t in phones_tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            # print(p, s, e)
            # print("Word_idx", word_idx, words_tier.intervals[word_idx].text, word_end_times[word_idx], words_tier.intervals[word_idx].end_time)

            # Trim leading silences
            if not all_phones and not word_phones:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                word_phones.append(p)
                num_phones += 1

                if word_end_times[word_idx] == e:
                    all_phones.append(word_phones)
                    word_phones = []
                    end_time = e
                    end_idx = num_phones
                    num_words += 1
                    end_word = num_words

                    if word_idx == len(words_tier.intervals) - 1:  # That was the last word
                        durations.append(int(np.round(e * 22050 / 256) - np.round(s * 22050 / 256)))
                        break
                    word_idx += 1

            else:  # For silent phones (don't need em, don't want em)
                continue

            durations.append(int(np.round(e * 22050 / 256) - np.round(s * 22050 / 256)))

        print("Phones: ", all_phones)
        print(len(all_phones), word_idx, num_words, end_idx)
        print('length of durations pre slice', len(durations))

        # Trim tailing silences
        phones = all_phones[:num_words]
        durations = durations[:end_idx]

        flat_phones = list(chain.from_iterable(phones))
        print("flat_phones: ", len(flat_phones))
        print("Durations: ", len(durations))

        print(end_idx, num_words, end_word, word_idx)

        return phones, durations, start_time, end_time

    tg_path = os.path.join("preprocessed_data/Bryant", "TextGrid", "Bryant", "{}.TextGrid".format("LJ001-002"))
    # tg_path = "preprocessed_data/LJSpeech/TextGrid/LJSpeech/LJ001-0002.TextGrid"
    # tg_path = "preprocessed_data/LJSpeech/TextGrid/LJSpeech/LJ032-0007.TextGrid"
    # tg_path = "preprocessed_data/LJSpeech/TextGrid/LJSpeech/LJ036-0179.TextGrid"
    # tg_path = "preprocessed_data/LJSpeech/TextGrid/LJSpeech/LJ030-0041.TextGrid"
    # tg_path = "preprocessed_data/LJSpeech/TextGrid/LJSpeech/LJ002-0298.TextGrid"
    tg_path = f"preprocessed_data/LJSpeech/TextGrid/LJSpeech/{basename}.TextGrid"
    textgrid = tgt.io.read_textgrid(tg_path)

    phones, durations, start, end = get_alignment(textgrid)


"""Test Epitran"""
test_epitran = False
if test_epitran:
    import epitran
    from text.ipadict import db
    from string import punctuation
    import re

    epi = epitran.Epitran("spa-Latn")

    print(punctuation)
    new_punc = "¡!\"#$%&'()*+,-./:;<=>¿?@[\]^_`{|}~"

    text = "¡Hola! ¿cómo estás niño?"
    test_str = text.translate(str.maketrans('', '', new_punc))
    print(test_str)

    res = re.sub(r'[^\w\s]', '', text)  # Remove all non word or space characters
    print(res)

    phones = epi.transliterate(test_str)
    print(phones)

    print(phones.split())
    phones_by_word = phones.split()
    # for word in phones_by_word:
    #     for char in word:
    #         print(char, db[char]['unicode'], db[char]['id'])

    phones_by_word = [[char for char in word] for word in phones_by_word]
    print(phones_by_word)

    text = "Cześć, jak się masz?"
    test_str = text.translate(str.maketrans('', '', new_punc))
    epi = epitran.Epitran("pol-Latn")
    phones = epi.transliterate(test_str)
    print("EPITRAN:", phones)

    phones_by_word = phones.split()

    from text.tools import split_with_tie_bar

    print("EPITRAN:", [[char for char in word] for word in phones_by_word])
    phones_by_word = [split_with_tie_bar(word) for word in phones_by_word]
    print("EPITRAN:", phones_by_word)
    print("EPITRAN: ", list(chain.from_iterable(phones_by_word)))


"""Go through word_alignment"""
test_word_alignment = False
if test_word_alignment:
    word_alignment = np.array([[0, 0], [1, 1], [2, 2], [3, 2], [3, 3]])
    words = 4
    phoneme_alignment = {i: [] for i in range(words)}

    # pth = "preprocessed_data/Bryant/alignments/phone/Bryant-phone_alignment-LJ001-0002.npy"
    # phone_alignment = np.load(pth, allow_pickle=True)
    # print(phone_alignment)

    pth = "preprocessed_data/LJSpeech/alignments/phone/LJSpeech-phone_alignment-LJ001-0002.pkl"
    # Loading the dictionary
    with open(pth, 'rb') as f:
        phone_alignments = pickle.load(f)

    print(phone_alignments)
    print(len(phone_alignments))

"""Test Sentence Aligner"""
test_aligner = True
if test_aligner:
    from simalign import SentenceAligner
    # making an instance of our model.
    # You can specify the embedding model and all alignment settings in the constructor.
    myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="m")

    print("\n TESTING SENTENCE ALIGNER")

    # The source and target sentences should be tokenized to words.
    src_sentence = ["Hello,", "my", "name", "is", "Ditto", "and", "this", "is", "what", "I", "sound", "like"]
    # trg_sentence = ["Hallo,", "mein", "Name", "ist", "Ditto", "und", "so", "klinge", "ich"]
    trg_sentence = ["Cześć,", "nazywam", "się", "„Ditto”", "i", "tak", "właśnie", "brzmię"]
    """
    Polish:
    mwmf: [(0, 0), (1, 1), (2, 2), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6), (8, 0), (9, 7), (10, 8), (11, 8), (12, 8),
           (13, 9)]
    inter: [(0, 0), (1, 1), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6), (11, 8), (13, 9)]
    itermax: [(0, 0), (1, 1), (2, 2), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6), (9, 7), (10, 8), (11, 8), (12, 8),
              (13, 9)]
    """

    # The output is a dictionary with different matching methods.
    # Each method has a list of pairs indicating the indexes of aligned words (The alignments are zero-indexed).

    alignments = myaligner.get_word_aligns(src_sentence, trg_sentence)

    for matching_method in alignments:
        print(matching_method, ":", alignments[matching_method])

    src = "motorcycles. -- dallas police motorcycles preceded the pilot car."
    tgt = "motocicletas. -- las motocicletas de la policia de dallas precedieron al coche del piloto."
    src = "he now married although his salary was only a pound a week but he soon got on"
    tgt = "ahora se caso aunque su salario era solo de una libra por semana pero pronto consiguio"

    pth = f"raw_data/LJSpeech/LJSpeech/{basename}_src.lab"
    with open(pth, 'r') as f:
        src = f.read()
    print("Src text: ", src)

    # text = "que tienen la responsabilidad principal de suministrar informacion sobre amenazas potenciales,"
    pth = f"raw_data/LJSpeech/LJSpeech/{basename}_tgt.lab"
    with open(pth, 'r') as f:
        tgt = f.read()
    print("Tgt text: ", tgt)
    
    alignments = myaligner.get_word_aligns(src.split(), tgt.split())
    print("Src.split(): ", len(src.split()), src.split())
    print("Tgt.split(): ", len(tgt.split()), tgt.split())
    print(alignments)

    tgt = "d i n e ɾ o k e m o i s e s a b j a ɾ e s i b i d o e l p o l b o d e o ɾ o ɾ o b a d o d e l s w e ɡ ɾ o d e m o s s , d a b i s o i s a a k s , k j e n n u n k a f w e a r e s t a d o ."
    src = "IH1 T M AH1 S T HH AE1 V AH0 P IH1 R D T UW1 HH IH1 M sp DH AH0 T HH IY1 W AH0 Z sp AH0 N EY1 B AH0 L T AH0 K AH0 M AE1 N D sp IY1 V IH0 N DH IY0 AH0 T EH1 N SH AH0 N AH0 V HH IH1 Z F AE1 M L IY0"
    print(len(tgt.split()), len(src.split()))

"""Get different speaker embedding"""
test_embedding = False
if test_embedding:
    embedding_path = "/Users/bryantmcarthur/Documents/Ditto/SpeakerEncoder/outputs/tony.pkl_emb.pkl"

    with open(embedding_path, 'rb') as f:
        emb_dict = pickle.load(f)

    embedding = torch.from_numpy(emb_dict["default"]).to(device).unsqueeze(0).unsqueeze(0).expand(-1, 19, -1)
    print("Embedding shape: ", embedding.size())   # Mueller's embedding is 256, other is 128. We need 256

    embedding_copy = embedding.clone()
    embedding_copy2 = embedding_copy.clone()
    start = time.time()
    speaker_embs = torch.cat([embedding, embedding_copy, embedding_copy2], dim=0)

    print("Speaker Embeddings shape: ", speaker_embs.size(), "Time: ", time.time() - start)

"""Test Prosody Extractor"""
test_extractor = False
if test_extractor:
    # Create the model
    model = ProsodyExtractor(1, 128, 8).to(device)
    # Print the model summary
    print(model)
    config = preprocess_config

    get_mel = False
    if get_mel:
        STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

        wav_path = "/Users/bryantmcarthur/Documents/Ditto/experiment/tony.flac"
        sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = torch.from_numpy(wav.astype(np.float32))

        melspec, energy = get_mel_from_wav(wav, STFT)

        melspec = torch.from_numpy(melspec).to(device)
    else:
        mel_path = "preprocessed_data/LJSpeech/mel/LJSpeech-mel-LJ001-0002.npy"
        text_grid_path = "preprocessed_data/LJSpeech/TextGrid/LJSpeech/LJ001-0002.TextGrid"

        melspec = torch.from_numpy(np.load(mel_path)).to(device)

    melspec = melspec.unsqueeze(0).unsqueeze(0)   # To get dimension [1,1,W:X, H:80]
    print("Mel shape: ", melspec.size())

    e = model(melspec)
    print("e shape: ", e.size())  # [1, W:80, H:X, 128]
    print("shape of e after view", e.view(e.size()[0], 80, -1, e.size()[-1]).size())
    e = e.view(e.size()[0], 80, -1, e.size()[-1])

    # Somehow split this up by phoneme time steps
    preprocessor = Preprocessor(preprocess_config)
    # Get alignments TODO: Get phones from train.txt, and durations from duration folder
    textgrid = tgt.io.read_textgrid(text_grid_path)
    phone, duration, start, end = preprocessor.get_alignment(textgrid.get_tier_by_name("phones"))
    print("Phones", phone, len(phone))
    print("Duration", duration, len(duration))

    duration_path = "preprocessed_data/LJSpeech2/duration/LJSpeech-duration-LJ001-0002.npy"
    duration = np.load(duration_path)
    print("Duration", duration, len(duration)) # They match!

    print("start", start)
    print("end", end)

    phone_emb_chunks = []
    start_frame = 0
    for i in range(len(duration)):
        phone_emb_chunks.append(e[:, :, start_frame:start_frame + duration[i]])
        start_frame += duration[i]

    print("Phone emb chunks: ", len(phone_emb_chunks), phone_emb_chunks[0].size(), phone_emb_chunks[1].size(),
          phone_emb_chunks[2].size(), phone_emb_chunks[3].size(), phone_emb_chunks[4].size(), phone_emb_chunks[5].size())

    total_len = sum([phone_emb_chunk.size()[2] for phone_emb_chunk in phone_emb_chunks])
    print(total_len)


"""Test Prosody Predictor"""
test_predictor = False
if test_predictor:
    model = FastSpeech2Pros(preprocess_config, model_config).to(device)

    text = "Hello, how are you doing today?"
    texts = np.array([preprocess_english(text, preprocess_config)])

    print("Texts shape: ", texts.shape)

    ids = raw_texts = [text[:100]]
    speakers = np.array([0])
    text_lens = np.array([len(texts[0])])

    speakers = torch.from_numpy(speakers).long().to(device)
    texts = torch.from_numpy(texts).long().to(device)
    src_lens = torch.from_numpy(text_lens).to(device)

    max_src_lens = max(text_lens)
    print(max_src_lens)

    batch = (speakers, texts, src_lens, max_src_lens)

    with torch.no_grad():
        # Forward
        output = model(*batch)[0]
        print("Output shape: ", output.size())

    # h_si = torch.rand([1, 19, 256], device=device)
    # print("h_si shape: ", h_si.size())
    #
    # # Get Speaker Embedding created by speaker embedding repo
    # # style_wav = "/Users/bryantmcarthur/Documents/Ditto/experiment/tony.flac"
    # embedding_path = "/Users/bryantmcarthur/Documents/Ditto/SpeakerEncoder/outputs/tony.pkl_emb.pkl"
    #
    # with open(embedding_path, 'rb') as f:
    #     emb_dict = pickle.load(f)
    #
    # embedding = torch.from_numpy(emb_dict["default"]).to(device)
    # print("Embedding shape: ", embedding.size())
    #
    # # embedding = torch.cat((embedding, embedding))
    # # print("Embedding shape: ", embedding.size())
    #
    # # Adding new dimensions to tensor_2
    
    # embedding = embedding.unsqueeze(0).unsqueeze(0).expand(-1, 19, -1)
    #
    # h_sd = h_si + embedding
    #
    # print("h_sd shape: ", h_sd.size())
    #
    # # Create the model
    # model = ProsodyPredictor(256, 128, 4, 8).to(device)
    # # Print the model summary
    # print(model)
    #
    # e = model(h_sd, h_si)
    #
    # pi, mu, sigma = e
    # print("pi shape: ", pi.size())
    # print("mu shape: ", mu.size())
    # print("sigma shape: ", sigma.size())
    #
    # sample = model.sample2(h_sd, h_si)
    # print("Sample shape: ", sample.size())

    # sample = model.sample(h_sd, h_si)
    # print("Sample shape: ", sample.size())

"""Test phone alignment"""
test_phone_alignment = True
if test_phone_alignment:
    from itertools import chain

    print("\n TESTING PHONEME ALIGNER")

    def get_phoneme_alignment(word_alignments, src_phones, tgt_phones):
        """
        word_alignments: list of tuples of word alignments [(src_word_idx, tgt_word_idx), ...]
        src_phones: list of list of phones for each word in src language [[phone1, phone2, ...], ...]
        tgt_phones: list of list of phones for each word in tgt language [[phone1, phone2, ...], ...]
        Return:
        flat_phone_alignments: list of list of phone alignments for each tgt phone [[src_phone_idx, ...], ...]
        """
        phone_alignments = {}

        print("src_phones", len(src_phones), src_phones)
        print("tgt_phones", len(tgt_phones), tgt_phones)
        # print("word_alignments", word_alignments)
        
        src_phone_cumsums = np.cumsum([len(src_phone) for src_phone in src_phones])
        tgt_phone_cumsums = np.cumsum([len(tgt_phone) for tgt_phone in tgt_phones])

        print("cumsums", src_phone_cumsums, len(src_phone_cumsums))
        print("cumsums", tgt_phone_cumsums, len(tgt_phone_cumsums))
        
        # To flatten phones
        flat_src_phones = list(chain.from_iterable(src_phones))
        flat_tgt_phones = list(chain.from_iterable(tgt_phones))
        print("Flat tgt phones length", len(flat_tgt_phones))
        print("Flat src phones length", len(flat_src_phones))

        flat_phone_alignments = []
        flat_phone_alignments = [[] for _ in range(len(flat_tgt_phones))]

        for word_alignment in word_alignments:
            i, j = word_alignment[0], word_alignment[1]

            if i == 0:
                flat_src_phones_idx = 0
            
            else:
                flat_src_phones_idx = src_phone_cumsums[i-1]

            if j == 0:
                flat_tgt_phones_idx = 0
            else:
                flat_tgt_phones_idx = tgt_phone_cumsums[j-1]

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

            # print("len of tgt_wrd_phones", len(tgt_word_phones))
            while tgt_phone < len(tgt_word_phones):
                if flat_src_phones_idx == len(flat_src_phones):
                    flat_src_phones_idx -= 1
                    current_src_phone -= 1
                
                if (1-phone_accumulations) > phone_weight:   # Use all of the phone_weight left
                    phone_accumulations += phone_weight
                    if current_src_phone not in phone_alignment:
                        phone_alignment.append(current_src_phone)
                        flat_phone_alignment.append(flat_src_phones_idx)
                        
                    # Reset
                    phone_weight = len(src_word_phones) / len(tgt_word_phones)
                    tgt_phone += 1
                    
                    the_word.append(phone_alignment)
                    
                    flat_phone_alignments[flat_tgt_phones_idx].extend(flat_phone_alignment)
                                        
                    flat_tgt_phones_idx += 1
                    phone_alignment = []
                    flat_phone_alignment = []

                elif phone_weight == (1-phone_accumulations):   # Use all of the phone_weight left
                    phone_alignment.append(current_src_phone)
                    flat_phone_alignment.append(flat_src_phones_idx)
                    phone_accumulations = 0
                    current_src_phone += 1
                    flat_src_phones_idx += 1
                    tgt_phone += 1
                    
                    phone_weight = len(src_word_phones) / len(tgt_word_phones)
                    the_word.append(phone_alignment)
                    
                    flat_phone_alignments[flat_tgt_phones_idx].extend(flat_phone_alignment)
                    
                    flat_tgt_phones_idx += 1
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

            if j in phone_alignments[i] and phone_alignments[i][j]:
                # Append the_word to existing alignments
                for k, corresponding_tgt_phones in enumerate(the_word):
                    if k in phone_alignments[i][j]:
                        phone_alignments[i][j][k].extend(corresponding_tgt_phones)
                    else:
                        phone_alignments[i][j][k] = corresponding_tgt_phones
            else:
                phone_alignments[i][j] = {k: corresponding_tgt_phones for k, corresponding_tgt_phones in enumerate(the_word)}

        # TODO: Get rid of phone_alignments?
        # print("Phone alignments", phone_alignments)
        print()

        return flat_phone_alignments

    pth = f"preprocessed_data/LJSpeech/alignments/word/LJSpeech-word_alignment-{basename}.npy"
    # Loading the dictionary
    # with open(pth, 'rb') as f:
    #     word_alignments = pickle.load(f)
    word_alignments = np.load(pth)
    print("Word alignments", word_alignments)

    import epitran
    from string import punctuation
    import re

    epi = epitran.Epitran("spa-Latn")

    new_punc = "¡!\"#$%&'()*+,-./:;<=>¿?@[\]^_`{|}~"

    pth = f"raw_data/LJSpeech/LJSpeech/{basename}_src.lab"
    with open(pth, 'r') as f:
        text = f.read()
    print("Src text: ", text)

    # text = "que tienen la responsabilidad principal de suministrar informacion sobre amenazas potenciales,"
    pth = f"raw_data/LJSpeech/LJSpeech/{basename}_tgt.lab"
    with open(pth, 'r') as f:
        text = f.read()
    print("Tgt text: ", text)

    test_str = text.translate(str.maketrans('', '', new_punc))

    res = re.sub(r'[^\w\s]', '', text)  # Remove all non word or space characters

    my_phones = epi.transliterate(test_str)

    phones_by_word = my_phones.split()
    # for word in phones_by_word:
    #     for char in word:
    #         print(char, db[char]['unicode'], db[char]['id'])

    phones_by_word = [[char for char in word] for word in phones_by_word]
    print("phones_by_word", phones_by_word)
    print("phones", phones)

    # src_phones = "W IH1 CH K EH1 R IY0 DH AH0 M EY1 JH ER0 R IY0 S P AA2 N S AH0 B IH1 L AH0 T IY0 sp F R ER0 S AH0 P L AY1 IH0 NG IH2 N F ER0 M EY1 SH AH0 N AH0 B AW1 T P AH0 T EH1 N CH AH0 L TH R EH1 T S"
    # tgt_phones = "e s k ɾ i b j o d e s p w e s e n e l s e n t i d o d e k e e n e l m o m e n t o e n k e s e a b j a o f ɾ e s i d o u n a s e s i n o d e s k o n o s i d o b i n o a a s e s t a ɾ l e u n ɡ o l p e"

    # print("Length of tgt_phones", len(tgt_phones.split()))
    phone_alignment = get_phoneme_alignment(word_alignments, phones, phones_by_word)

    print("Flat Phone alignment", len(phone_alignment), phone_alignment)

    from utils.tools import pad_inhomogeneous_2D, flip_mapping
    alignments = torch.from_numpy(pad_inhomogeneous_2D([phone_alignment, phone_alignment[:26]])).int().to(device)

    print("Alignments", alignments.shape, alignments.max().item())

    max_src_idx=len(list(chain.from_iterable(phones)))
    print("Max src idx", max_src_idx)

    reverse_alignments = flip_mapping(alignments, src_seq_len=max_src_idx)
    print("Reverse Alignments", reverse_alignments.shape, reverse_alignments.max().item())
    print(reverse_alignments[0])

    # src_pitch = np.load("preprocessed_data/LJSpeech/pitch/LJSpeech-pitch-LJ050-0116.npy")
    src_pitch = np.load(f"preprocessed_data/LJSpeech/pitch/LJSpeech-pitch-{basename}.npy")

"""Test realign p_e_d"""
test_realign_p_e_d = False
if test_realign_p_e_d:
    from utils.tools import pad_inhomogeneous_2D, pad_1D
    from text import text_to_sequence

    src_pitch = np.load("preprocessed_data/LJSpeech/pitch/LJSpeech-pitch-LJ050-0116.npy")
    src_pitch = torch.from_numpy(src_pitch).to(device).unsqueeze(0)
    print("src_pitch", src_pitch.shape, src_pitch)

    def realign_p_e_d(alignments, p_e_d):
        new_pitch = torch.zeros(p_e_d.size(0), len(alignments[0]), device=p_e_d.device)
        for b, alignment in enumerate(alignments):
            for j, src_indices in enumerate(alignment):
                new_pitch[b][j] = torch.mean(torch.tensor([p_e_d[b][i] for i in src_indices]))
        return new_pitch

    
    src_phones = "{K AO1 Z D M AH1 CH IH0 K S AY1 T M AH0 N T sp AE1 T DH AH0 T AY1 M sp AA1 N AH0 K AW1 N T AH0 V DH AH0 M AE1 G N AH0 T UW2 D AH0 V DH AH1 F R AO1 D sp AE1 N D DH IY0 S IY1 M IH0 NG P R OW1 B AH0 T IY0 AH0 V DH AH0 K AH1 L P R IH0 T}"
    tgt_phones = "{k a u s o m u t ͡ ʃ o ɾ e b w e l o e n e s e m o m e n t o d e b i d o a l a m a ɡ n i t u d d e l f ɾ a u d e i l a a p a ɾ e n t e p ɾ o b i d a d d e l k u l p a b l e .}"

    print(len(src_phones.split()), len(tgt_phones.split()))

    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
    src_phone = np.array(text_to_sequence(src_phones, cleaners, 'en'))
    tgt_phone = np.array(text_to_sequence(tgt_phones, cleaners, 'es'))

    print("len(src_phone)", len(src_phone))
    print("len(tgt_phone)", len(tgt_phone))

    pth = "preprocessed_data/LJSpeech/alignments/word/LJSpeech-word_alignment-LJ010-0299.npy"
    word_alignments = np.load(pth)

    text = "Causó mucho revuelo en ese momento debido a la magnitud del fraude y la aparente probidad del culpable."
    test_str = text.translate(str.maketrans('', '', new_punc))

    res = re.sub(r'[^\w\s]', '', text)  # Remove all non word or space characters

    my_phones = epi.transliterate(test_str)

    phones_by_word = my_phones.split()
    # for word in phones_by_word:
    #     for char in word:
    #         print(char, db[char]['unicode'], db[char]['id'])

    phones_by_word = [[char for char in word] for word in phones_by_word]
    # print(phones_by_word)

    pth = "preprocessed_data/LJSpeech/alignments/phone/LJSpeech-phone_alignment-LJ010-0299.pkl"
    with open(pth, 'rb') as f:
        saved_phone_alignment = pickle.load(f)
    # print("saved phone alignments", saved_phone_alignment)

    tg_path = "preprocessed_data/LJSpeech/TextGrid/LJSpeech/LJ010-0299.TextGrid"
    textgrid = tgt.io.read_textgrid(tg_path)

    phones, durations, start, end = get_alignment(textgrid)

    phone_alignment2 = get_phoneme_alignment(word_alignments, phones, phones_by_word)

    # print("Flat Phone alignment", len(phone_alignment), phone_alignment)

    alignments = torch.from_numpy(pad_inhomogeneous_2D([phone_alignment, phone_alignment2])).int().to(device)

    # print("Alignments", alignments.shape, alignments)

    src_pitch = np.load("preprocessed_data/LJSpeech/pitch/LJSpeech-pitch-LJ050-0116.npy")
    src_pitch2 = np.load("preprocessed_data/LJSpeech/pitch/LJSpeech-pitch-LJ010-0299.npy")
    # src_pitch2 = torch.from_numpy(src_pitch2).to(device).unsqueeze(0)
    
    src_pitches = torch.from_numpy(pad_1D([src_pitch, src_pitch2])).float().to(device)
    # print("src_pitches", src_pitches.shape, src_pitches)
    new_pitch = realign_p_e_d(alignments, src_pitches)

    # print("New pitch", new_pitch.shape, new_pitch)