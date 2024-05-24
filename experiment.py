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

"""Test TextGrid"""
test_textgrid = False
if test_textgrid:
    import tgt
    import os
    tg_path = os.path.join("preprocessed_data/Bryant", "TextGrid", "Bryant", "{}.TextGrid".format("LJ001-002"))
    tg_path = "preprocessed_data/Bryant/TextGrid/Bryant/LJ001-0002.TextGrid"
    # tg_path = "preprocessed_data/LJSpeech2/TextGrid/LJSpeech/LJ001-0004.TextGrid"

    # Get src time alignments
    textgrid = tgt.io.read_textgrid(tg_path)

    phones_tier = textgrid.get_tier_by_name("phones")
    words_tier = textgrid.get_tier_by_name("words")

    print(words_tier)
    word_end_times = [w.end_time for w in words_tier._objects]
    print(word_end_times)

    sil_phones = ["sil", "sp", "spn"]

    all_phones = []
    word_phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    word_idx = 0
    num_phones = 0
    for t in phones_tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trim leading silences
        if all_phones == [] and word_phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            # For ordinary phones
            word_phones.append(p)
            num_phones += 1
            if word_end_times[word_idx] == e:
                print(p)
                word_idx += 1
                all_phones.append(word_phones)
                word_phones = []

            end_time = e
            end_idx = num_phones
        else:
            # For silent phones
            print("SIlent", p)
            all_phones.append(p)
            num_phones += 1

        durations.append(int(np.round(e * 22050 / 256) - np.round(s * 22050 / 256)))

    # Trim tailing silences
    phones = all_phones[:len(word_end_times)]
    durations = durations[:end_idx]

    print("Phones: ", phones)
    print("Durations: ", durations)

    # To flatten if necessary
    from itertools import chain
    print(list(chain.from_iterable(phones)))

    # return phones, durations, start_time, end_time

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
    print(phones)

    phones_by_word = phones.split()

    def split_with_tie_bar(text):
        result = []
        i = 0
        while i < len(text):
            if i + 2 < len(text) and text[i + 1] == '͡':
                # Group the character with the tie bar and the following character
                result.append(text[i:i + 3])
                i += 3
            else:
                # Just append the single character
                result.append(text[i])
                i += 1
        return result

    phones_by_word = [split_with_tie_bar(word) for word in phones_by_word]

    print(phones_by_word)

"""Go through word_alignment"""
test_word_alignment = True
if test_word_alignment:
    word_alignment = np.array([[0, 0], [1, 1], [2, 2], [3, 2], [3, 3]])
    words = 4
    phoneme_alignment = {i: [] for i in range(words)}

    # pth = "preprocessed_data/Bryant/alignments/phone/Bryant-phone_alignment-LJ001-0002.npy"
    # phone_alignment = np.load(pth, allow_pickle=True)
    # print(phone_alignment)

    pth = "preprocessed_data/Bryant/alignments/phone/Bryant-phone_alignment-LJ001-0002.npy.pkl"
    # Loading the dictionary
    with open(pth, 'rb') as f:
        phone_alignments = pickle.load(f)

    print(phone_alignments)


"""Test Sentence Aligner"""
test_aligner = False
if test_aligner:
    from simalign import SentenceAligner
    # making an instance of our model.
    # You can specify the embedding model and all alignment settings in the constructor.
    myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="m")

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
        mel_path = "preprocessed_data/LJSpeech2/mel/LJSpeech-mel-LJ001-0002.npy"
        text_grid_path = "preprocessed_data/LJSpeech2/TextGrid/LJSpeech/LJ001-0002.TextGrid"

        melspec = torch.from_numpy(np.load(mel_path)).to(device)

    melspec = melspec.unsqueeze(0).unsqueeze(0)   # To get dimension [1,1,H:80,W:462]
    print("Mel shape: ", melspec.size())

    e = model(melspec)
    print("e shape: ", e.size())
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

    h_si = torch.rand([1, 19, 256], device=device)
    print("h_si shape: ", h_si.size())

    # Get Speaker Embedding created by speaker embedding repo
    # style_wav = "/Users/bryantmcarthur/Documents/Ditto/experiment/tony.flac"
    embedding_path = "/Users/bryantmcarthur/Documents/Ditto/SpeakerEncoder/outputs/tony.pkl_emb.pkl"

    with open(embedding_path, 'rb') as f:
        emb_dict = pickle.load(f)

    embedding = torch.from_numpy(emb_dict["default"]).to(device)
    print("Embedding shape: ", embedding.size())

    # embedding = torch.cat((embedding, embedding))
    # print("Embedding shape: ", embedding.size())

    # Adding new dimensions to tensor_2
    embedding = embedding.unsqueeze(0).unsqueeze(0).expand(-1, 19, -1)

    h_sd = h_si + embedding

    print("h_sd shape: ", h_sd.size())

    # Create the model
    model = ProsodyPredictor(256, 1, 4, 8).to(device)
    # Print the model summary
    print(model)

    e = model(h_sd, h_si)

    pi, mu, sigma = e
    print("pi shape: ", pi.size())
    print("mu shape: ", mu.size())
    print("sigma shape: ", sigma.size())