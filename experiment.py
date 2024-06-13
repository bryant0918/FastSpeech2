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
    tg_path = os.path.join("preprocessed_data/Bryant", "TextGrid", "Bryant", "{}.TextGrid".format("LJ001-002"))
    tg_path = "preprocessed_data/Bryant/TextGrid/Bryant/LJ001-0002.TextGrid"
    tg_path = "preprocessed_data/LJSpeech/TextGrid/LJSpeech/LJ032-0007.TextGrid"
    tg_path = "preprocessed_data/LJSpeech/TextGrid/LJSpeech/LJ036-0179.TextGrid"
    tg_path = "preprocessed_data/LJSpeech/TextGrid/LJSpeech/LJ030-0041.TextGrid"

    # Get src time alignments
    textgrid = tgt.io.read_textgrid(tg_path)

    # phones_tier = textgrid.get_tier_by_name("phones")
    # words_tier = textgrid.get_tier_by_name("words")
    # word_end_times = [w.end_time for w in words_tier._objects]

    # sil_phones = ["sil", "sp"]   # Not 'spn'

    # all_phones, word_phones, durations = [], [], []
    # start_time, end_time = 0, 0
    # end_idx, word_idx = 0, 0
    # num_phones, num_words = 0, 0

    # for t in phones_tier._objects:
    #     s, e, p = t.start_time, t.end_time, t.text
    #     print(p,s,e)
    #     print("Word_idx", word_idx, words_tier.intervals[word_idx].text, word_end_times[word_idx], words_tier.intervals[word_idx].end_time)

    #     # Trim leading silences
    #     if all_phones == [] and word_phones == []:
    #         if p in sil_phones:
    #             print("Continuing")
    #             continue
    #         else:
    #             start_time = s

    #     if p not in sil_phones:
    #         if p == "spn" and words_tier.intervals[word_idx].text == "<unk>":
    #             # For spoken noise
    #             word_phones.append(p)
    #             num_phones += 1
    #             if not isinstance(all_phones[-1], list):
    #                 if word_end_times[word_idx] == e:
    #                     all_phones[-1] = word_phones
    #                     word_phones = []
    #                     end_time = e
    #                     end_idx = num_phones

    #                     if word_idx == len(words_tier.intervals) - 1:  # That was the last word
    #                         break
    #                     word_idx += 1
    #             else:
    #                 if word_end_times[word_idx] == e:
    #                     all_phones.append(word_phones)
    #                     word_phones = []
    #                     end_time = e
    #                     end_idx = num_phones
    #                     num_words += 1

    #                     if word_idx == len(words_tier.intervals) - 1:  # That was the last word
    #                         break
    #                     word_idx += 1


    #         elif p == "spn" and words_tier.intervals[word_idx].text != "<unk>":
    #             if not isinstance(all_phones[-1], list):
    #                 all_phones[-1] = p
    #             else:
    #                 all_phones.append(p)
    #             num_phones += 1
    #             num_words += 1

    #         # For ordinary phones
    #         else:
    #             word_phones.append(p)
    #             num_phones += 1

    #             if word_end_times[word_idx] == e:
    #                 all_phones.append(word_phones)
    #                 word_phones = []
    #                 end_time = e
    #                 end_idx = num_phones
    #                 num_words += 1

    #                 if word_idx == len(words_tier.intervals)-1:  # That was the last word
    #                     break

    #                 word_idx += 1


    #     else:  # For silent phones
    #         all_phones.append(p)
    #         num_phones += 1
    #         num_words += 1

    #         # ending number of words will not be the same as word index since words_tier excludes silent phones
    #         # but we will keep them. I am getting word alignment assuming praat gets all the words. However,
    #         # there are <unk> words so I need to append ['spn'] when there is an <unk> word for my alignment.
    #         # Maybe I can check my lexicon for a pronounciation as well. I should also account for this in the duration,
    #         # pitch and energy as well.

    #     # durations.append(int(np.round(e * self.sampling_rate / self.hop_length) -
    #     #                      np.round(s * self.sampling_rate / self.hop_length)))
    #     durations.append(int(np.round(e * 22050 / 256) - np.round(s * 22050 / 256)))

    # print("Phones: ", all_phones)
    # print(len(all_phones), word_idx, num_words)
    # # Trim tailing silences
    # phones = all_phones[:num_words]
    # durations = durations[:end_idx]

    # print("Phones: ", phones)

    # # return phones, durations, start_time, end_time

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
        print(p, s, e)
        print("Word_idx", word_idx, words_tier.intervals[word_idx].text, word_end_times[word_idx], words_tier.intervals[word_idx].end_time)

        # Trim leading silences
        if not all_phones and not word_phones:
            if p in sil_phones:
                print("Continuing")
                continue
            else:
                start_time = s

        if p not in sil_phones:
            if p == "spn" and words_tier.intervals[word_idx].text == "<unk>":
                word_phones.append(p)
                num_phones += 1
                if not isinstance(all_phones[-1], list) if all_phones else False:
                    if word_end_times[word_idx] == e:
                        all_phones[-1] = word_phones
                        word_phones = []
                        end_time = e
                        end_idx = num_phones

                        if word_idx == len(words_tier.intervals) - 1:  # That was the last word
                            break
                        word_idx += 1
                else:
                    if word_end_times[word_idx] == e:
                        all_phones.append(word_phones)
                        word_phones = []
                        end_time = e
                        end_idx = num_phones
                        num_words += 1

                        if word_idx == len(words_tier.intervals) - 1:  # That was the last word
                            break
                        word_idx += 1

            elif p == "spn" and words_tier.intervals[word_idx].text != "<unk>":
                if not isinstance(all_phones[-1], list) if all_phones else False:
                    all_phones[-1] = p
                else:
                    all_phones.append(p)
                num_phones += 1
                num_words += 1

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

        durations.append(int(np.round(e * 22050 / 256) - np.round(s * 22050 / 256)))

    print("Phones: ", all_phones)
    print(len(all_phones), word_idx, num_words)

    # Trim tailing silences
    phones = all_phones[:num_words]
    durations = durations[:end_idx]

    flat_phones = list(chain.from_iterable(phones))
    print("flat_phones: ", len(flat_phones), flat_phones)
    print("Durations: ", len(durations))

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
test_word_alignment = False
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

    src = "motorcycles. -- dallas police motorcycles preceded the pilot car."
    tgt = "motocicletas. -- las motocicletas de la policia de dallas precedieron al coche del piloto."

    alignments = myaligner.get_word_aligns(src.split(), tgt.split())
    print("Src.split(): ", src.split())
    print("Tgt.split(): ", tgt.split())
    print(alignments)

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
