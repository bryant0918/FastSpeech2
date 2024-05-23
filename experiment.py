from model import FastSpeech2_Pros
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
test_extractor = True
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
    for i in range(len(starts)):
        phone_emb_chunks.append(e[:, :, start_frame:start_frame + duration[i]])
        start_frame += duration[i]

    print("Phone emb chunks: ", len(phone_emb_chunks), phone_emb_chunks[0].size(), phone_emb_chunks[1].size(),
          phone_emb_chunks[2].size(), phone_emb_chunks[3].size(), phone_emb_chunks[4].size(), phone_emb_chunks[5].size())

    total_len = sum([phone_emb_chunk.size()[2] for phone_emb_chunk in phone_emb_chunks])
    print(total_len)

"""Test Prosody Predictor"""
test_predictor = False
if test_predictor:
    model = FastSpeech2_Pros(preprocess_config, model_config).to(device)

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

# def plot_data(x, y):
#     plt.hist2d(x, y, bins=35)
#     plt.xlim(-8, 8)
#     plt.ylim(-1, 1)
#     plt.axis('off')
#
#
# x, y = gen_data()
# x = torch.Tensor(x)
# y = torch.Tensor(y)
#
# print(x.size())
# print(y.size())
#
# n_iterations = 10
#
# # Create the model
# model = BaseProsodyPredictor(1, 1, 4, 8)
# # Print the model summary
# print(model)
#
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iterations)
#
# for i in range(n_iterations):
#     optimizer.zero_grad()
#     if i == 0:
#         e = model(x)
#     else:
#         e = model(x, e)
#     loss = model.phone_loss(e, y).mean()
#     loss.backward()
#     optimizer.step()
#     scheduler.step()
#
# print("Loss: ", loss.item)
#
# with torch.no_grad():
#     y_hat = model.sample(x)
#
# plt.figure(figsize=(8, 3))
# plt.subplot(1, 2, 1)
# plot_data(x[:, 0].numpy(), y[:, 0].numpy())
# plt.title("Observed data")
# plt.subplot(1, 2, 2)
# plot_data(x[:, 0].numpy(), y_hat[:, 0].numpy())
# plt.title("Sampled data")
# plt.show()