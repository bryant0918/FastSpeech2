import json
import math
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from text import text_to_sequence, lang_to_id
from utils.tools import pad_1D, pad_2D, pad_inhomogeneous_2D, flip_mapping, realign_p_e_d, custom_round


class TrainDataset(Dataset):
    def __init__(self, filename, preprocess_config, train_config, sort=False, drop_last=False):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.src_lang, self.tgt_lang, self.basename, self.speaker, self.text, self.raw_text, self.translation, self.raw_translation = \
            self.process_meta(filename)

        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        raw_translation = self.raw_translation[idx]
        src_lang = lang_to_id[self.src_lang[idx]]
        tgt_lang = lang_to_id[self.tgt_lang[idx]]

        src_phone = np.array(text_to_sequence(self.text[idx], self.cleaners, self.src_lang[idx]))
        tgt_phone = np.array(text_to_sequence(self.translation[idx], self.cleaners, self.tgt_lang[idx])) # TODO: Cleaners here?
        
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        # Prepend zero for language token
        pitch = np.insert(np.load(pitch_path), 0, 0)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        # Prepend zero for language token
        energy = np.insert(np.load(energy_path), 0, 0)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        # Prepend zero for language token
        duration = np.insert(np.load(duration_path), 0, 0)

        # TODO: Get Random Speaker Embedding
        speaker_emb_path = os.path.join(self.preprocessed_path, "speaker_emb", "{}.pkl_emb.pkl".format(speaker))        
        with open(speaker_emb_path, 'rb') as f:
            emb_dict = pickle.load(f)

        embedding = torch.from_numpy(emb_dict["default"])

        alignments_path = os.path.join(self.preprocessed_path, "alignments", "phone",
                                       "{}-phone_alignment-{}.pkl".format(speaker, basename))
        
        with open(alignments_path, 'rb') as f:
            alignments = pickle.load(f)
        # alignments = np.load(alignments_path)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "src_lang": src_lang,
            "text": src_phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "tgt_lang": tgt_lang,
            "translation": tgt_phone,
            "raw_translation": raw_translation,
            "speaker_emb": embedding,
            "alignments": alignments,
        }

        return sample

    def process_meta(self, filename):
        src_lang, tgt_lang, name, speaker, text, raw_text, translation, raw_translation = [], [], [], [], [], [], [], []
        with open(os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8") as f:

            for line in f.readlines():
                sl, tl, n, s, t, r, tr, rtr = line.strip("\n").split("|")
                src_lang.append(sl)
                tgt_lang.append(tl)
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
                translation.append(tr)
                raw_translation.append(rtr)

        return src_lang, tgt_lang, name, speaker, text, raw_text, translation, raw_translation

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        src_langs = [data[idx]["src_lang"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        tgt_langs = [data[idx]["tgt_lang"] for idx in idxs]
        translations = [data[idx]["translation"] for idx in idxs]
        raw_translations = [data[idx]["raw_translation"] for idx in idxs]
        speaker_embeddings = [data[idx]["speaker_emb"] for idx in idxs]
        alignments = [data[idx]["alignments"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        translation_lens = np.array([translation.shape[0] for translation in translations])

        speakers = np.array(speakers)
        src_langs = np.array(src_langs)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        tgt_langs = np.array(tgt_langs)
        translations = pad_1D(translations)
        alignments = pad_inhomogeneous_2D(alignments)

        # Debugging but maybe add reverse_alignments to dataset
        reverse_alignments = flip_mapping(torch.from_numpy(alignments).int(), np.shape(texts)[1])
        if np.shape(alignments)[1] + 1 != np.shape(translations)[1] or reverse_alignments.shape[1] + 1 != np.shape(texts)[1]:
            print(np.shape(alignments)[1], np.shape(translations)[1],reverse_alignments.shape[1], np.shape(texts)[1])
            raise ValueError("Alignments and Texts must have the same length")
        
        alignments = torch.from_numpy(alignments).int()
        pitches = torch.from_numpy(pitches).float()
        energies = torch.from_numpy(energies).float()
        durations = torch.from_numpy(durations).long()
        realigned_p = realign_p_e_d(alignments, pitches)
        realigned_e = realign_p_e_d(alignments, energies)
        realigned_d = realign_p_e_d(alignments, durations)
        realigned_d = custom_round(realigned_d)
            
        return (ids, raw_texts, raw_translations, speakers, src_langs, texts, text_lens, max(text_lens), mels, mel_lens,
                max(mel_lens), tgt_langs, translations, translation_lens, max(translation_lens), speaker_embeddings, alignments, 
                pitches, energies, durations, realigned_p, realigned_e, realigned_d)

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class PreTrainDataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.src_lang, _, self.basename, self.speaker, self.text, self.raw_text, _, _ = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        src_lang = lang_to_id[self.src_lang[idx]]
        
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners, self.src_lang[idx]))
        
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        # Prepend zero for language token
        pitch = np.insert(np.load(pitch_path), 0, 0)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        # Prepend zero for language token
        energy = np.insert(np.load(energy_path), 0, 0)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        # Prepend zero for language token
        duration = np.insert(np.load(duration_path), 0, 0)
        
        # Get Speaker Embedding
        speaker_emb_path_mean = os.path.join(self.preprocessed_path, "speaker_emb", speaker, "{}.pkl".format(speaker))        
        speaker_emb_path_indiv = os.path.join(self.preprocessed_path, "speaker_emb", speaker, "{}.pkl_emb.pkl".format(basename))
        
        with open(speaker_emb_path_mean, 'rb') as f:
            emb_dict = pickle.load(f)
        mean_embedding = torch.from_numpy(emb_dict["mean"])

        with open(speaker_emb_path_indiv, 'rb') as f:
            emb_dict = pickle.load(f)
        indiv_embedding = torch.from_numpy(emb_dict["default"])

        embedding = np.mean([mean_embedding, indiv_embedding], axis=0)
        
        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text_lang": src_lang,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "speaker_emb": embedding,
        }

        return sample

    def process_meta(self, filename):
        src_lang, tgt_lang, name, speaker, text, raw_text, translation, raw_translation = [], [], [], [], [], [], [], []
        with open(os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8") as f:

            for line in f.readlines():
                sl, tl, n, s, t, r, tr, rtr = line.strip("\n").split("|")
                src_lang.append(sl)
                tgt_lang.append(tl)
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
                translation.append(tr)
                raw_translation.append(rtr)

        return src_lang, tgt_lang, name, speaker, text, raw_text, translation, raw_translation


    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        text_langs = [data[idx]["text_lang"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        speaker_embeddings = [data[idx]["speaker_emb"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        text_langs = np.array(text_langs)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return (
            ids,
            raw_texts,
            speakers,
            text_langs,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            speaker_embeddings,
            pitches,
            energies,
            durations,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class SynthDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(filepath)
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = torch.from_numpy(np.load(mel_path)).unsqueeze(0).unsqueeze(0)

        speaker_emb_path = os.path.join(self.preprocess_config["path"]["preprocessed_path"], "speaker_emb",
                                        "{}.pkl_emb.pkl".format(speaker))
        with open(speaker_emb_path, 'rb') as f:
            emb_dict = pickle.load(f)

        embedding = torch.from_numpy(emb_dict["default"]).unsqueeze(0).unsqueeze(0).expand(-1, 19, -1)

        return (basename, speaker_id, phone, raw_text, embedding, mel)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        speaker_embs = [d[4] for d in data]
        mels = [d[5] for d in data]

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens), speaker_embs, mels


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.utils import to_device

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    preprocess_config = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )