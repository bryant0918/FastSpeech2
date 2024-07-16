from miipher.dataset.preprocess_for_infer import PreprocessForInfer
from miipher.lightning_module import MiipherLightningModule
from lightning_vocoders.models.hifigan.xvector_lightning_module import HiFiGANXvectorLightningModule
import torch
import torchaudio
import hydra
import os
from tqdm import tqdm

from text.tools import split_with_tie_bar
from itertools import chain
import epitran
from text.cmudict import CMUDict

class MiipherInference:
    def __init__(self, config):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        miipher_path = config['preprocessing']['miipher']['miipher_path']
        self.miipher = MiipherLightningModule.load_from_checkpoint(miipher_path, map_location=self.device)
        self.vocoder = HiFiGANXvectorLightningModule.load_from_checkpoint(config['preprocessing']['miipher']['vocoder_path'], map_location=self.device)
        self.xvector_model = hydra.utils.instantiate(self.vocoder.cfg.data.xvector.model)
        # self.xvector_model = xvector_model.to(self.device)
        
        self.preprocessor = PreprocessForInfer(self.miipher.cfg)
        if torch.cuda.is_available():
            self.preprocessor.cfg.preprocess.text2phone_model.is_cuda=True
        else:
            self.preprocessor.cfg.preprocess.text2phone_model.is_cuda=False

        self.input_dir = config['path']['raw_path']
        self.language = config['preprocessing']['text']['language']
        self.sr = config['preprocessing']['audio']['sampling_rate']

        if self.language == "en":
            self.cmu = CMUDict("lexicon/librispeech-lexicon.txt")
        elif self.language == "es":
            self.epi = epitran.Epitran('spa-Latn')

    @torch.inference_mode()
    def process_audio(self, wav_path, transcript=None, lang_code=None, phones=None):
        wav,sr =torchaudio.load(wav_path)
        wav = wav[0].unsqueeze(0).to(self.device)
        batch = self.preprocessor.process(
            'test',
            (torch.tensor(wav),sr),
            word_segmented_text=transcript,
            lang_code=lang_code,
            phoneme_text=phones
        )

        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor): 
                batch[key] = batch[key].to('cpu')

        (
            phone_feature,
            speaker_feature,
            degraded_ssl_feature,
            _,
        ) = self.miipher.feature_extractor(batch)

        phone_feature = phone_feature.to(self.device)
        speaker_feature = speaker_feature.to(self.device)
        degraded_ssl_feature = degraded_ssl_feature.to(self.device)

        cleaned_ssl_feature, _ = self.miipher(phone_feature,speaker_feature,degraded_ssl_feature)

        vocoder_xvector = self.xvector_model.encode_batch(batch['degraded_wav_16k'].view(1,-1)).squeeze(1)

        cleaned_wav = self.vocoder.generator_forward({"input_feature": cleaned_ssl_feature, "xvector": vocoder_xvector.to(self.device)})[0].T

        torchaudio.save(wav_path, cleaned_wav.view(1,-1).cpu(), sample_rate=self.sr, format='wav')

    # TODO: Change so that we process through miipher before prepare_align.py
    def process_directory(self, speakers):
        for i, speaker in enumerate(tqdm(os.listdir(self.input_dir))):
            if speakers is not None and speaker not in speakers:
                continue
            for file_name in os.listdir(os.path.join(self.input_dir, speaker)):
                if file_name.endswith('.wav'):
                    input_path = os.path.join(self.input_dir, speaker, file_name)

                    base_name = file_name[:-4]
                    with open(os.path.join(self.input_dir, speaker, f"{base_name}.lab"), "r") as f:
                        raw_text = f.readline().strip("\n")
                    
                    # Get Phonetic Transcription
                    if self.language == "es":
                        phones = self.epi.transliterate(raw_text)
                        phones = phones.split()
                        phones = [split_with_tie_bar(word) for word in phones]
                        # Flatten tgt_phones to save to train.txt, val.txt
                        flat_phones = list(chain.from_iterable(phones))
                    
                    elif self.language == "en":
                        phones = []
                        flat_phones = []
                        for word in raw_text.split():
                            try:
                                phones.append(self.cmu.lookup(word)[0])
                                flat_phones.extend(self.cmu.lookup(word)[0].split())
                            except TypeError:
                                print("Word", word, self.cmu.lookup(word), base_name)
                                continue

                    self.process_audio(input_path, phones=' '.join(flat_phones))


# {'phoneme_input_ids': {'input_ids': tensor([[  0,   8,  14,   9,  16,   9,  14,  21,   6,  23,   6,  18,   8,   9,
#           14,   9,  14,  21,  37,  14,  50,  14,  20,  14,  50,  22,   9,   8,
#           23,  14,   5, 141,  14,  14,   5,  22,   5, 158,  14,   9,  21,  14,
#           20, 125,   6,  23, 158,  14,  15,   6,  13,  16,   5, 104,  19,  14,
#            5, 104,  19,   6,  22,  18,   6,   5,   6,  43,  23,   6, 104,  19,
#            6,   9,   2]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1]])}, 'degraded_ssl_input': {'input_values': tensor([[-0.0210, -0.0197, -0.0188,  ...,  0.0086,  0.0212,  0.0190]]), 
#                                        'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]], dtype=torch.int32)}, 
#                                        'degraded_wav_16k': tensor([[-0.0020, -0.0019, -0.0019,  ...,  0.0003,  0.0012,  0.0011]]), 
#                                        'degraded_wav_16k_lengths': tensor([146252])}