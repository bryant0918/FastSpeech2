from TTS.utils.synthesizer import Synthesizer
import torch
from TTS.utils.manage import ModelManager
import os
import shutil
from tqdm import tqdm

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# load model manager
models_path = "/home/ditto/miniconda3/envs/augmenter/lib/python3.9/site-packages/TTS/.models.json"
manager = ModelManager(models_path)
# manager.list_models()

""" No Need to do prepare_align afterward. Next: Miipher, MFA, Preprocess, Speaker Embeddings"""

def augment(text, out_path):
    # model_name = 'tts_models/en/vctk/fast_pitch'
    # model_name = 'tts_models/en/ljspeech/tacotron2-DDC'
    model_name = 'tts_models/en/ek1/tacotron2'
    # model_name = 'tts_models/en/sam/tacotron-DDC'
    # model_name = 'tts_models/es/mai/tacotron2-DDC'
    model_path, config_path, model_item = manager.download_model(model_name)

    vocoder_name = model_item["default_vocoder"]
    if vocoder_name is None:
        vocoder_name = "vocoder_models/en/vctk/hifigan_v2"
    vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)

    encoder_path = '/home/ditto/Ditto/SpeakerEncoder/pretrained_model/Muellers/best_model.pth.tar'
    encoder_config_path = '/home/ditto/Ditto/SpeakerEncoder/pretrained_model/Muellers/config.json'

    # List available 🐸TTS models
    synth = Synthesizer(model_path, config_path, tts_speakers_file=None, vocoder_checkpoint=vocoder_path,
                        vocoder_config=vocoder_config_path, encoder_checkpoint=encoder_path, encoder_config=encoder_config_path, use_cuda=True)

    speaker_wav = 'raw_data/LibriTTS/14/14_208_000001_000000.wav'

    wav = synth.tts(text, speaker_wav=speaker_wav)

    synth.save_wav(wav, out_path)


def data_augmenter(preprocessed_dir, raw_dir):
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    
    with open(os.path.join(preprocessed_dir, 'train.txt'), 'r') as train_file, open(os.path.join(preprocessed_dir, 'val.txt'), 'r') as val_file:
        lines = train_file.readlines() + val_file.readlines()

    for i, line in tqdm(enumerate(lines)):
        # Get 25% of wav files
        if i % 4 != 0:
            continue

        _, _, basename, speaker, _, src_text, _, tgt_text = line.strip().split('|')
        
        aug_basename = basename + '_a'
        aug_speaker = 'augmented'
        out_path = os.path.join(raw_dir, aug_basename + '.wav')

        if not os.path.exists(out_path):
            augment(src_text, out_path)

        out_src = os.path.join(raw_dir, aug_basename + '.lab')
        out_tgt = os.path.join(raw_dir, aug_basename + '_tgt.lab')

        with open(out_src, 'w') as src_file, open(out_tgt, 'w') as tgt_file:
            src_file.write(src_text)
            tgt_file.write(tgt_text)        
        
        og_alignment_path = os.path.join(preprocessed_dir, "alignments", "word", "{}-word_alignment-{}.npy".format(speaker, basename))
        new_alignment_path = os.path.join(preprocessed_dir, "alignments", "word", "{}-word_alignment-{}.npy".format(aug_speaker, aug_basename))
        shutil.copy(og_alignment_path, new_alignment_path)

    print("Done.")
        


if __name__ == "__main__":
    preprocessed_dir = 'preprocessed_data/LJSpeech'
    raw_dir = 'raw_data/LJSpeech/augmented'
    data_augmenter(preprocessed_dir, raw_dir)
