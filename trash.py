"""Build tiny train dataset for testing speed of GPUs."""

import os
import shutil

def build_tiny():
    "Build tiny train dataset from separating val.txt"
    english_dir = 'LJSpeech'
    spanish_dir = 'Spanish_TedX'

    tiny_dir = 'preprocessed_data/tiny'
    if not os.path.isdir(tiny_dir):
        os.makedirs(tiny_dir)

    with open('preprocessed_data/val.txt', 'r') as f:
        val_files = f.readlines()

    for val_file in val_files:
        parts = val_file.strip().split('|')

        lang = parts[0]
        basename = parts[2]
        speaker = parts[3]
        
        # Get file paths
        if lang == 'en':
            alignment_path = os.path.join(english_dir, 'alignments', 'phone', f'{speaker}-phone_alignment-{basename}.pkl')
            duration_path = os.path.join(english_dir, 'duration', f'{speaker}-duration-{basename}.npy')
            energy_path = os.path.join(english_dir, 'energy', f'{speaker}-energy-{basename}.npy')
            mel_path = os.path.join(english_dir, 'mel', f'{speaker}-mel-{basename}.npy')
            pitch_path = os.path.join(english_dir, 'pitch', f'{speaker}-pitch-{basename}.npy')
            speaker_emb_path = os.path.join(english_dir, 'speaker_emb', speaker, f'{basename}.pkl')
            mean_speaker_emb_path = os.path.join(english_dir, 'speaker_emb', speaker, f'{speaker}.pkl')
            textgrid_path = os.path.join(english_dir, 'TextGrid', speaker, f'{basename}.TextGrid')
        
        elif lang == 'es':
            alignment_path = os.path.join(spanish_dir, 'alignments', 'phone', f'{speaker}-phone_alignment-{basename}.pkl')
            duration_path = os.path.join(spanish_dir, 'duration', f'{speaker}-duration-{basename}.npy')
            energy_path = os.path.join(spanish_dir, 'energy', f'{speaker}-energy-{basename}.npy')
            mel_path = os.path.join(spanish_dir, 'mel', f'{speaker}-mel-{basename}.npy')
            pitch_path = os.path.join(spanish_dir, 'pitch', f'{speaker}-pitch-{basename}.npy')
            speaker_emb_path = os.path.join(spanish_dir, 'speaker_emb', speaker, f'{basename}.pkl')
            mean_speaker_emb_path = os.path.join(spanish_dir, 'speaker_emb', speaker, f'{speaker}.pkl')
            textgrid_path = os.path.join(spanish_dir, 'TextGrid', speaker, f'{basename}.TextGrid')

        # Create directories
        if not os.path.isdir(os.path.dirname(os.path.join(tiny_dir, alignment_path))):
            os.makedirs(os.path.dirname(os.path.join(tiny_dir, alignment_path)))
        if not os.path.isdir(os.path.dirname(os.path.join(tiny_dir, duration_path))):
            os.makedirs(os.path.dirname(os.path.join(tiny_dir, duration_path)))
        if not os.path.isdir(os.path.dirname(os.path.join(tiny_dir, energy_path))):
            os.makedirs(os.path.dirname(os.path.join(tiny_dir, energy_path)))
        if not os.path.isdir(os.path.dirname(os.path.join(tiny_dir, mel_path))):
            os.makedirs(os.path.dirname(os.path.join(tiny_dir, mel_path)))
        if not os.path.isdir(os.path.dirname(os.path.join(tiny_dir, pitch_path))):
            os.makedirs(os.path.dirname(os.path.join(tiny_dir, pitch_path)))
        if not os.path.isdir(os.path.dirname(os.path.join(tiny_dir, speaker_emb_path))):
            os.makedirs(os.path.dirname(os.path.join(tiny_dir, speaker_emb_path)))
        if not os.path.isdir(os.path.dirname(os.path.join(tiny_dir, mean_speaker_emb_path))):
            os.makedirs(os.path.dirname(os.path.join(tiny_dir, mean_speaker_emb_path)))
        if not os.path.isdir(os.path.dirname(os.path.join(tiny_dir, textgrid_path))):
            os.makedirs(os.path.dirname(os.path.join(tiny_dir, textgrid_path)))

        # Copy files
        shutil.copy(os.path.join('preprocessed_data', alignment_path), os.path.join(tiny_dir, alignment_path))
        shutil.copy(os.path.join('preprocessed_data', duration_path), os.path.join(tiny_dir, duration_path))
        shutil.copy(os.path.join('preprocessed_data', energy_path), os.path.join(tiny_dir, energy_path))
        shutil.copy(os.path.join('preprocessed_data', mel_path), os.path.join(tiny_dir, mel_path))
        shutil.copy(os.path.join('preprocessed_data', pitch_path), os.path.join(tiny_dir, pitch_path))
        shutil.copy(os.path.join('preprocessed_data', speaker_emb_path), os.path.join(tiny_dir, speaker_emb_path))
        shutil.copy(os.path.join('preprocessed_data', textgrid_path), os.path.join(tiny_dir, textgrid_path))
        if not os.path.isfile(os.path.join(tiny_dir, mean_speaker_emb_path)):
            shutil.copy(os.path.join('preprocessed_data', mean_speaker_emb_path), os.path.join(tiny_dir, mean_speaker_emb_path))

    # Copy speakers.json and stats.json
    shutil.copy(f'preprocessed_data/{english_dir}/speakers.json', os.path.join(tiny_dir, english_dir, 'speakers.json'))
    shutil.copy(f'preprocessed_data/{spanish_dir}/speakers.json', os.path.join(tiny_dir, spanish_dir, 'speakers.json'))
    shutil.copy(f'preprocessed_data/{english_dir}/stats.json', os.path.join(tiny_dir, english_dir, 'stats.json'))
    shutil.copy(f'preprocessed_data/{spanish_dir}/stats.json', os.path.join(tiny_dir, spanish_dir, 'stats.json'))

    # Create new train.txt and val.txt from val_files
    with open(os.path.join(tiny_dir, 'train.txt'), 'w') as f:
        f.writelines(val_files[:int(len(val_files)*0.8)])

    with open(os.path.join(tiny_dir, 'val.txt'), 'w') as f:
        f.writelines(val_files[int(len(val_files)*0.8):])

    print("Done.")


if __name__ == '__main__':
    build_tiny()
    