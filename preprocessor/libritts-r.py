import os

def move_files(config):
    in_dir = config["path"]["corpus_path"] # LibriTTS-R
    out_dir = config["path"]["raw_path"]

    # Get list of files of failed speech restoration
    dataset_path = os.path.dirname(in_dir)
    sub_dataset_name = os.path.basename(in_dir)
    with open(os.path.join(dataset_path, "libritts_r_failed_speech_restoration_examples", f"{sub_dataset_name}_bad_sample_list.txt"), "r") as f:
        bad_sample_list = [line.strip() for line in f]

    for bad_sample in bad_sample_list:
        _, _, speaker, chapter, file_name = bad_sample.split("/")
        base_name = file_name[:-4]
        wav_path = os.path.join(os.path.dirname(dataset_path), "LibriTTS", sub_dataset_name, speaker, chapter, file_name)
        out_path1 = os.path.join(in_dir, speaker, chapter, file_name)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        os.rename(wav_path, out_path)


        