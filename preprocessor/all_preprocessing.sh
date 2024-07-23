
# Specifically for Spanish Dataset Right Now
raw_dir="raw_data/Spanish_new"

# Get the textgrid files
mamba activate aligner

mfa validate $raw_dir spanish_mfa spanish_mfa

oov_file="/home/ditto/Documents/MFA/Spanish_new/split3/log/normalize_oov.log"
dict="/home/ditto/Documents/MFA/pretrained_models/dictionary/spanish_mfa.dict"
oov_words="/home/ditto/Documents/MFA/Spanish_new/oov_words.txt"

python3 mfa.py 

# Preprocess
config_file="config/LibriTTS/preprocess_es.yaml"
python3 nohup python3 preprocess.py $config_file > preprocess.log 2>&1 &

# Get Speaker Embeddings
nohup python3 SpeakerEncoder/compute_embeddings.py --output_path preprocessed_data/Spanish_new/speaker_emb --input_path $raw_dir > embeddings.log 2>&1 &

# Combine Datasets
python3 combine_datasets.py --dir1 preprocessed_data/LibriTTS --dir2 preprocessed_data/Spanish_new
