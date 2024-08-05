# FastSpeech 2 - PyTorch Implementation

This is a PyTorch implementation of Microsoft's text-to-speech system [**FastSpeech 2: Fast and High-Quality End-to-End Text to Speech**](https://arxiv.org/abs/2006.04558v1).
This project is based on [xcmyz&#39;s implementation](https://github.com/xcmyz/FastSpeech) of FastSpeech. Feel free to use/modify the code.

There are several versions of FastSpeech 2.
This implementation is more similar to [version 1](https://arxiv.org/abs/2006.04558v1), which uses F0 values as the pitch features.
On the other hand, pitch spectrograms extracted by continuous wavelet transform are used as the pitch features in the [later versions](https://arxiv.org/abs/2006.04558).

![](./img/model.png)

# Updates

- 2021/7/8: Release the checkpoint and audio samples of a multi-speaker English TTS model trained on LibriTTS
- 2021/2/26: Support English and Mandarin TTS
- 2021/2/26: Support multi-speaker TTS (AISHELL-3 and LibriTTS)
- 2021/2/26: Support MelGAN and HiFi-GAN vocoder

# Audio Samples

Audio samples generated by this implementation can be found [here](https://ming024.github.io/FastSpeech2/).

# Quickstart

## Dependencies

You can install the Python dependencies with

```
pip3 install -r requirements.txt
```

then

```commandline
conda install -c conda-forge pyworld montreal-forced-aligner
conda instal conda-forge::charset-normalizer
```

Then you must unzip the vocoders

```
unzip hifigan/generator_LJSpeech.pth.tar.zip -d hifigan/
unzip hifigan/generator_universal.pth.tar.zip -d hifigan/
```

## Inference

You have to download the [pretrained models](https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F?usp=sharing) and put them in ``output/ckpt/LJSpeech/``,  ``output/ckpt/AISHELL3``, or ``output/ckpt/LibriTTS/``.

For English single-speaker TTS, run

```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

For Mandarin multi-speaker TTS, try

```
python3 synthesize.py --text "大家好" --speaker_id SPEAKER_ID --restore_step 600000 --mode single -p config/AISHELL3/preprocess.yaml -m config/AISHELL3/model.yaml -t config/AISHELL3/train.yaml
```

For English multi-speaker TTS, run

```
python3 synthesize.py --text "YOUR_DESIRED_TEXT"  --speaker_id SPEAKER_ID --restore_step 800000 --mode single -p config/LibriTTS/preprocess.yaml -m config/LibriTTS/model.yaml -t config/LibriTTS/train.yaml
```

The generated utterances will be put in ``output/result/``.

To copy from remote server back to local machine:

```commandline
scp "ditto@136.36.160.77:/home/ditto/Documents/ProsodyCloning/FastSpeech2/output/result/LJSpeech/<name of file>" /Users/bryantmcarthur/Downloads/
scp "ditto@Emotiv:/home/ditto/Datasets/Speech/es/tedx_spanish_corpus/files/Speaker_Info.xls" /Users/bryant/Documents/Ditto/
```

Here is an example of synthesized mel-spectrogram of the sentence "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition", with the English single-speaker TTS model.
![](./img/synthesized_melspectrogram.png)

## Batch Inference

Batch inference is also supported, try

```
python3 synthesize.py --source preprocessed_data/LJSpeech/val.txt --restore_step 900000 --mode batch -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

to synthesize all utterances in ``preprocessed_data/LJSpeech/val.txt``

## Controllability

The pitch/volume/speaking rate of the synthesized utterances can be controlled by specifying the desired pitch/energy/duration ratios.
For example, one can increase the speaking rate by 20 % and decrease the volume by 20 % by

```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml --duration_control 0.8 --energy_control 0.8
```

```
python3 synthesize.py --text "Hi my name is Ditto and this is my voice" --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml --duration_control .8 --energy_control .8
```

# Set up and Connect to Instance

## Create Filesystem

You must first create a filesystem in the region that contains the GPU instances you will use. Create the filesystem and then launch in instance.

Once it is booted up and running you can copy your preprocessed data to the filesystem using tar and scp:

```
tar -czvf tiny.tar.gz preprocessed_data/tiny

scp -i ~/.ssh/lambda-labs.pem tiny.tar.gz ubuntu@192.222.52.43:/home/ubuntu/emotiv-data-NTX

tar -xzvf tiny.tar.gz
```

```
scp -i path/to/key.pem -r path/to/local/folder ubuntu@hostIP:path/to/remote/folder

rsync -az -e "ssh -i ~/.ssh/lambda-labs.pem" --ignore-existing preprocessed_data/ ubuntu@192.222.52.43:/home/ubuntu/emotiv-data-NTX/preprocessed_data
```

also transfer the pull_docker.sh file.

```
scp -i ~/.ssh/lambda-labs.pem pull_docker.sh ubuntu@192.222.52.43:/home/ubuntu/emotiv-data-NTX/pull_docker.sh
```

Next ssh into the server `ssh -i ~/.ssh/lambda-labs.pem ubuntu@IPaddress`

Configure AWS with `aws configure` have key and secret key handy.

Then install jq with `sudo apt-get install jq`.

Then you can pull the docker .tar file onto the filesystem.

```
sudo bash emotiv-data-NTX/pull_docker.sh
```

remember to move it to the filesystem `mv fastspeech2_1-0-0.tar emotiv-data-NTX/`

Now when you stop the instance and restart it the data and dockerized image will already be available.

## Set up Repo

Make sure you are uploading your most recent versions to docker.

```
docker build -t fastspeech2 .
bash docker_upload.sh
```

ssh into the server: `ssh -i ~/.ssh/lambda-labs.pem ubuntu@IPaddress`

If you need to pull down a more recent docker version:

Have access key and secret key ready.
```
cd emotiv-data-NTX
sudo aws configure
sudo apt-get install jq
sudo bash pull_docker.sh
```

If the docker image already on the filesystem is good then just load it:

```
sudo docker load -i fastspeech2_1-0-0.tar
```

Now you should be ready to train. Make sure config files are up-to-date with new paths to preprocessed_data on filesystem.

```
sudo docker run -it -p 6006:6006 --gpus all --entrypoint /bin/bash -v /home/ubuntu/emotiv-data-NTX:/emotiv-data-NTX fastspeech2
```

```
sudo docker run -it -p 12355:12355 --gpus all --entrypoint /bin/bash -v /home/ubuntu/emotiv-data-NTX:/emotiv-data-NTX fastspeech2

sudo docker run -it -p 12355:12355 --gpus all --entrypoint /bin/bash -v /home/ubuntu/preprocessed_data:/preprocessed_data fastspeech2
```

Don't forget to `conda activate Emotiv` and you should be good to go.

# Training

## Datasets

The supported datasets are

- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/): a single-speaker English dataset consists of 13100 short audio clips of a female speaker reading passages from 7 non-fiction books, approximately 24 hours in total.

``wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2``

- Spanish: Ted Talk spanish audio clips with multiple different speakers
  and transcripts, approximately 24 hours in total.
- [AISHELL-3](http://www.aishelltech.com/aishell_3): a Mandarin TTS dataset with 218 male and female speakers, roughly 85 hours in total.
- [LibriTTS](https://research.google/tools/datasets/libri-tts/): a multi-speaker English dataset containing 585 hours of speech by 2456 speakers.

``wget https://www.openslr.org/resources/141/train_clean_360.tar.gz``

We take LJSpeech as an example hereafter.

## Preprocessing

### Prepare Align

First, we need to take the metadata csv files that contain the fileID, transcription as well as the
.wav file of the audio and translate the text to our tgt language to train on (if translation doesn't already exist) and
resave the transcription and translation as a .lab file and copy over the .wav file in order to create the TextGrid files.

We also perform a word alignment on the transcription and translation and save it separately to be used later.

run

```
python3 prepare_align.py config/LJSpeech/preprocess.yaml
```

```
nohup python3 prepare_align.py config/LJSpeech/preprocess.yaml &
ps -ef | grep 2014664
nohup python3 prepare_align.py config/LibriTTS/preprocess_es.yaml > align.log 2>&1 &
2886708
```

### Clean Audio

Use Miipher model to create high quality audio to train on. If not done already adjust so that can be run in tandem with prepare_align.py.

Use `conda activate augmenter`.

```
nohup python3 clean_audio.py config/LJSpeech/preprocess_es.yaml > clean_audio.log 2>&1 &
nohup python3 clean_audio.py config/LibriTTS/preprocess_es.yaml > clean_audio.log 2>&1 &
```

### Get the TextGrid Files

#### <u>For Datasets with existing TextGrid files </u>

As described in the paper, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) is used to obtain the alignments between the utterances and the phoneme sequences.
Alignments of the supported datasets are provided [here](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4?usp=sharing).

```
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1ukb8o-SnqhXCxq7drI3zye3tZdrGvQDA' -O preprocessed_data/LJSpeech/LJSpeech.zip
```

or copy the file to the server by

```
scp /Users/bryantmcarthur/Downloads/LJSpeech.zip ditto@136.36.160.77:/home/ditto/Documents/ProsodyCloning/FastSpeech2/preprocessed_data/LJSpeech
scp /Users/bryant/Downloads/LibriTTS.zip ditto@Emotiv:/home/ditto/Ditto/FastSpeech2/preprocessed_data/LibriTTS
```

You have to unzip the files in ``preprocessed_data/LJSpeech/TextGrid/``.

``unzip preprocessed_data/LJSpeech/LJSpeech.zip -d preprocessed_data/LJSpeech/``

#### <u>For New Datasets without Existing TextGrid Files </u>

For new Datasets you will need to generate TextGrid files yourself. Instructions are [here](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html) for installation and [here](https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/index.html#first-steps-align-pretrained) for aligning.

Create new conda (mamba) environment for all mfa stuff

```
conda activate base
conda install -c conda-forge mamba
mamba create -n aligner -c conda-forge montreal-forced-aligner
mamba activate aligner
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install speechbrain
```

Next download the models you will need.
I will use english_us_arpa in the ReadMe but find whichever language models you need on the website.

```
mfa model download acoustic <spanish_mfa>
mfa model download dictionary <english_us_arpa>
mfa model download g2p <english_us_arpa>
```

You should be able to run `mfa model inspect acoustic <english_us_arpa>` and it will output information about the <english_us_arpa> acoustic model. Same goes for dictionary and g2p models.

We have one more step to make sure our dataset directory is properly setup to run MFA:
run `create_mfa_directory(raw_dir)` in `mfa.py`

Now validate your dataset by (note: <english_us_arpa> here is referring to acoustic and dictionary models):

create directory name in MFA folder to ensure oov file writes. `mkdir -p /home/ditto/Documents/MFA/<dir_name>`
```
mfa validate raw_dir <english_us_arpa> <english_us_arpa>
```

Next you will want to get all the out of vocabulary words from the dataset and use the G2P model to add the pronounciation to the dictionary model.

IMPORTANT: Make sure to grab the normalize_oov.log from "/home/ditto/Documents/MFA/<corpus>/split3/log/normalize_oov.log" DURING mfa validate. Generating that file during "Normalizing text..." is one of the first steps. That <corpus> directory is a temp directory and will be deleted after mfa validate is complete. So copy the file somewhere else to use for merging your dictionaries. `cp /home/ditto/Documents/MFA/<corpus>/split3/log/normalize_oov.log /home/ditto/Documents/MFA/<corpus>_normalize_oov.log`

As soon as you have that file copied over you may proceed and you do not need to wait for the validate to finish.

From `mfa.py` first run `get_oov_words(oov_log_file, dict, oov_words)` to get the out of vocabulary words in a list. Next use the G2P model to generate a dictionary for those words:

```
mfa g2p path/to/oov_words.txt <english_us_arpa> path/to/generated_dictionary.txt
```

Then from `mfa.py` run `merge_dictionaries(pretrained_dict_path, generated_dict_path)`

Now you are ready to revalidate and align the corpus. First remove the MFA corpus directory in MFA: `rm -r '/home/ditto/Documents/MFA/<corpus>`. \<corpus\> here will be the folder name of raw_dir. Then validate and align:

```
mfa validate raw_dir <english_us_arpa> <english_us_arpa> -j 21 --use_mp --no_final_clean --overwrite

mfa align raw_dir <english_us_arpa> <english_us_arpa> preprocessed_data_dir/TextGrid -j 16 --use_mp --overwrite --clean
```

You should now see TextGrid files in preprocessed_data_dir

### Preprocessing Script

After that, run the preprocessing script by

```
python3 preprocess.py config/LJSpeech/preprocess.yaml
```

```
nohup python3 preprocess.py config/LJSpeech/preprocess.yaml &

nohup python3 preprocess.py config/LibriTTS/preprocess_es.yaml > preprocess.log 2>&1 &
14496
```

Word quechua None raw_data/Spanish_new/22386e8db9e5ecb8f12dedf9581eec968bcc0fb0adee382a793fe37f37e05ee64109d0c4b893d18b57694301f0424161d3415ad0cbe62d7bd56ba583a8ef23c6/common_voice_es_34372698_tgt.lab

### Get Speaker Embeddings

This can be done in parallel with preprocessing script since it just reads audio files and computes embeddings. That's why it is split to two different scripts to speed up processing.

```
python SpeakerEncoder/compute_embeddings.py --output_path preprocessed_data/LJSpeech/speaker_emb --input_path raw_data/LJSpeech
```

```
nohup python SpeakerEncoder/compute_embeddings.py --output_path preprocessed_data/Spanish_new/speaker_emb --input_path raw_data/Spanish_new > embeddings.log 2>&1 &
11855
```

### Combine two datasets to train on

Change dataset paths in .py file.

```
python combine_datasets.py
```

## PreTrain

Pretrain is especially for the Prosody Extractor and Predictor, but will also update synthesizer weights (Encoder, Adapter, Decoder).

```
python3 pretrain.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/pretrain.yaml
```

```
nohup python3 pretrain.py -p config/LJSpeech/preprocess.yaml -p2 config/LJSpeech/preprocess_es.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/pretrain.yaml -w 16 &

4091228
```

```
nohup python3 pretrain.py --restore_step 281751 -p config/LJSpeech/preprocess.yaml -p2 config/LJSpeech/preprocess_es.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/pretrain.yaml -w 16 &

861748
```

```
python3 pretrain.py -p config/Tiny/preprocess.yaml -p2 config/Tiny/preprocess_es.yaml -m config/Tiny/model.yaml -t config/Tiny/pretrain.yaml -w 16

python3 pretrain_ddp.py -p config/Tiny/LL_preprocess.yaml -p2 config/Tiny/LL_preprocess_es.yaml -m config/Tiny/LL_model.yaml -t config/Tiny/LL_pretrain.yaml -w 16 --ip 192.222.52.202

python3 pretrain.py -p config/Tiny/LL_preprocess.yaml -p2 config/Tiny/LL_preprocess_es.yaml -m config/Tiny/LL_model.yaml -t config/Tiny/LL_pretrain.yaml -w 24
```

### On GPU

```
nohup python3 pretrain.py -p config/LJSpeech/LL_preprocess.yaml -p2 config/LJSpeech/LL_preprocess_es.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/LL_pretrain.yaml -w 104 &

19
```

## Train Synthesizer

Train your model with

```
nohup python3 train.py --from_pretrained_ckpt 175000 -p config/LJSpeech/preprocess.yaml -p2 config/LJSpeech/preprocess_es.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml &

nohup python3 train.py --restore_step 30000 -p config/LJSpeech/preprocess.yaml -p2 config/LJSpeech/preprocess_es.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml &

1562882
```

```
python3 train.py --from_pretrained_ckpt 175000 -p config/LJSpeech/preprocess.yaml -p2 config/LJSpeech/preprocess_es.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

```
python3 train.py -p config/Tiny/preprocess.yaml -p2 config/Tiny/preprocess_es.yaml -m config/Tiny/model.yaml -t config/Tiny/train.yaml -w 16

python3 train_ddp.py -p config/Tiny/LL_preprocess.yaml -p2 config/Tiny/LL_preprocess_es.yaml -m config/Tiny/LL_model.yaml -t config/Tiny/LL_train.yaml -w 16
```

The model takes less than 10k steps (less than 1 hour on my GTX1080Ti GPU) of training to generate audio samples with acceptable quality, which is much more efficient than the autoregressive models such as Tacotron2.

# TensorBoard

To transfer to local port when you ssh

```
ssh -L 6006:127.0.0.1:6006 ditto@136.36.160.77
```

Use to view locally

```
tensorboard --logdir output/log/LJSpeech
```

or to transfer from the server

```commandline
tensorboard --logdir=output/log/LJSpeech/pretrain --bind_all
```

```commandline
tensorboard --logdir=/emotiv-data-NTX/output/log/LJSpeech --bind_all

```

Then on local machine go to http://127.0.0.1:6006.

The loss curves, synthesized mel-spectrograms, and audios are shown.

![](./img/tensorboard_loss.png)
![](./img/tensorboard_spec.png)
![](./img/tensorboard_audio.png)

# Implementation Issues

If you get a `segmentation fault` then in your command line first `export LD_LIBRARY_PATH=""`

- Following [xcmyz&#39;s implementation](https://github.com/xcmyz/FastSpeech), I use an additional Tacotron-2-styled Post-Net after the decoder, which is not used in the original FastSpeech 2.
- Gradient clipping is used in the training.
- In my experience, using phoneme-level pitch and energy prediction instead of frame-level prediction results in much better prosody, and normalizing the pitch and energy features also helps. Please refer to ``config/README.md`` for more details.

Please inform me if you find any mistakes in this repo, or any useful tips to train the FastSpeech 2 model.

# References

- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
- [xcmyz&#39;s FastSpeech implementation](https://github.com/xcmyz/FastSpeech)
- [TensorSpeech&#39;s FastSpeech 2 implementation](https://github.com/TensorSpeech/TensorflowTTS)
- [rishikksh20&#39;s FastSpeech 2 implementation](https://github.com/rishikksh20/FastSpeech2)

# Citation

```
@INPROCEEDINGS{chien2021investigating,
  author={Chien, Chung-Ming and Lin, Jheng-Hao and Huang, Chien-yu and Hsu, Po-chun and Lee, Hung-yi},
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Investigating on Incorporating Pretrained and Learnable Speaker Representations for Multi-Speaker Multi-Style Text-to-Speech}, 
  year={2021},
  volume={},
  number={},
  pages={8588-8592},
  doi={10.1109/ICASSP39728.2021.9413880}}
```
