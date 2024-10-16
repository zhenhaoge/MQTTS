# infer a singe pair of (reference audio, text) using MQTTS
#
# reference script: infer.py (batch inference)
#
# Zhenhao Ge, 2024-10-16

import librosa
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import tester
import argparse
from dp.phonemizer import Phonemizer
import soundfile as sf
import pyloudnorm as pyln
import os
from pathlib import Path
import json
import numpy as np
from numpy.linalg import norm
from librosa.util import normalize
from collections import Counter
from shutil import copyfile
import json
import time
import subprocess
import re
from pathlib import Path

home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'mqtts')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

def get_audio(speaker_path, meter, sample_rate=16000):

    audio, sr = sf.read(speaker_path) # load audio (shape: samples, channels)
    # assert sr == sample_rate, 'sampling rate is {} (should be {})'.format(sr, sample_rate)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    loudness = meter.integrated_loudness(audio) # measure loudness
    audio = pyln.normalize.loudness(audio, loudness, -20.0)
    return audio

def cos_sim(A,B):
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    return cosine

def get_hostname():
    hostname = subprocess.check_output('hostname').decode('ascii').rstrip()
    return hostname

def get_gpu_info(device):
    line_as_bytes = subprocess.check_output("nvidia-smi -L", shell=True)
    lines = line_as_bytes.decode("ascii").split('\n')
    lines = [line for line in lines if line != '']
    line = lines[device]
    string = re.sub("\(.*?\)","()", line).replace('()','').strip()
    return string

def parse_args():
    usage = 'usage: online inference using MQTTS'
    parser = argparse.ArgumentParser(description=usage)

    # Path
    parser.add_argument('---phonemizer_dict_path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--speaker-path', type=str, required=True)
    parser.add_argument('--sentence-path', type=str, required=True)
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--spkr-embedding-path', type=str, default=None)

    # Data
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-samples', type=int, default=-1)

    #Sampling
    parser.add_argument('--use-repetition-gating', action='store_true')
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--sampling-temperature', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=-1)
    parser.add_argument('--min-top-k', type=int, default=1)
    parser.add_argument('--top-p', type=float, default=0.7)
    parser.add_argument('--length-penalty-max-length', type=int, default=50)
    parser.add_argument('--length-penalty-max-prob', type=float, default=0.8)
    parser.add_argument('--max-output-length', type=int, default=100000)
    parser.add_argument('--phone-context-window', type=int, default=4)

    #Speech Prior
    parser.add_argument('--noise-seed', default=-1, help='-1 means no seed, random noise')
    parser.add_argument('--clean-speech-prior', action='store_true')
    parser.add_argument('--prior-noise-level', type=float, default=1e-5)
    parser.add_argument('--prior-frame', type=int, default=3)

    #Other
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--nreps', type=int, default=1)

    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # interactive mode
    args = argparse.ArgumentParser()

    data_path = os.path.join(os.getcwd(), 'datasets', 'GigaSpeech-Zhenhao')
    assert os.path.isdir(data_path), f'data path {data_path} does not exist!'
    speaker_rel_path = 'segment/youtube/P0000/YOU1000000038/YOU1000000038_S0000079.wav'
    args.speaker_path = os.path.join(data_path, speaker_rel_path)
    assert os.path.isfile(args.speaker_path), f'speaker path {args.speaker_path} does not exist!'

    sentence_rel_path = speaker_rel_path.replace('.wav', '.txt')
    args.sentence_path = os.path.join(data_path, sentence_rel_path)
    assert os.path.isfile(args.sentence_path), f'sentence path {args.sentence_path} does not exist!'

    args.config_path = os.path.join(os.getcwd(), 'ckpt', 'OTS', 'config.json')
    args.model_path = os.path.join(os.getcwd(), 'ckpt', 'OTS', 'transformer.ckpt')
    args.output_dir = os.path.join(os.getcwd(), 'outputs', 'infer_samples_2')
    args.noise_seed = 0

    args.sample_rate = 16000
    args.phonemizer_dict_path = os.path.join(work_path, 'en_us_cmudict_forward.pt')
    args.batch_size = 2
    args.num_samples = 16
    args.top_p = 0.8
    args.min_top_k = 2
    args.max_output_length = 100000
    args.phone_context_window = 3
    args.clean_speech_prior = True
    args.spkr_embedding_path = ''
    args.prior_noise_level = 1e-5
    args.prior_frame = 3
    args.device = 0
    args.nreps = 3

    # print out data information
    print('speaker path: {}'.format(args.speaker_path))
    print('sentence path: {}'.format(args.sentence_path))
    print('nreps: {}'.format(args.nreps))

    # print out input configuration
    print('phonemizer dict path: {}'.format(args.phonemizer_dict_path))
    print('model path: {}'.format(args.model_path))
    print('config path: {}'.format(args.config_path))
    print('input path: {}'.format(args.input_path))
    print('output dir: {}'.format(args.outputdir))
    print('batch size: {}'.format(args.batch_size))
    print('num of samples: {}'.format(args.num_samples))
    print('CUDA device: {}'.format(args.device))
    print('clean speech prior: {}'.format(args.clean_speech_prior))

    # add phoneset to args
    args.phoneset = [l.rstrip() for l in open('phoneset.txt', 'r').readlines()]

    # read transformer config
    with open(args.config_path, 'r') as f:
        argdict = json.load(f)
        assert argdict['sample_rate'] == args.sample_rate, \
            'Sampling rate not consistent, stated {}, but the model is trained on {}'.format(
            args.sample_rate, argdict['sample_rate'])
        argdict.update(args.__dict__)
        args.__dict__ = argdict

    hostname = get_hostname()
    gpu_info = get_gpu_info(args.device)
    print(f'host: {hostname}')
    print(f'gpu info: {gpu_info}')

    # create output dir (if needed)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        print('create output dir: {}'.format(args.output_dir))
    else:
        print('use existing output dir: {}'.format(args.output_dir))

    # load the G2P model
    meter = pyln.Meter(args.sample_rate) # create BS.1770 meter (default block size: 400ms)
    phonemizer = Phonemizer.from_checkpoint(args.phonemizer_dict_path)

    # load model
    model = tester.Wav2TTS_infer(args)
    model.cuda(device=args.device)
    model.vocoder.generator.remove_weight_norm()
    model.vocoder.encoder.remove_weight_norm()
    model.eval()

    # read audio
    audio = get_audio(args.speaker_path, meter, args.sample_rate)

    # read sentence
    sentence = open(args.sentence_path, 'r').readlines()[0].strip()

    # convert sentence to phones
    phones = phonemizer(sentence.strip().lower(), lang='en_us').replace('[', ' ').replace(']', ' ').split()
    phones = [''.join(i for i in phone if not i.isdigit()) for phone in phones if phone.strip()]

    for i in range(args.nreps):

        print(f'processing {i}/{nreps} ...')

        # inference
        time_start = time.time()
        synthetic = model([audio], [phones]) # get batch version of the synthesized audios
        time_end = time.time()
        time_elapsed = time_end - time_start

        # get the reference field
        ref_field = os.path.splitext(os.path.basename(args.speaker_path))[0]

        # copy reference file to the output dir
        speaker_path2 = os.path.join(args.output_dir, f'reference-{i}-{ref_field}.wav')
        copyfile(args.speaker_path, speaker_path2)

        # get speaker embedding from the reference audio
        ref_spkr_embedding = model.extract_spkr_embedding(audio)

        # get speaker embedding from the synthesized audio
        syn_spkr_embedding = model.extract_spkr_embedding(synthetic[0])

        # compute speaker similarity score in cosine similarity
        sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)

        # get duration and rtf
        dur_ref_audio = len(audio) / args.sample_rate
        dur_syn_audio = len(synthetic[0]) / args.sample_rate
        dur_process = time_elapsed
        rtf = dur_process / dur_syn_audio

        # write the syn audio file
        output_audiofile = os.path.join(args.output_dir, f'sentence-{i}-{ref_field}.wav')
        sf.write(output_audiofile, synthetic[0], args.sample_rate)

        # write the text file
        output_textfile = os.path.join(args.output_dir, f'sentence-{i}-{ref_field}.txt')
        open(output_textfile, 'w').writelines(sentence+'\n')

        meta = {'ref-wav': speaker_path2,
                'syn-wav': output_audiofile,
                'syn-text': sentence,
                'sss': f'{sss:.3f}',
                'dur-ref': f'{dur_ref_audio:.3f}',
                'dur-syn': f'{dur_syn_audio:.3f}',
                'dur-proc': f'{dur_process:.3f}',
                'rtf': f'{rtf:.2f}',
                'hostname': hostname,
                'gpu': gpu_info}

        # write the meta json file
        output_jsonfile = os.path.join(args.output_dir, f'sentence-{i}-{ref_field}.json')
        with open(output_jsonfile, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)