import librosa
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tester import Wav2TTS_infer
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

# import importlib
# importlib.reload(tester)

parser = argparse.ArgumentParser()

#Path
parser.add_argument('--phonemizer_dict_path', type=str, required=True)
parser.add_argument('--outputdir', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--spkr_embedding_path', type=str, default=None)

#Data
parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_samples', type=int, default=-1)

#Sampling
parser.add_argument('--use_repetition_gating', action='store_true')
parser.add_argument('--repetition_penalty', type=float, default=1.0)
parser.add_argument('--sampling_temperature', type=float, default=1.0)
parser.add_argument('--top_k', type=int, default=-1)
parser.add_argument('--min_top_k', type=int, default=1)
parser.add_argument('--top_p', type=float, default=0.7)
parser.add_argument('--length_penalty_max_length', type=int, default=50)
parser.add_argument('--length_penalty_max_prob', type=float, default=0.8)
parser.add_argument('--max_output_length', type=int, default=100000)
parser.add_argument('--phone_context_window', type=int, default=4)

#Speech Prior
parser.add_argument('--noise_seed', default=-1, help='-1 means no seed, random noise')
parser.add_argument('--clean_speech_prior', action='store_true')
parser.add_argument('--prior_noise_level', type=float, default=1e-5)
parser.add_argument('--prior_frame', type=int, default=3)

#Other
parser.add_argument('--device', type=int, default=0)

# --- runtime mode (start) ---
args = parser.parse_args()
# --- runtime mode (end) ---

# # --- interactive mode (start)

# args = argparse.ArgumentParser()

# args.input_path  = os.path.join(os.getcwd(), "speaker_to_text.json")

# args.input_path = "/home/users/zge/code/repo/mqtts/speaker_to_text.json"
# args.config_path = "ckpt/OTS/config.json"
# args.model_path = "ckpt/OTS/transformer.ckpt"
# args.outputdir = "outputs/infer_samples_1"
# args.noise_seed = 0

# # args.input_path = "/home/users/zge/code/repo/mqtts_alex/datasets/test_sets/WER.json"
# # args.config_path = "ckpt/OTS_alex/config.json"
# # args.model_path = "ckpt/OTS_alex/transformer.ckpt"
# # args.outputdir = "outputs/infer_samples"
# # args.noise_seed = 1

# args.sample_rate = 16000
# args.phonemizer_dict_path = "en_us_cmudict_forward.pt"
# args.batch_size = 2
# args.num_samples = 16
# args.top_p = 0.8
# args.min_top_k = 2
# args.max_output_length = 100000
# args.phone_context_window = 3
# args.clean_speech_prior = True
# args.spkr_embedding_path = ''
# args.prior_noise_level = 1e-5
# args.prior_frame = 3
# args.device = 0

# # --- interactive mode (end) ---

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# args.phoneset = ['<pad>', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', ',', '.']
# open('phoneset.txt', 'w').writelines('\n'.join(args.phoneset) + '\n')
args.phoneset = [l.rstrip() for l in open('phoneset.txt', 'r').readlines()]

# read transformer config
with open(args.config_path, 'r') as f:
    argdict = json.load(f)
    assert argdict['sample_rate'] == args.sample_rate, \
        'Sampling rate not consistent, stated {}, but the model is trained on {}'.format(
        args.sample_rate, argdict['sample_rate'])
    argdict.update(args.__dict__)
    args.__dict__ = argdict

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

# def extract_spkr_embedding(model, wav):
#     # normalize wav
#     wav = normalize(wav) * 0.95
#     # convert wav (np.ndarray:(nsamples,) -> torch.Tensor: (1,nsamples))
#     wav = torch.FloatTensor(wav).unsqueeze(0)
#     # get speaker embedding (np.ndarry: (spkr_emb_dim:512,))
#     speaker_embedding = model.spkr_embedding({'waveform': wav, 'sample_rate': model.hp.sample_rate})
#     return speaker_embedding

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

if __name__ == '__main__':

    hostname = get_hostname()
    gpu_info = get_gpu_info(args.device)
    print(f'host: {hostname}')
    print(f'gpu info: {gpu_info}')

    # create output dir (if needed)
    if not os.path.isdir(args.outputdir):
        os.makedirs(args.outputdir)
        print('create output dir: {}'.format(args.outputdir))
    else:
        print('use existing output dir: {}'.format(args.outputdir))

    # load the G2P model
    meter = pyln.Meter(args.sample_rate) # create BS.1770 meter (default block size: 400ms)
    phonemizer = Phonemizer.from_checkpoint(args.phonemizer_dict_path)

    # load in a list of (speaker path and sentence)
    with open(args.input_path, 'r') as f:
        input_file = json.load(f)
    num_input_file = len(input_file)
    print('# of input files in {}: {}'.format(args.input_path, num_input_file))

    # select the first num_samples file
    if args.num_samples == -1:
        num_samples = num_input_file
    else:
        num_samples = min(num_input_file, args.num_samples)
    print('first {} input files are selected'.format(num_samples))

    # torch.cuda.set_device(1)

    model = tester.Wav2TTS_infer(args)
    model.cuda(device=args.device)
    model.vocoder.generator.remove_weight_norm()
    model.vocoder.encoder.remove_weight_norm()
    model.eval()

    i_wavs, i_phones, written = [], [], 0
    num_batches = int(np.ceil(num_samples/args.batch_size))
    file_groups = [{} for _ in range(num_samples)]
    for i, (speaker_path, sentence) in enumerate(input_file[:num_samples]):

        # speaker_path, sentence = input_file[i]

        if type(speaker_path) is list and len(speaker_path) == 1:
            speaker_path = speaker_path[0]

        # speaker_path = os.path.join(os.getcwd(), speaker_path)
        if args.spkr_embedding_path:
            audio = os.path.join(args.spkr_embedding_path, os.path.basename(speaker_path)[:-4] + '.npy')
        else:
            audio = get_audio(speaker_path, meter, args.sample_rate)
        i_wavs.append(audio)

        # convert sentence to phones
        phones = phonemizer(sentence.strip().lower(), lang='en_us').replace('[', ' ').replace(']', ' ').split()
        phones = [''.join(i for i in phone if not i.isdigit()) for phone in phones if phone.strip()]
        i_phones.append(phones)

        if len(i_wavs) == args.batch_size or i == num_samples - 1:

            # print('stop here!')
            # break

            batch_idx = (written//args.batch_size)+1 # start from 1
            print ('Inferencing batch {}, total {} baches.'.format(batch_idx, num_batches))

            # batch inference
            time_start = time.time()
            synthetic = model(i_wavs, i_phones) # get batch version of the synthesized audios
            time_end = time.time()
            time_elapsed = time_end - time_start

            for j, s in enumerate(synthetic):

                # s = synthetic[j] # synthesized audio

                # get reference file id
                ref_path = input_file[written][0]
                if type(ref_path) is list and len(ref_path) == 1:
                    ref_path = ref_path[0]
                ref_field = os.path.splitext(os.path.basename(ref_path))[0]

                # get output file id
                output_fileid = '{:02d}-{}'.format(written+1, ref_field)

                # copy reference file to the output dir
                ref_path2 = os.path.join(args.outputdir, 'reference-{}.wav'.format(output_fileid))
                copyfile(ref_path, ref_path2)

                # get reference speaker embedding from reference audio
                ref_spkr_embedding = model.extract_spkr_embedding(i_wavs[j])
                # wav = i_wavs[j]
                # from librosa.util import normalize
                # wav = normalize(wav) * 0.95
                # wav = torch.FloatTensor(wav).unsqueeze(0)
                # ref_spkr_embedding = model.spkr_embedding({'waveform': wav, 'sample_rate': model.hp.sample_rate})

                # get synthesized speaker embedding
                syn_spkr_embedding = model.extract_spkr_embedding(s)

                # compute speaker similarity score in cosine similarity
                sss = cos_sim(ref_spkr_embedding, syn_spkr_embedding)

                # write the syn audio file
                output_audiofile = os.path.join(args.outputdir, 'sentence-{}.wav'.format(output_fileid))
                sf.write(output_audiofile, s, args.sample_rate)

                # write the text file
                output_textfile = os.path.join(args.outputdir, 'sentence-{}.txt'.format(output_fileid))
                text = input_file[written][1]
                open(output_textfile, 'w').writelines(text+'\n')

                # get sss from output files (get rid of the difference in normalization)
                ref_spkr_embedding2 = model.extract_spkr_embedding(get_audio(ref_path2, meter))
                syn_spkr_embedding2 = model.extract_spkr_embedding(get_audio(output_audiofile, meter))
                sss2 = cos_sim(ref_spkr_embedding2, syn_spkr_embedding2)

                dur_ref_audio = len(i_wavs[j]) / args.sample_rate
                dur_syn_audio = len(s) / args.sample_rate
                dur_process = time_elapsed / args.batch_size
                rtf = dur_process / dur_syn_audio

                file_groups[written] = {'ref-wav': ref_path2, 'syn-wav': output_audiofile, 'syn-txt': text,
                                        'sss': '{:.3f}'.format(sss), 'sss2': '{:.3f}'.format(sss2),
                                        'dur-ref': '{:.3f}'.format(dur_ref_audio),
                                        'dur-syn': '{:.3f}'.format(dur_syn_audio),
                                        'dur-proc': '{:.3f}'.format(dur_process),
                                        'rtf': '{:.2f}'.format(rtf), 'batch-size': str(args.batch_size),
                                        'hostname': hostname, 'gpu': gpu_info}

                written += 1

            # for k in range(written-args.batch_size, written):
            #     print(file_groups[k])

            i_wavs, i_phones = [], []

    # print the avg. speaker similarity score
    sss_mean = np.mean([float(file_group['sss']) for file_group in file_groups])
    print('mean spkr similarity score: {:.3f}'.format(sss_mean))

    # print the avg. rtf
    rtf_mean = np.mean([float(file_group['rtf']) for file_group in file_groups])
    print('mean RTF (batch size: {}): {:.3f}'.format(args.batch_size, rtf_mean))

    # write file group (ref filename, syn filename, syn text, sss) to a json file for future reference
    file_group_jsonfile = os.path.join(args.outputdir, 'file_groups.json')
    with open(file_group_jsonfile, 'w') as fp:
        json.dump(file_groups, fp, indent=2)
    print('wrote file group json file to {}'.format(os.path.join(os.getcwd(), file_group_jsonfile)))
