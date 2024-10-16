# move dataset (e.g., gigaspeech) to a backup dir to save space in my home dir
#
# Zhenhao Ge, 2024-04-16

import os
from shutil import copyfile, rmtree
import json
import glob
import datasets

from pathlib import Path
home_dir = str(Path.home())

def write_txt_file(txt_file, text):
    with open(txt_file, 'w') as f:
        f.write('{}\n'.format(text))

def write_json_file(json_file, meta):
    with open(json_file, 'w') as f:
        json.dump(meta, f)   

# change data downloading dir from default (~/.cache/huggingface/datasets) to save space in my home dir
datasets.config.HF_DATASETS_CACHE = '/home/users/zge/data1/datasets/cache'

# once datasets.config.HF_DATASETS_CACHE is set , the default cache dir is changed
# datasets.config.DOWNLOADED_DATASETS_PATH = Path(datasets.config.HF_DATASETS_CACHE)

# download gigaspeech (xs)
gs = datasets.load_dataset("speechcolab/gigaspeech", "xs", use_auth_token=True)

# set destination data dir
dest_dir = '/home/users/zge/data1/datasets/GigaSpeech-HF'

copy_completed = {}
for cat in gs.keys():

    print('moving {} data ...'.format(cat))

    # create output dir
    out_dir = os.path.join(dest_dir, cat)
    os.makedirs(out_dir, exist_ok=True)

    nsamples = len(gs[cat])
    for i in range(nsamples):

        src_file = gs[cat][i]['audio']['path']
        filename = os.path.basename(src_file)
        basename = os.path.splitext(filename)[0]
        print('copying {} sample {}/{}: {} ...'.format(cat, i+1, nsamples, basename))

        # set the audio, text and conf file
        des_wav_file = os.path.join(out_dir, filename)
        des_txt_file = os.path.join(out_dir, '{}.txt'.format(basename))
        des_json_file = os.path.join(out_dir, '{}.json'.format(basename))

        # copy the audio file
        if not os.path.isfile(des_wav_file):
            copyfile(src_file, des_wav_file)

        # write the text file
        if not os.path.isfile(des_txt_file):
            text = gs[cat][i]['text']
            write_txt_file(des_txt_file, text)

        # write the conf file
        if not os.path.isfile(des_json_file):
            meta = {k:v for k, v in gs[cat][i].items() if (k != 'audio' and k != 'text')}
            write_json_file(des_json_file, meta)

    des_wav_files = glob.glob(os.path.join(out_dir, '*.wav'))
    nsamples2 = len(des_wav_files)
    if nsamples2 == nsamples:
        copy_completed[cat] = True

if all(copy_completed.values()):
    print('data copy completed, deleting the original ...')

    # delete the download folder (large)
    dir_to_be_deleted = os.path.join(home_dir, '.cache/huggingface/datasets', 'downloads')
    rmtree(dir_to_be_deleted)

    # delete the speechcolab__gigaspeech folder (small)
    dir_to_be_deleted = os.path.join(home_dir, '.cache/huggingface/datasets', 'speechcolab___gigaspeech')
    rmtree(dir_to_be_deleted, ignore_errors=True)








