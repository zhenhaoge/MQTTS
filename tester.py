import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.wildttstransformer import TTSDecoder
from modules.transformers import TransformerEncoderLayer, TransformerEncoder, TransformerDecoder, TransformerDecoderLayer
from torch.utils import data
from modules.vocoder import Vocoder
import soundfile as sf
import librosa
from librosa.util import normalize
from pyannote.audio import Inference
import random
from tqdm import tqdm

background_noise_dur = 5.0

class Wav2TTS_infer(nn.Module):
    def __init__(self, hp):

        super().__init__()
        self.hp = hp
        self.hp.init = 'std'
        self.TTSdecoder = TTSDecoder(hp, len(self.hp.phoneset))

        self.phone_embedding = nn.Embedding(len(self.hp.phoneset), hp.hidden_size, padding_idx=self.hp.phoneset.index('<pad>'))

        self.spkr_embedding = Inference("pyannote/embedding", window="whole")

        # add attribute 'spkr_linear' (later will be overwritten by self.load())
        spkr_emb_dim = self.spkr_embedding.model.embedding.out_features
        self.spkr_linear = nn.Linear(spkr_emb_dim, hp.hidden_size)

        # load model (this step must be after self.spkr_linear
        # o.w. self.spkr_linear will overwrite weights in self.spkr_linear)
        self.load()

        self.vocoder = Vocoder(hp.vocoder_config_path, hp.vocoder_ckpt_path, with_encoder=True)

    def load(self):
        state_dict = torch.load(self.hp.model_path, map_location='cuda:{}'.format(self.hp.device))['state_dict']
        print (self.load_state_dict(state_dict, strict=False))

    def extract_spkr_embedding(self, wav):

        # normalize wav
        wav = normalize(wav) * 0.95
        # convert wav (np.ndarray:(nsamples,) -> torch.Tensor: (1,nsamples))
        wav = torch.FloatTensor(wav).unsqueeze(0)
        # get speaker embedding (np.ndarry: (spkr_emb_dim:512,))
        speaker_embedding = self.spkr_embedding({'waveform': wav, 'sample_rate': self.hp.sample_rate})
        return speaker_embedding

    def forward(self, wavs, phones):

        self.eval()
        with torch.no_grad():

            batch_size = len(wavs)
            speaker_embeddings = []

            for wav in wavs:
                if self.hp.spkr_embedding_path:
                    speaker_embedding = np.load(wav)
                else:
                    speaker_embedding = self.extract_spkr_embedding(wav)
                speaker_embeddings.append(speaker_embedding)

            # for wav in wavs:
            #     if model.hp.spkr_embedding_path:
            #         speaker_embedding = np.load(wav)
            #     else:
            #         speaker_embedding = model.extract_spkr_embedding(wav)
            #     speaker_embeddings.append(speaker_embedding)

            # convert spkr embeddings (np.ndarray: (batch_size,spkr_emb_dim) -> torch.Tensor: (batch_size,spkr_emb_dim))
            speaker_embeddings = np.array(speaker_embeddings)
            speaker_embeddings = torch.cuda.FloatTensor(speaker_embeddings)

            # normalize spkr embeddings (row-wise)
            norm_spkr = F.normalize(speaker_embeddings, dim=-1)

            # linear transform spkr embeddings (batch_size, spkr_emb_dim) -> (batch_size, hidden_size)
            speaker_embedding = self.spkr_linear(norm_spkr)

            # torch.manual_seed(0)
            # model.spkr_linear = nn.Linear(512, args.hidden_size).cuda()
            # speaker_embedding = model.spkr_linear(norm_spkr)

            # generate random noise for 5 seconds per sample
            if self.hp.noise_seed != -1: torch.manual_seed(self.hp.noise_seed)
            nsamples_background_noise = int(self.hp.sample_rate * background_noise_dur)
            low_background_noise = torch.randn(batch_size, nsamples_background_noise) * self.hp.prior_noise_level
            # nsamples_background_noise = int(model.hp.sample_rate * background_noise_dur)
            # low_background_noise = torch.randn(batch_size, nsamples_background_noise) * model.hp.prior_noise_level

            # get base prior (batch_size, n_frames:nsamples_background_noise/256, n_code_groups)
            base_prior = self.vocoder.encode(low_background_noise.cuda(self.hp.device))
            # base_prior = model.vocoder.encode(low_background_noise.cuda(model.hp.device))

            if self.hp.clean_speech_prior:
                prior = base_prior[:, :self.hp.prior_frame]
                # prior = base_prior[:, :model.hp.prior_frame]
            else:
                prior = None

            # initiate phone features and phone masks (both with dim: (batch_size, maxlen))
            phone_features, phone_masks = [], []
            for phone in phones:
                phone = [self.hp.phoneset.index(ph) for ph in phone if ph != ' ' and ph in self.hp.phoneset]
                # phone = [model.hp.phoneset.index(ph) for ph in phone if ph != ' ' and ph in model.hp.phoneset]
                phone = np.array(phone)
                phone_features.append(phone)

            # pad phones
            maxlen = max([len(x) for x in phone_features])
            for i, ph in enumerate(phone_features):
                # get #phones to be padded
                to_pad = maxlen - len(ph)
                pad = np.zeros([to_pad,], dtype=np.float32)
                pad.fill(self.hp.phoneset.index('<pad>'))
                # pad.fill(model.hp.phoneset.index('<pad>'))
                phone_features[i] = np.concatenate([ph, pad], 0)
                mask = [False] * len(ph)+ [True] * to_pad # True on padded indeces
                phone_masks.append(mask)

            # get phone embedding
            phone_masks = torch.cuda.BoolTensor(phone_masks)
            phone_features = torch.cuda.LongTensor(phone_features)
            phone_embedding = self.phone_embedding(phone_features)
            # phone_embedding = model.phone_embedding(phone_features)

            synthetic = self.TTSdecoder.inference_topkp_sampling_batch(phone_embedding, speaker_embedding, phone_masks, prior=prior)
            # synthetic = model.TTSdecoder.inference_topkp_sampling_batch(phone_embedding, speaker_embedding, phone_masks, prior=prior)

            # print('check!')
            # print('phone_embedding shape: {}'.format(phone_embedding.shape))
            # print('phone_embedding: {}'.format(phone_embedding)) # [2, 140, 768]
            # print('norm_spkr shape: {}'.format(norm_spkr.shape))
            # print('norm_spkr: {}'.format(norm_spkr[0][:10]))
            # print('speaker_embedding shape: {}'.format(speaker_embedding.shape))
            # print('speaker_embedding: {}'.format(speaker_embedding[0][:10]))
            # print('prior shape: {}'.format(prior.shape))
            # print('prior: {}'.format(prior))
            # print('synthetic shape: {}'.format([len(s) for s in synthetic]))
            # print('synthetic: {}'.format(synthetic[0][:10]))

            padded_synthetic, lengths = [], []
            maxlen = max([len(x) for x in synthetic])
            for i, s in enumerate(synthetic):
                to_pad = maxlen - len(s)
                lengths.append(len(s) * 256) # Have to change according to vocoder stride!
                pad = base_prior[i, base_prior.size(1)//2].unsqueeze(0).expand(to_pad, -1)
                if self.hp.clean_speech_prior:
                    s = torch.cat([prior[i, :], s, pad], 0)
                else:
                    s = torch.cat([s, pad], 0)
                padded_synthetic.append(s)
            padded_synthetic = torch.stack(padded_synthetic, 0)
            synthetic = self.vocoder(padded_synthetic, norm_spkr)
            outputs = []
            for l, s in zip(lengths, synthetic):
                if self.hp.clean_speech_prior:
                    l = l + self.hp.prior_frame * 256
                outputs.append(s[0, : l].cpu().numpy())
            return outputs
