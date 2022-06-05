# -*- coding: utf-8 -*-
"""
Author: Andrej Leban
Created on Sun May 29 13:05:27 2022
"""

import io
import pickle as pk

import sounddevice as sd
import soundfile as sf
import torch

import time

from infowavegan import WaveGANGenerator

# cf: https://github.com/pytorch/pytorch/issues/16797

class CPU_Unpickler(pk.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


if __name__ == "__main__":
    # generator = CPU_Unpickler(open("generator.pkl", 'rb')).load()

    # discriminator = CPU_Unpickler(open("discriminator.pkl", 'rb')).load()

    # inp, fs = sf.read("../GANdata/8words_train/ask_1.wav")

    # genData = generator.cpu()(torch.randn(1, 100)).detach().numpy()[0][0]
    # sd.play(genData, fs)

    # Load generator from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = WaveGANGenerator(slice_len=65536)
    G.load_state_dict(torch.load("epoch750_step41305_G.pt", map_location=device))
    G.to(device)
    G.eval()

    for i in range(100):
        # Generate from random noise
        z = torch.FloatTensor(1, 100).uniform_(-1, 1).to(device)
        genData = G(z)[0, 0, :].detach().cpu().numpy()
        # write(f'out.wav', sample_rate, (genData * 32767).astype(np.int16))
        sd.play(genData, 32000)
        time.sleep(1)
