# -*- coding: utf-8 -*-
"""
Author: Andrej Leban
Created on Sun May 29 13:05:27 2022
"""

import argparse
import os
import time

import sounddevice as sd
# import soundfile as sf
import torch

from infowavegan import WaveGANGenerator
from utils import get_continuation_fname

# cf: https://github.com/pytorch/pytorch/issues/16797
# class CPU_Unpickler(pk.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name == '_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         else:
#             return super().find_class(module, name)

if __name__ == "__main__":
    # generator = CPU_Unpickler(open("generator.pkl", 'rb')).load()
    # discriminator = CPU_Unpickler(open("discriminator.pkl", 'rb')).load()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='Directory where checkpoints are saved'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        required=True,
        help='Training Directory'
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Q-net categories'
    )
    parser.add_argument(
        '--slice_len',
        type=int,
        default=16384,
    )

    args = parser.parse_args()
    epoch = args.epoch
    dir = args.dir
    sample_rate = args.sample_rate
    slice_len = args.slice_len

    # Load generator from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fname, _ = get_continuation_fname(epoch, dir)
    G = WaveGANGenerator(slice_len=slice_len)
    G.load_state_dict(torch.load(os.path.join(dir, fname + "_G.pt"),
                                 map_location = device))
    G.to(device)
    G.eval()

    # Generate from random noise
    for i in range(100):
        z = torch.FloatTensor(1, 100).uniform_(-1, 1).to(device)
        genData = G(z)[0, 0, :].detach().cpu().numpy()
        # write(f'out.wav', sample_rate, (genData * 32767).astype(np.int16))
        sd.play(genData, sample_rate)
        time.sleep(1)
