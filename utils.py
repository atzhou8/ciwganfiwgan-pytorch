import os
import re
import torch
import torch.optim as optim
import numpy as np

from infowavegan import WaveGANGenerator, WaveGANDiscriminator, WaveGANQNetwork

def get_continuation_fname(CONT, logdir):
    if CONT == "":
        # Take last
        files = [f for f in os.listdir(logdir) if os.path.isfile(os.path.join(logdir, f))]
        epochNames = [re.match(f"epoch(\d+)_step\d+.*\.pt$", f) for f in files]
        epochs = [match.group(1) for match in filter(lambda x: x is not None, epochNames)]
        maxEpoch = sorted(epochs, reverse=True, key=int)[0]

        fPrefix = f'epoch{maxEpoch}_step'
        fnames = [re.match(f"({re.escape(fPrefix)}\d+).*\.pt$", f) for f in files]
        # Take first if multiple matches (unlikely)
        fname = ([f for f in fnames if f is not None][0]).group(1)
    else:
        # parametrized by the epoch
        fPrefix = f'epoch{CONT}_step'
        files = [f for f in os.listdir(logdir) if os.path.isfile(os.path.join(logdir, f))]
        fnames = [re.match(f"({re.escape(fPrefix)}\d+).*\.pt$", f) for f in files]
        # Take first if multiple matches (unlikely)
        fname = ([f for f in fnames if f is not None][0]).group(1)

    return fname
