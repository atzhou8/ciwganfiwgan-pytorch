import torch
import numpy as np
import torch.optim as optim
from scipy.io.wavfile import read
from torch.utils.data import DataLoader
from infowavegan import WaveGANGenerator, WaveGANDiscriminator, WaveGANQNetwork
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
import pickle as pk
from tqdm import tqdm


class AudioDataSet:
    def __init__(self, datadir, slice_len):
        print("Loading data")
        dir = os.listdir(datadir)
        x = np.zeros((len(dir), 1, slice_len))
        i = 0
        for file in tqdm(dir):
            audio = read(datadir + file)[1]
            if audio.shape[0] < slice_len:
                audio = np.pad(audio, (0, slice_len - audio.shape[0]))
            audio = audio[:slice_len]
            audio = audio.astype(np.float32) / 32767
            audio /= np.max(np.abs(audio))
            x[i, 0, :] = audio
            i += 1

        self.len = len(x)
        self.audio = torch.from_numpy(np.array(x, dtype=np.float32))

    def __getitem__(self, index):
        return self.audio[index]

    def __len__(self):
        return self.len


def gradient_penalty(G, D, real, fake, epsilon):
    x_hat = epsilon * real + (1 - epsilon) * fake
    scores = D(x_hat)
    grad = torch.autograd.grad(
        outputs=scores,
        inputs=x_hat,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True
    )[0]
    grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1)  # norm along each batch
    penalty = ((grad_norm - 1) ** 2).unsqueeze(1)
    return penalty


if __name__ == "__main__":
    # Training Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datadir',
        type=str,
        required=True,
        help='Training Directory'
    )
    parser.add_argument(
        '--logdir',
        type=str,
        required=True,
        help='Log Directory'
    )
    parser.add_argument(
        '--num_categ',
        type=int,
        default=1,
        help='Q-net categories'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=5000,
        help='Epochs'
    )
    parser.add_argument(
        '--slice_len',
        type=int,
        default=16384,
        help='Epochs'
    )

    # Q-net Arguments
    Q_group = parser.add_mutually_exclusive_group()
    Q_group.add_argument(
        '--ciw',
        action='store_true',
        help='Trains a ciwgan'
    )
    Q_group.add_argument(
        '--fiw',
        action='store_true',
        help='Trains a fiwgan'
    )
    args = parser.parse_args()
    train_Q = args.ciw or args.fiw

    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datadir = args.datadir
    logdir = args.logdir
    SLICE_LEN = args.slice_len
    NUM_CATEG = args.num_categ
    NUM_EPOCHS = args.num_epochs
    WAVEGAN_DISC_NUPDATES = 5
    BATCH_SIZE = 16
    LAMBDA = 10
    LEARNING_RATE = 1e-4
    BETA1 = 0.5
    BETA2 = 0.9

    # Load data
    dataset = AudioDataSet(datadir, SLICE_LEN)
    dataloader = DataLoader(
        dataset,
        BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    # Load models
    G = WaveGANGenerator(slice_len=SLICE_LEN,).to(device).train()
    D = WaveGANDiscriminator(slice_len=SLICE_LEN).to(device).train()
    if train_Q:
        Q = WaveGANQNetwork(slice_len=SLICE_LEN, num_categ=NUM_CATEG).to(device).train()

    # Optimizers
    optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    if args.fiw:
        optimizer_Q = optim.RMSprop(Q.parameters(), lr=LEARNING_RATE)
        criterion_Q = torch.nn.BCEWithLogitsLoss()
    elif args.ciw:
        optimizer_Q = optim.RMSprop(Q.parameters(), lr=LEARNING_RATE)
        criterion_Q = torch.nn.CrossEntropyLoss()

    # Set Up Writer
    writer = SummaryWriter(logdir)
    step = 0
    for epoch in range(NUM_EPOCHS):
        print("Epoch {} of {}".format(epoch, NUM_EPOCHS))
        print("-----------------------------------------")
        pbar = tqdm(dataloader)
        real = dataset[:BATCH_SIZE].to(device)
        for i, real in enumerate(pbar):
            # D Update
            optimizer_D.zero_grad()
            real = real.to(device)
            epsilon = torch.rand(BATCH_SIZE, 1, 1).repeat(1, 1, SLICE_LEN).to(device)
            _z = torch.FloatTensor(BATCH_SIZE, 100 - NUM_CATEG).uniform_(-1, 1).to(device)
            c = torch.FloatTensor(BATCH_SIZE, NUM_CATEG).bernoulli_().to(device)
            z = torch.cat((c, _z), dim=1)
            fake = G(z)
            penalty = gradient_penalty(G, D, real, fake, epsilon)

            D_loss = torch.mean(D(fake) - D(real) + LAMBDA * penalty)
            writer.add_scalar('Loss/Discriminator', D_loss.detach().item(), i)
            D_loss.backward()
            optimizer_D.step()

            if i % WAVEGAN_DISC_NUPDATES == 0:
                optimizer_G.zero_grad()
                if train_Q:
                    optimizer_Q.zero_grad()
                _z = torch.FloatTensor(BATCH_SIZE, 100 - NUM_CATEG).uniform_(-1, 1).to(device)
                c = torch.FloatTensor(BATCH_SIZE, NUM_CATEG).bernoulli_().to(device)
                z = torch.cat((c, _z), dim=1)
                G_z = G(z)

                # G Loss
                G_loss = torch.mean(-D(G_z))
                G_loss.backward(retain_graph=True)
                writer.add_scalar('Loss/Generator', G_loss.detach().item(), i)

                # Q Loss
                if train_Q:
                    Q_loss = criterion_Q(Q(G_z), c)
                    Q_loss.backward()
                    writer.add_scalar('Loss/Q_Network', Q_loss.detach().item(), i)
                    optimizer_Q.step()

                # Update
                optimizer_G.step()
            step += 1

        # torch.save(G.state_dict(), f'./checkpoints/epoch{epoch}_step{step}_G.pt')
        # torch.save(D.state_dict(), f'./checkpoints/epoch{epoch}_step{step}_D.pt')
        # if train_Q:
            # torch.save(Q.state_dict(), f'./checkpoints/epoch{epoch}_step{step}_Q.pt')

    # NOTE: temporary workaround:
    pk.dump(G, open("generator.pkl", "wb"))
    pk.dump(D, open("discriminator.pkl", "wb"))
