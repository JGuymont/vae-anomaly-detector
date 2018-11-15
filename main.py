import json
import os
import pandas as pd
from unidecode import unidecode

import torch
from torch.autograd import Variable

from core.data import Data, DataLoader
from core.vae import VAE

CONFIGS = json.load(open('config.json', 'r'))
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#device = 'cpu'

def main():
    data_path = os.path.join(CONFIGS['data']['dir'], CONFIGS['data']['filename'])
    data = Data(data_path, split=[0.7, 0.15, 0.15])
    data.preprocess()
    data.vectorize()

    trainloader = DataLoader(data.train(), batch_size=128)
    trainloader2 = DataLoader(data.train(), batch_size=1)
    devloader = DataLoader(data.valid(), batch_size=1)

    vae = VAE(data.input_dim_(), 256, 128, device).to(device)

    vae.train(trainloader, trainloader2, devloader)
    
if __name__ == '__main__':
    main()