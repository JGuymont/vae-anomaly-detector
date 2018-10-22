import json
import os
import pandas as pd

from core.data import read_csv

CONFIGS = json.load(open('config.json', 'r'))

def main():
    data_path = os.path.join(CONFIGS['data']['dir'], CONFIGS['data']['filename'])
    inputs, targets = read_csv(data_path)
    print(inputs)
if __name__ == '__main__':
    main()
