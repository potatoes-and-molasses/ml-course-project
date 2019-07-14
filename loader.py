import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
from pathlib import Path
from functools import lru_cache


def pad_collate(batch):
    # find longest sequence
    max_input_length = max([len(x[0]) for x in batch])
    max_output_length = max([len(x[1]) for x in batch])

    batch = [(x[0] + (max_input_length - len(x[0])) * [0],  # input
              x[1] + (max_output_length - len(x[1])) * [-1],  # output
              len(x[0]),  # input length
              len(x[1]),  # output length
              x[2]  # path
              ) for x in batch]
    # stack all
    input = torch.stack([torch.tensor(x[0], dtype=torch.long) for x in batch], dim=0)
    output = torch.stack([torch.tensor(x[1], dtype=torch.long) for x in batch], dim=0)
    input_len = torch.tensor([x[2] for x in batch], dtype=torch.long)
    output_len = torch.tensor([x[3] for x in batch], dtype=torch.long)
    paths = [x[4] for x in batch]

    return input, output, input_len, output_len, paths


def make_dataset(root):
    return [str(s) for s in Path(root).glob('**/*')]


class ParaLoader(Dataset):
    def __init__(self, root, word_dict_path):
        super(ParaLoader, self).__init__()
        self.data_files = make_dataset(root)
        self.word_dict = np.load(word_dict_path, allow_pickle=True).item()

    def process_file(self, idx):
        with open(self.data_files[idx], 'r') as f:
            i, o = f.read().split('|||')

        inp_embedding = [self.word_dict[w] for w in ['<SOS>'] + i.split() + ['<EOS>']]
        out_embedding = [self.word_dict[w] for w in o.split() + ['<EOS>']]

        return inp_embedding, out_embedding

    def __getitem__(self, idx):
        input, output = self.process_file(idx)

        return input, output, self.data_files[idx]

    @lru_cache(maxsize=None)
    def __len__(self):
        return len(self.data_files)


if __name__ == '__main__':
    a = next(
        enumerate(DataLoader(ParaLoader(root='data/train', word_dict_path='words_dict.npy'), collate_fn=pad_collate,
                             batch_size=1)))
    print(a)
