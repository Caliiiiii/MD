import torch
from torch.utils.data import Dataset
import numpy as np


class dataset(Dataset):
    """wrap in PyTorch Dataset"""

    def __init__(self, examples):
        """
        :param examples: examples returned by VUA_All_Processor or Verb_Processor
        """
        super(dataset, self).__init__()
        self.data = examples

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    batch = list(zip(*batch))  #
    ids_ls, segs_ls, att_mask_ls, ids_rs, att_mask_rs, segs_rs, neighbor_mask, labels = batch

    ids_ls = torch.tensor(ids_ls, dtype=torch.long)
    segs_ls = torch.tensor(segs_ls, dtype=torch.long)
    att_mask_ls = torch.tensor(att_mask_ls, dtype=torch.long)

    ids_rs = torch.tensor(ids_rs, dtype=torch.long)  # - (16,70)
    att_mask_rs = torch.tensor(att_mask_rs, dtype=torch.long)  # - (16,70)
    segs_rs = torch.tensor(segs_rs, dtype=torch.long)  # - (16,70)

    neighbor_mask = torch.tensor(neighbor_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return ids_ls, segs_ls, att_mask_ls, ids_rs, att_mask_rs, segs_rs, neighbor_mask, labels


def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)  

    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.LongTensor(s)  
    return tokens
