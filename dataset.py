from PIL import Image
from transformers import GPT2TokenizerFast
import numpy as np
import torch

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

#TODO Data Augmentation

# NOTE: mean and std for ViT is 0.5, unlike the standard ImageNet mean and std.

class Dataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data.iloc[idx, :]
        image = datum['image']
        caption = datum['caption']

        image = Image.open(image).convert('RGB')
        image = np.array(image)

        caption = f"{caption}<|endoftext|>"

        input_ids = tokenizer(caption, truncation=True)['input_ids']

        labels = input_ids.copy()
        labels[:-1] = input_ids[1:]

        return image, input_ids, labels

# See more info on batch collation here: https://plainenglish.io/blog/understanding-collate-fn-in-pytorch-f9d1742647d3

def prepare_bath(batch):
    image = [i[0] for i in batch]
    input_ids = [i[1] for i in batch]
    labels = [i[2] for i in batch]

    image = torch.stack(image, dim=0)

    input_ids = tokenizer.pad(
        {'input_ids': input_ids},
        padding='longest',
        return_attention_mask=False,
        return_tensors='pt'
    )['input_ids']

    labels = tokenizer.pad(
        {'input_ids': labels},
        padding='longest',
        return_attention_mask=False,
        return_tensors='pt'
    )['input_ids']

    mask = (input_ids != tokenizer.pad_token_id).long()

    labels[mask == 0] = -100

    return image, input_ids, labels


