import json
import math
from pathlib import Path
from typing import Union, Optional, Dict

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.models.bert.tokenization_bert import BertTokenizer
from models.error_maker import ErrorMaker


class CgecDataset(Dataset):

  def __init__(self,
               file_path: Path,
               encoder_tokenizer: Union[PreTrainedTokenizerFast,
                                        PreTrainedTokenizer],
               decoder_tokenizer: Union[PreTrainedTokenizerFast,
                                        PreTrainedTokenizer],
               max_length: int = 512,
               error_maker: Optional[ErrorMaker] = None,
               seg_dict: Optional[Dict[str, int]] = None,
               pos_dict: Optional[Dict[str, int]] = None,
               dataset_ratio=1.0):
    self.samples = open(file_path, 'r').readlines()
    self.encoder_tokenizer = encoder_tokenizer
    self.decoder_tokenizer = decoder_tokenizer
    self.encoder_max_length = max_length
    self.decoder_max_length = max_length
    self.error_maker = error_maker
    self.seg_dict = seg_dict
    self.pos_dict = pos_dict
    self.dataset_ratio = dataset_ratio

  def __len__(self):
    return math.ceil(len(self.samples) * self.dataset_ratio)

  def __getitem__(self, idx):
    sample = dict()
    data = json.loads(self.samples[idx].strip())
    source = data["source"]
    correct = data["correct"]
    if self.error_maker:
      _, source = self.error_maker.dynamic_mask(source)

    source_inputs = self.encoder_tokenizer(source,
                                           padding="max_length",
                                           truncation=True,
                                           add_special_tokens=True,
                                           max_length=self.encoder_max_length)
    target_inputs = self.decoder_tokenizer(correct,
                                           padding="max_length",
                                           truncation=True,
                                           add_special_tokens=True,
                                           max_length=self.decoder_max_length)
    sample["input_ids"] = source_inputs.input_ids
    sample["attention_mask"] = source_inputs.attention_mask
    sample["decoder_input_ids"] = target_inputs.input_ids
    sample["decoder_attention_mask"] = target_inputs.attention_mask
    sample["labels"] = target_inputs.input_ids.copy()

    # it is very important to remember to ignore the loss of the padded labels.
    # In Transformers this can be done by setting the label to -100.
    sample["labels"] = [
        -100 if token == self.decoder_tokenizer.pad_token_id else token
        for token in sample["labels"]
    ]
    return sample


def collate_cgec_batch_fn(batch):
  sample = dict()
  for item in batch:
    for key, values in item.items():
      if key not in sample:
        sample[key] = []
      sample[key].append(torch.as_tensor(values))
  for k, v in sample.items():
    sample[k] = torch.stack(v)
  return sample


def main():
  pass


if __name__ == '__main__':
  main()
