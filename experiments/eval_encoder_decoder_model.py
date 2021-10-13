# -*- coding: utf-8 -*-

import sys
import csv
from pathlib import Path
import os
import torch
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(os.fspath(PROJECT_DIR))

from torch.utils.data import DataLoader

from models.dataset import collate_cgec_batch_fn
from experiments.utils import batch_decode

WORKER_NUM = os.cpu_count()


class TestDataset:

  def __init__(
      self,
      file_path,
      tokenizer,
      max_seq_length=256,
  ):
    self.tokenizer = tokenizer
    self.max_seq_length = max_seq_length
    self.dataset = open(file_path, 'r').readlines()

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    data = self.dataset[idx].strip()
    inputs = self.tokenizer(
        data,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
        max_length=self.max_seq_length,
        # return_tensors="pt",
    )
    return inputs


def get_test_dataloader(
    file_path,
    encoder_tokenizer,
    batch_size=32,
    max_seq_length=256,
):
  test_dataset = TestDataset(
      file_path=file_path,
      tokenizer=encoder_tokenizer,
      max_seq_length=max_seq_length,
  )
  return DataLoader(
      test_dataset,
      batch_size=batch_size,
      shuffle=False,
      drop_last=False,
      collate_fn=collate_cgec_batch_fn,
      num_workers=WORKER_NUM,
  )


def eval_test_result(
    test_dataloader,
    result_path,
    decoder_tokenizer,
    model,
    max_seq_length=256,
    num_beams=12,
    device='cpu',
):
  model.eval()

  with open(result_path, 'w', newline='', encoding="utf-8") as fout:
    writer = csv.writer(fout, delimiter='\t')

    for inputs in tqdm(test_dataloader):
      input_ids = inputs["input_ids"]
      attention_mask = inputs["attention_mask"]
      input_ids = input_ids.to(device)
      attention_mask = attention_mask.to(device)

      with torch.no_grad():
        outputs = model.generate(
            input_ids,
            bos_token_id=decoder_tokenizer.cls_token_id,
            eos_token_id=decoder_tokenizer.sep_token_id,
            pad_token_id=decoder_tokenizer.pad_token_id,
            decoder_start_token_id=decoder_tokenizer.cls_token_id,
            attention_mask=attention_mask,
            max_length=max_seq_length,
            min_length=0,
            num_beams=num_beams,
        )

      batch_sentences = batch_decode(outputs, decoder_tokenizer)
      for sentence in batch_sentences:
        writer.writerow([sentence])
      fout.flush()
