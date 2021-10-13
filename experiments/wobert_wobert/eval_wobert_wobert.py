# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import os
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(os.fspath(PROJECT_DIR))

import torch
from transformers import EncoderDecoderModel

from experiments.eval_encoder_decoder_model import eval_test_result, get_test_dataloader
from experiments.utils import (
    create_tokenizer,
    prepare_evaluate,
    set_eval_parser,
    update_parser_defaults,
)


def main():
  hyparam_file = Path(__file__).resolve().parent / "hyparams.json"
  parser = set_eval_parser()
  update_parser_defaults(hyparam_file, parser, train=False)
  args = parser.parse_args()

  args.model_name = "wobert_wobert"
  args.project_dir = PROJECT_DIR
  ckpt_dir, output_file = prepare_evaluate(args)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  vocab_file = PROJECT_DIR / "vocabs" / "wobert_vocab.txt"
  encoder_tokenizer = create_tokenizer(
      vocab_path=vocab_file,
      tokenizer_type="wobert",
      use_custom_sos_eos=False,
  )
  decoder_tokenizer = create_tokenizer(
      vocab_path=vocab_file,
      tokenizer_type="wobert",
      use_custom_sos_eos=True,
  )

  test_dataloader = get_test_dataloader(
      PROJECT_DIR / "data" / "test.txt",
      encoder_tokenizer,
      args.batch_size,
      max_seq_length=args.max_seq_length,
  )

  encoder_decoder_model = EncoderDecoderModel.from_pretrained(ckpt_dir).to(
      device)

  eval_test_result(
      test_dataloader=test_dataloader,
      result_path=output_file,
      decoder_tokenizer=decoder_tokenizer,
      model=encoder_decoder_model,
      max_seq_length=args.max_seq_length,
      num_beams=args.num_beams,
      device=device,
  )


if __name__ == '__main__':
  main()
