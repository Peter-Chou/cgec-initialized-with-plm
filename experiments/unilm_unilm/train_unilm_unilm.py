# -*- coding: utf-8 -*-
"""model id.

unilm's oid sha256: 0f8655c49be2307e004ae6883eb10ba697b4853cf54ce22dc7214b0750c8b4cf

"""

import sys
from pathlib import Path
import os
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(os.fspath(PROJECT_DIR))

from transformers import EncoderDecoderModel

from models.dataset import collate_cgec_batch_fn
from models.trainer import Trainer
from experiments.utils import (
    set_train_parser,
    update_parser_defaults,
    get_latest_epoch_checkpoint,
    create_tokenizer,
    create_datasets,
    set_logger,
)


def main():
  model_name = "unilm_unilm"
  train_path = PROJECT_DIR / "data/cgec/train.txt"
  eval_path = PROJECT_DIR / "data/cgec/val.txt"
  hyparam_file = Path(__file__).resolve().parent / "hyparams.json"

  logger = set_logger()
  parser = set_train_parser()
  update_parser_defaults(hyparam_file, parser)
  args = parser.parse_args()
  args.logger = logger
  args.dataloader_drop_last = True

  # if args.output_dir is None:
  if args.trainset_ratio < 1.0:
    train_stage = "preselect"
  else:

    train_stage = "formal"

  args.output_dir = PROJECT_DIR / "output" / model_name / train_stage
  os.makedirs(args.output_dir, exist_ok=True)

  args.log_path = args.output_dir / "{}.log".format(train_stage)

  vocab_file = PROJECT_DIR / "vocabs" / "vocab.txt"
  encoder_tokenizer = create_tokenizer(
      vocab_path=vocab_file,
      tokenizer_type="bert",
      use_custom_sos_eos=False,
  )
  decoder_tokenizer = create_tokenizer(
      vocab_path=vocab_file,
      tokenizer_type="bert",
      use_custom_sos_eos=True,
  )

  train_dataset, eval_dataset = create_datasets(
      project_dir=PROJECT_DIR,
      train_path=train_path,
      eval_path=eval_path,
      encoder_tokenizer=encoder_tokenizer,
      decoder_tokenizer=decoder_tokenizer,
      args=args,
  )

  logger.info("model name: {}".format(model_name))
  logger.info("train stage: {}".format(train_stage))
  recover_step = get_latest_epoch_checkpoint(args.output_dir)
  args.recover_step = recover_step

  if recover_step is None:
    logger.info("initializing model.")
    encoder_decoder_model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        "peterchou/unilm-chinese-base",
        "peterchou/unilm-chinese-base",
        tie_encoder_decoder=False,
    )
  else:
    logger.info("recover model of epoch {}.".format(recover_step))
    encoder_decoder_model = EncoderDecoderModel.from_pretrained(
        os.path.join(args.output_dir, "epoch-{}".format(recover_step)))

  logger.info("*" * 50 + "  training arguments  " + "*" * 50)
  for arg in vars(args):
    logger.info(' {}: {}'.format(arg, getattr(args, arg) or ''))

  logger.info("*" * 50 + "  start training  " + "*" * 50)

  trainer = Trainer(
      encoder_decoder_model=encoder_decoder_model,
      args=args,
      data_collator=collate_cgec_batch_fn,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
  )

  trainer.train()


if __name__ == '__main__':
  main()
