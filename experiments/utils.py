# -*- coding: utf-8 -*-

import csv
import os
import glob
import torch
import argparse
import logging
import inspect
import json

import transformers

from pathlib import Path
from typing import List, Iterable
from tqdm import tqdm
from transformers.tokenization_utils_base import to_py_obj

from models.tokenizer import CgecTokenizer
from models.wobert_tokenizer import WoBertTokenizer
from models.error_maker import ErrorMaker
from models.dataset import CgecDataset

logger = logging.getLogger(__name__)


def load_json(file_path):
  with open(file_path, 'r') as fin:
    return json.load(fin)


def set_train_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output_dir",
      default=None,
      type=str,
      help=
      "The output directory where the model predictions and checkpoints will be written."
  )
  parser.add_argument(
      "--log_dir",
      default='',
      type=str,
      help="The output directory where the log will be written.")
  parser.add_argument('--seed',
                      type=int,
                      default=1024,
                      help="random seed for initialization")
  parser.add_argument(
      "--max_seq_length",
      default=256,
      type=int,
      help=
      "The maximum total input sequence length after WordPiece tokenization. \n"
      "Sequences longer than this will be truncated, and sequences shorter \n"
      "than this will be padded.")
  parser.add_argument(
      "--train_batch_size",
      default=32,
      type=int,
      help=
      "Total batch size for training with / without gradient accumulation steps."
  )
  parser.add_argument('--trainset_ratio',
                      default=1.0,
                      type=float,
                      help="the percentage of trainset to train the model")
  parser.add_argument("--eval_batch_size",
                      default=4,
                      type=int,
                      help="Total batch size for eval.")
  parser.add_argument(
      '--gradient_accumulation_steps',
      type=int,
      default=1,
      help=
      "Number of updates steps to accumulate before performing a backward/update pass."
  )
  parser.add_argument(
      '--fp16',
      action='store_true',
      help="Whether to use 16-bit float precision instead of 32-bit")
  parser.add_argument(
      '--fp16_opt_level',
      type=str,
      default='O1',
      help=
      "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
      "See details at https://nvidia.github.io/apex/amp.html")
  parser.add_argument('--learning_rate',
                      default=3e-05,
                      type=float,
                      help="learning rate")
  parser.add_argument("--max_grad_norm",
                      default=1.0,
                      type=float,
                      help="Max gradient norm.")
  parser.add_argument(
      '--weight_decay',
      # default=0.0,
      default=0.01,
      type=float,
      help="weight decay of adamW optimizer")
  parser.add_argument('--label_smoothing_factor',
                      default=0.1,
                      type=float,
                      help="the factor of label smoothing")
  parser.add_argument('--adam_beta1',
                      default=0.9,
                      type=float,
                      help="beta1 of adamW optimizer")
  parser.add_argument('--adam_beta2',
                      default=0.999,
                      type=float,
                      help="beta2 of adamW optimizer")
  parser.add_argument('--adam_epsilon',
                      default=1e-08,
                      type=float,
                      help="epsilon of adamW optimizer")
  parser.add_argument('--lr_scheduler_type',
                      default="linear",
                      type=str,
                      help="type of learning rate scheduler type")
  parser.add_argument('--num_train_epochs',
                      default=10,
                      type=int,
                      help="number of train epochs")
  parser.add_argument('--warmup_ratio',
                      default=0.06,
                      type=float,
                      help="warmup ratio")
  parser.add_argument('--logging_steps',
                      default=50,
                      type=int,
                      help="number of steps per logging")
  parser.add_argument('--dataloader_num_workers',
                      default=10,
                      type=int,
                      help="number of workers to preprocess data")
  parser.add_argument('--no_cuda',
                      action='store_true',
                      help="Whether not to use CUDA when available")
  parser.add_argument('--no_dynamic_mask',
                      action='store_true',
                      help="use dynamic mask on trainset")
  return parser


def set_eval_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument("--ckpt",
                      type=str,
                      required=True,
                      help="the checkpoint to load.")
  parser.add_argument("--max_seq_length",
                      type=int,
                      default=256,
                      help="the checkpoint to load.")
  parser.add_argument("--batch_size",
                      type=int,
                      default=4,
                      help="batch size of test data")
  parser.add_argument("--num_beams",
                      type=int,
                      default=12,
                      help="num_beams in beam search")
  parser.add_argument('--preselect',
                      action='store_true',
                      help="eval preselect model")
  return parser


def update_parser_defaults(f: Path,
                           parser: argparse.ArgumentParser,
                           train=True):
  if f.exists():
    data = load_json(f)
    hyparams = data.get("train") if train else data.get("eval")
    if hyparams is not None:
      parser.set_defaults(**hyparams)
      return True
  return False


def create_tokenizer(
    vocab_path: Path,
    tokenizer_type: str = "bert",
    use_custom_sos_eos: bool = False,
):
  tokenizer = None
  cls_token = "[SOS]" if use_custom_sos_eos else "[CLS]"
  sep_token = "[EOS]" if use_custom_sos_eos else "[SEP]"
  if tokenizer_type == "bert":
    tokenizer = CgecTokenizer(
        vocab_path,
        cls_token=cls_token,
        sep_token=sep_token,
    )
  elif tokenizer_type == "wobert":
    tokenizer = WoBertTokenizer(
        vocab_path,
        cls_token=cls_token,
        sep_token=sep_token,
    )
  return tokenizer


def create_datasets(
    project_dir: Path,
    train_path: Path,
    eval_path: Path,
    encoder_tokenizer: transformers.PreTrainedTokenizer,
    decoder_tokenizer: transformers.PreTrainedTokenizer,
    args: argparse.Namespace,
):
  if args.no_dynamic_mask:
    error_maker = None
    args.logger.info("use no mask on training set.")
  else:
    error_maker = ErrorMaker(
        token_freq_file=project_dir / "resources/tokens_freq.json",
        pinyin_freq_file=project_dir / "resources/pinyin_freq.json",
        text_mask_ratio=0.7,
    )
    args.logger.info("use dynamic mask on training set.")

  train_dataset = CgecDataset(
      train_path,
      encoder_tokenizer,
      decoder_tokenizer,
      max_length=args.max_seq_length,
      error_maker=error_maker,
      dataset_ratio=args.trainset_ratio,
  )
  eval_dataset = CgecDataset(
      eval_path,
      encoder_tokenizer,
      decoder_tokenizer,
      max_length=args.max_seq_length,
  )
  return train_dataset, eval_dataset


def create_dataset_and_tokenizers(
    project_dir: Path,
    train_path: str,
    eval_path: str,
    args: argparse.ArgumentParser,
    use_custom_sos_eos: bool = True,
):
  encoder_tokenizer = create_tokenizer(
      vocab_path=project_dir / "vocabs" / "vocab.txt",
      tokenizer_type="bert",
      use_custom_sos_eos=False,
  )
  decoder_tokenizer = create_tokenizer(
      vocab_path=project_dir / "vocabs" / "vocab.txt",
      tokenizer_type="bert",
      use_custom_sos_eos=use_custom_sos_eos,
  )

  train_dataset, eval_dataset = create_datasets(
      project_dir=project_dir,
      train_path=train_path,
      eval_path=eval_path,
      encoder_tokenizer=encoder_tokenizer,
      decoder_tokenizer=decoder_tokenizer,
      args=args,
  )
  return encoder_tokenizer, decoder_tokenizer, train_dataset, eval_dataset


def get_test_result(source,
                    result_path,
                    encoder_tokenizer,
                    decoder_tokenizer,
                    model,
                    max_seq_length=256,
                    num_beams=12,
                    device='cpu'):
  test_data = open(source, 'r').readlines()
  results = []

  model.eval()

  for data in tqdm(test_data):
    data = data.strip()
    inputs = encoder_tokenizer(data,
                               padding="max_length",
                               truncation=True,
                               add_special_tokens=True,
                               max_length=max_seq_length,
                               return_tensors="pt")

    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
      outputs = model.generate(
          input_ids,
          attention_mask=attention_mask,
          max_length=max_seq_length,
          num_beams=num_beams,
      )

    batch_sentences = batch_decode(outputs, decoder_tokenizer)
    for sentence in batch_sentences:
      results.append([sentence])
  save_results_to_file(results, result_path)


def batch_decode(token_ids, tokenizer):
  batch_sentences = []
  token_ids = to_py_obj(token_ids)
  for ids in token_ids:
    tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
    batch_sentences.append("".join(tokens))
  return batch_sentences


def save_results_to_file(results: Iterable[List[str]],
                         save_path: Path,
                         delimiter: str = '\t',
                         encoding: str = "utf-8") -> None:
  """save results to file

  Args:
    results (Iterable): A iterable of results
    save_path (str): the path of file which results save to
    delimiter (str): A string indicates items are separated by delimiter char
    encoding (str): encoding for saved file
  """
  with open(save_path, 'w', newline='', encoding=encoding) as fout:
    writer = csv.writer(fout, delimiter=delimiter)
    for result in results:
      writer.writerow(result)


def get_latest_epoch_checkpoint(output_dir):
  fn_model_list = glob.glob(os.path.join(output_dir, "epoch-*"))
  if not fn_model_list:
    return None
  model_ckpts = [int(fn.split('-')[-1]) for fn in fn_model_list]
  if model_ckpts:
    return max(model_ckpts)
  return None


def prepare_evaluate(args):
  stage = "preselect" if args.preselect else "formal"
  output_dir = args.project_dir / "output" / args.model_name / stage
  eval_dir = args.project_dir / "evaluation" / args.model_name / stage
  eval_dir.mkdir(exist_ok=True, parents=True)

  ckpt_dir = output_dir / args.ckpt
  output_file = eval_dir / "{}_{}_{}.txt".format(args.model_name, stage,
                                                 args.ckpt)
  return ckpt_dir, output_file


def set_logger(log_path=None,
               file_logging=False,
               file_logging_level="INFO",
               console_logging_level="INFO",
               name=None):
  """logging 设置使得同时输出到文件和console
    Args:
        log_path (str): log 文件的地址, None就不保存日志到文件
        file_logging_level (str): "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" 中一个
        console_logging_level (str): "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" 中一个
        name (str): logger的名字 当None时，默认使用module 的__name__
    Return:
        (logging object): 设置好的logger
    """

  frm = inspect.stack()[1]
  mod = inspect.getmodule(frm[0])
  if name is None:
    # use calling module's __name__
    logger = logging.getLogger(mod.__name__)
  else:
    logger = logging.getLogger(name)

  logger.setLevel(getattr(logging, console_logging_level.upper()))

  if not logger.handlers:
    if log_path is not None and len(log_path) > 0:  # log to log file
      log_dir = os.path.dirname(log_path)
      if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

      file_handler = logging.FileHandler(log_path)
      file_formatter = logging.Formatter(
          "%(asctime)s - %(levelname)s -  %(message)s", "%Y-%m-%d %H:%M:%S")
      file_handler.setFormatter(file_formatter)
      logger.addHandler(file_handler)

    # log to console
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s -  %(message)s", "%Y-%m-%d %H:%M:%S")
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(getattr(logging, console_logging_level.upper()))
    logger.addHandler(stream_handler)

  return logger
