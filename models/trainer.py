# -*- coding: utf-8 -*-

import glob
import math
import os
import random
from argparse import Namespace

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import EncoderDecoderModel
from transformers.optimization import AdamW, get_scheduler
from transformers.trainer_pt_utils import LabelSmoother


class Trainer:

  def __init__(
      self,
      encoder_decoder_model: EncoderDecoderModel,
      args: Namespace,
      data_collator=None,
      train_dataset=None,
      eval_dataset=None,
  ):
    self.args = args
    self.logger = args.logger
    self.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    self.loss_fn = LabelSmoother(
        epsilon=args.label_smoothing_factor,
        ignore_index=-100,
    )
    self.eval_loss_fn = CrossEntropyLoss(ignore_index=-100)

    self.encoder_decoder_model = encoder_decoder_model
    self.data_collator = data_collator
    self.train_dataset = train_dataset
    self.eval_dataset = eval_dataset

    self.train_batch_size = int(args.train_batch_size /
                                args.gradient_accumulation_steps)

    self.train_dataloader = self.get_train_dataloader()
    self.eval_dataloader = self.get_eval_dataloader()

    self.t_total = int(
        len(self.train_dataloader) * args.num_train_epochs /
        args.gradient_accumulation_steps)
    self.warmup_steps = math.ceil(self.t_total * args.warmup_ratio)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    self.is_in_train = False

  def get_train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size=self.train_batch_size,
        shuffle=True,
        collate_fn=self.data_collator,
        drop_last=self.args.dataloader_drop_last,
        num_workers=self.args.dataloader_num_workers,
    )

  def get_eval_dataloader(self):
    return DataLoader(
        self.eval_dataset,
        batch_size=self.args.eval_batch_size,
        shuffle=False,
        collate_fn=self.data_collator,
        drop_last=self.args.dataloader_drop_last,
        num_workers=self.args.dataloader_num_workers,
    )

  def load_train_states(self, recover_step=None):
    global_step = 1

    self.optimizer, self.scheduler = self.create_optimizer_and_scheduler(
        self.t_total)

    if self.args.fp16:
      try:
        from apex import amp
      except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
        )
      self.encoder_decoder_model, self.optimizer = amp.initialize(
          self.encoder_decoder_model,
          self.optimizer,
          opt_level=self.args.fp16_opt_level)

    if recover_step:
      global_step = math.floor(recover_step * self.t_total /
                               self.args.num_train_epochs)

      self.logger.info("***** Recover optimizer: %d *****", recover_step)
      optim_recover = torch.load(os.path.join(self.args.output_dir,
                                              "epoch-{}".format(recover_step),
                                              "optim.bin"),
                                 map_location='cpu')
      if hasattr(optim_recover, 'state_dict'):
        optim_recover = optim_recover.state_dict()
      self.optimizer.load_state_dict(optim_recover)
      if self.args.fp16:
        self.logger.info("***** Recover amp: %d *****", recover_step)
        amp_recover = torch.load(os.path.join(self.args.output_dir,
                                              "epoch-{}".format(recover_step),
                                              "amp.bin"),
                                 map_location='cpu')
        amp.load_state_dict(amp_recover)

      self.logger.info("***** Recover scheduler: %d *****", recover_step)
      scheduler_recover = torch.load(os.path.join(
          self.args.output_dir, "epoch-{}".format(recover_step), "sched.bin"),
                                     map_location='cpu')
      self.scheduler.load_state_dict(scheduler_recover)

    return global_step

  def create_optimizer_and_scheduler(self, num_training_steps: int):
    """
      Setup the optimizer and the learning rate scheduler.
      We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
      Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
      """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in self.encoder_decoder_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": self.args.weight_decay,
        },
        {
            "params": [
                p for n, p in self.encoder_decoder_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer_cls = AdamW
    optimizer_kwargs = {
        "betas": (self.args.adam_beta1, self.args.adam_beta2),
        "eps": self.args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = self.args.learning_rate

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    scheduler = get_scheduler(
        self.args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=self.warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler

  def train(self, resume_from_checkpoint=True):
    self.is_in_train = True

    recover_step = self.args.recover_step

    self.encoder_decoder_model = self.encoder_decoder_model.to(self.device)

    global_step = self.load_train_states(recover_step)

    self.logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    self.encoder_decoder_model.zero_grad()

    self.logger.info("***** Running training *****")
    self.logger.info("  forward batch size = %d", self.train_batch_size)
    self.logger.info("  backward batch size = {:d}".format(
        self.train_batch_size * self.args.gradient_accumulation_steps))
    self.logger.info("  Num forward steps = {}".format(
        len(self.train_dataloader) * self.args.num_train_epochs))
    self.logger.info("  Num backward steps = {} (warmup steps: {})".format(
        self.t_total, self.warmup_steps))

    gradient_accumulation_loss = 0
    start_epoch = recover_step + 1 if recover_step is not None else 1

    for i_epoch in range(start_epoch, self.args.num_train_epochs + 1):
      for step, batch in enumerate(self.train_dataloader):
        loss_tuple = self.training_step(inputs=batch)
        gradient_accumulation_loss += loss_tuple[0]

        if (step + 1) % self.args.gradient_accumulation_steps == 0:
          self.optimizer.step()
          self.scheduler.step()  # Update learning rate schedule
          self.optimizer.zero_grad()
          global_step += 1

          if global_step % self.args.logging_steps == 0:
            self.logger.info(
                "epoch: {:.2f}  step: {},  lr: {:.4E} \t loss: {:.6f}".format(
                    global_step / self.t_total * self.args.num_train_epochs,
                    global_step, self.optimizer.param_groups[0]['lr'],
                    gradient_accumulation_loss))
          gradient_accumulation_loss = 0
      self.save_model_states(i_epoch)

      self.evaluate(i_epoch)

      self.logger.info("***** CUDA.empty_cache() *****")
      torch.cuda.empty_cache()

    self.is_in_train = False

  def evaluate(self, i_epoch):
    self.logger.info("evaluation start.")
    self.encoder_decoder_model.eval()

    num_eval = len(self.eval_dataloader)
    total_perplexity = 0

    with torch.no_grad():
      for step, inputs in enumerate(self.eval_dataloader):
        inputs = prepare_inputs(inputs, self.device)
        outputs = self.encoder_decoder_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
        )

        gec_logits = outputs.logits
        # we are doing next-token prediction; shift prediction scores and input ids by one
        gec_logits = gec_logits[:, :-1, :].contiguous().to(self.device)
        gec_labels = inputs["labels"]
        gec_labels = gec_labels[:, 1:].contiguous()

        gec_loss = self.eval_loss_fn(
            gec_logits.view(-1, self.args.target_vocab_size),
            gec_labels.view(-1))

        perplexity = torch.exp(gec_loss).item()
        total_perplexity += perplexity
    self.logger.info("*" * 100)
    self.logger.info("epoch:{}, perplexity = {}".format(
        i_epoch, total_perplexity / num_eval))
    self.logger.info("*" * 100)

  def save_model_states(self, i_epoch):
    self.logger.info("** ** * Saving fine-tuned model and optimizer ** ** * ")
    self.encoder_decoder_model.save_pretrained(
        os.path.join(os.fspath(self.args.output_dir),
                     "epoch-{}".format(i_epoch)))

    output_optim_file = os.path.join(self.args.output_dir,
                                     "epoch-{}".format(i_epoch), "optim.bin")

    self.logger.info("optimizer params saved in {}".format(output_optim_file))
    torch.save(self.optimizer.state_dict(), output_optim_file)

    if self.args.fp16:
      output_amp_file = os.path.join(self.args.output_dir,
                                     "epoch-{}".format(i_epoch), "amp.bin")

      self.logger.info("amp saved in {}".format(output_amp_file))
      torch.save(amp.state_dict(), output_amp_file)

    output_sched_file = os.path.join(self.args.output_dir,
                                     "epoch-{}".format(i_epoch), "sched.bin")

    self.logger.info("scheduler saved in {}".format(output_sched_file))
    torch.save(self.scheduler.state_dict(), output_sched_file)

  def training_step(self, inputs):
    self.encoder_decoder_model.train()

    inputs = prepare_inputs(inputs, self.device)

    outputs = self.encoder_decoder_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        decoder_input_ids=inputs["decoder_input_ids"],
        decoder_attention_mask=inputs["decoder_attention_mask"],
    )
    gec_logits = outputs.logits

    # we are doing next-token prediction; shift prediction scores and input ids by one
    gec_logits = gec_logits[:, :-1, :].contiguous().to(self.device)

    gec_labels = inputs["labels"]
    gec_labels = gec_labels[:, 1:].contiguous()
    gec_loss = self.loss_fn(
        {"logits": gec_logits.view(-1, self.args.target_vocab_size)},
        gec_labels.view(-1))

    if self.args.gradient_accumulation_steps > 1:
      gec_loss = gec_loss / self.args.gradient_accumulation_steps

    if self.args.fp16:
      with amp.scale_loss(gec_loss, self.optimizer) as scaled_loss:
        scaled_loss.backward()
      torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer),
                                     self.args.max_grad_norm)
    else:
      gec_loss.backward()
      torch.nn.utils.clip_grad_norm_(self.encoder_decoder_model.parameters(),
                                     self.args.max_grad_norm)

    return (gec_loss.item(),)


def get_latest_epoch_checkpoint(output_dir):
  fn_model_list = glob.glob(os.path.join(output_dir, "epoch-*"))
  if not fn_model_list:
    return None
  model_ckpts = [int(fn.split('-')[-1]) for fn in fn_model_list]
  if model_ckpts:
    return max(model_ckpts)
  return None


def prepare_inputs(inputs, device):
  return {k: v.to(device) for k, v in inputs.items()}
