# -*- coding: utf-8 -*-
from transformers.optimization import Adafactor, AdamW, get_scheduler


def create_optimizer_and_scheduler(model, args, num_training_steps: int):
  """
    Setup the optimizer and the learning rate scheduler.
    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
    """
  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
      {
          "params": [
              p for n, p in model.named_parameters()
              if not any(nd in n for nd in no_decay)
          ],
          "weight_decay": args.weight_decay,
      },
      {
          "params": [
              p for n, p in model.named_parameters()
              if any(nd in n for nd in no_decay)
          ],
          "weight_decay": 0.0,
      },
  ]
  optimizer_cls = AdamW
  optimizer_kwargs = {
      "betas": (args.adam_beta1, args.adam_beta2),
      "eps": args.adam_epsilon,
  }
  optimizer_kwargs["lr"] = args.learning_rate

  optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

  lr_scheduler = get_scheduler(
      args.lr_scheduler_type,
      optimizer,
      num_warmup_steps=args.warmup_steps,
      num_training_steps=num_training_steps,
  )
  return optimizer, lr_scheduler
