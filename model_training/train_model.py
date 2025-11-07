import os
from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer
import pickle


args = OmegaConf.load('rnn_args.yaml')
trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()
