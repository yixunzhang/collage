from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
import torch
import torch.nn as nn
import tvm
from tvm import relay

def get_gpt2_model():
    configuration = GPT2Config(torchscript=True)
    model = GPT2Model(configuration)

    return model
    #return GPT2Model.from_pretrained('gpt2')
