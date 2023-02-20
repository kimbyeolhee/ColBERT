import logging
import random

import numpy as np
import torch
from transformers import AutoTokenizer


def set_seed(config):
    random.seed(config.utils.seed)
    np.random.seed(config.utils.seed)
    torch.manual_seed(config.utils.seed)
    if not config.utils.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.utils.seed)

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def load_tokenizer(config):
    return AutoTokenizer.from_pretrained(config.model.name_or_path)
