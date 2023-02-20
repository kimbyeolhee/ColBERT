import argparse
import os
import sys

from omegaconf import OmegaConf

from data_loader import load_dataset
from utils import init_logger, load_tokenizer, set_seed


path = os.path.dirname(os.path.abspath(__file__)) # /opt/ml/colbert/colbert
sys.path.append(path)

def main(args):
    config = OmegaConf.load(f"{path}/{args.config}")
    init_logger()
    set_seed(config)

    tokenizer = load_tokenizer(config)

    # load dataset
    load_dataset(config, tokenizer, "train")

    # load model

    # load trainer

    # train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, default="base_config.yaml")

    args = parser.parse_args()
    main(args)
