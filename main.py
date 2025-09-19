import os
import torch
import random
import pathlib
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModel
from torch.utils.data.distributed import DistributedSampler

from args import get_arguments
from data_setup import preprocess_data, get_labels, LLMHallucinationDataset

def setup_distributed(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'  # Set the master address
    os.environ['MASTER_PORT'] = '12355'      # Set the master port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Destroy the process group."""
    dist.destroy_process_group()

def run(rank, world_size):
    """Main function for distributed training."""
    setup_distributed(rank, world_size)

    args = get_arguments()

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    data_path = pathlib.Path(args.DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    train_df = preprocess_data(data_path, 'train', tokenizer)
    dev_df = preprocess_data(data_path, 'dev', tokenizer)

    labels_to_ids, ids_to_labels = get_labels(train_df)
    train_df['labels'] = train_df['labels'].map(labels_to_ids).fillna(0).astype(int)
    dev_df['labels'] = dev_df['labels'].map(labels_to_ids).fillna(0).astype(int)

    train_data = LLMHallucinationDataset(train_df)
    dev_data = LLMHallucinationDataset(dev_df)

    print(rank)

    # Synchronize processes after directory creation
    cleanup_distributed()

def main():
    # Get number of GPUs
    world_size = torch.cuda.device_count()
    # Spawn processes
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()