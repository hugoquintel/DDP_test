import os
import torch
import random
import pathlib
import numpy as np
from torch import nn
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.utils.data.distributed import DistributedSampler

from args import get_arguments
from model import ClassificationLayers
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

    random.seed(args.RANDOM_SEED)
    np.random.seed(args.RANDOM_SEED)
    torch.manual_seed(args.RANDOM_SEED)
    torch.cuda.manual_seed_all(args.RANDOM_SEED)

    data_path = pathlib.Path(args.DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    train_df = preprocess_data(data_path, 'train', tokenizer)
    dev_df = preprocess_data(data_path, 'dev', tokenizer)

    labels_to_ids, ids_to_labels = get_labels(train_df)
    train_df['labels'] = train_df['labels'].map(labels_to_ids).fillna(0).astype(int)
    dev_df['labels'] = dev_df['labels'].map(labels_to_ids).fillna(0).astype(int)

    train_data = LLMHallucinationDataset(train_df)
    dev_data = LLMHallucinationDataset(dev_df)

    no_workers = os.cpu_count()

    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=False)
    train_dataloader = DataLoader(train_data, batch_size=args.TRAIN_BATCH, pin_memory=True,
                                  num_workers=no_workers, sampler=train_sampler)
    dev_sampler = DistributedSampler(dev_data, num_replicas=world_size, rank=rank, shuffle=False)
    dev_dataloader = DataLoader(dev_data, batch_size=args.DEV_BATCH, pin_memory=True,
                                num_workers=no_workers, sampler=dev_sampler)

    plm = AutoModel.from_pretrained(args.PLM).to(rank)
    cls = ClassificationLayers(plm.config, labels_to_ids).to(rank)

    plm = nn.parallel.DistributedDataParallel(plm, device_ids=[rank])
    cls = nn.parallel.DistributedDataParallel(cls, device_ids=[rank])

    train_params = ({'params': plm.parameters(), 'lr': args.PLM_LR},
                    {'params': cls.parameters(), 'lr': args.CLS_LR})

    optimizer_map = {'ASGD': optim.ASGD, 'Adadelta': optim.Adadelta, 'Adagrad': optim.Adagrad, 'Adam': optim.Adam,
                     'AdamW': optim.AdamW, 'Adamax': optim.Adamax, 'LBFGS': optim.LBFGS, 'NAdam': optim.NAdam, 'RAdam': optim.RAdam,
                     'RMSprop': optim.RMSprop, 'Rprop': optim.Rprop,'SGD': optim.SGD, 'SparseAdam': optim.SparseAdam}
    
    optimizer = optimizer_map[args.OPTIMIZER](train_params)
    loss_function = nn.CrossEntropyLoss()

    
    
    for epoch in range(args.EPOCHS):
        train_sampler.set_epoch(epoch)
        plm.train()
        cls.train()
        loss_total = 0
        for batch_index, data in enumerate(train_dataloader):
            responses_input_ids = data['responses_input_ids'].to(rank)
            responses_attention_mask = data['responses_attention_mask'].to(rank)
            labels = data['labels'].to(rank)
            
            plm_logit = plm(input_ids=responses_input_ids, attention_mask=responses_attention_mask).last_hidden_state[:, 0, :]
            logit = cls(plm_logit)
            loss = loss_function(logit, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss

        loss_total_all = [torch.zeros_like(loss_total)] * world_size
        dist.all_gather(loss_total_all, loss_total)

        if rank == 0:
            print(f'Total loss: {sum(loss_total_all):.5f}')





        plm.eval()
        cls.eval()
        labels_dev_true, labels_dev_pred = [], []
        with torch.inference_mode():
            for batch_index, data in enumerate(dev_dataloader):
                responses_input_ids = data['responses_input_ids'].to(rank)
                responses_attention_mask = data['responses_attention_mask'].to(rank)
                labels = data['labels'].to(rank)

                plm_logit = plm(input_ids=responses_input_ids, attention_mask=responses_attention_mask).last_hidden_state[:, 0, :]
                logit = cls(plm_logit)
                labels_dev_true.extend(labels.tolist())
                labels_dev_pred.extend(logit.argmax(dim=-1).tolist())

        labels_dev_true_all = [None] * world_size
        labels_dev_pred_all = [None] * world_size
        dist.all_gather_object(labels_dev_true_all, labels_dev_true)
        dist.all_gather_object(labels_dev_pred_all, labels_dev_pred)

        print(labels_dev_true_all)

        # if rank == 0:
        #     labels_dev_true_combined = []
        #     labels_dev_pred_combined = []
        #     for true, pred in zip(labels_dev_true_all, labels_dev_pred_all):
        #         labels_dev_true_combined.extend(true)
        #         labels_dev_pred_combined.extend(pred)


        # Add average loss


        
        


            

        
        
    

    print('finish')




    # Synchronize processes after directory creation
    cleanup_distributed()

def main():
    # Get number of GPUs
    world_size = torch.cuda.device_count()
    # Spawn processes
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()