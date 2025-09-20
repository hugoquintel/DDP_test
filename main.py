import os
import torch
import random
import pathlib
import numpy as np
from torch import nn
from torch import optim
from sklearn import metrics
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.utils.data.distributed import DistributedSampler

from args import get_arguments
from model import TransformerClassifier
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
    tokenizer = AutoTokenizer.from_pretrained(args.PLM)
    train_df = preprocess_data(args, data_path, 'train', tokenizer)
    dev_df = preprocess_data(args, data_path, 'dev', tokenizer)

    labels_to_ids, ids_to_labels = get_labels(train_df)
    train_df['labels'] = train_df['labels'].map(labels_to_ids).fillna(0).astype(int)
    dev_df['labels'] = dev_df['labels'].map(labels_to_ids).fillna(0).astype(int)

    train_data = LLMHallucinationDataset(train_df)
    dev_data = LLMHallucinationDataset(dev_df)

    no_workers = os.cpu_count()

    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_data, batch_size=args.TRAIN_BATCH, pin_memory=True,
                                  num_workers=no_workers, sampler=train_sampler)
    dev_sampler = DistributedSampler(dev_data, num_replicas=world_size, rank=rank, shuffle=False)
    dev_dataloader = DataLoader(dev_data, batch_size=args.DEV_BATCH, pin_memory=True,
                                num_workers=no_workers, sampler=dev_sampler)
    
    prompts_contexts_plm = AutoModel.from_pretrained(args.PLM).to(rank)
    responses_plm = AutoModel.from_pretrained(args.PLM).to(rank)
    cls = TransformerClassifier(prompts_contexts_plm.config, labels_to_ids).to(rank)

    prompts_contexts_plm = nn.parallel.DistributedDataParallel(prompts_contexts_plm, device_ids=[rank], grad_as_bucket_view=True)
    responses_plm = nn.parallel.DistributedDataParallel(responses_plm, device_ids=[rank], grad_as_bucket_view=True)
    cls = nn.parallel.DistributedDataParallel(cls, device_ids=[rank], grad_as_bucket_view=True)

    train_params = ({'params': prompts_contexts_plm.parameters(), 'lr': args.PLM_LR},
                    {'params': responses_plm.parameters(), 'lr': args.PLM_LR},
                    {'params': cls.parameters(), 'lr': args.CLS_LR})

    optimizer_map = {'ASGD': optim.ASGD, 'Adadelta': optim.Adadelta, 'Adagrad': optim.Adagrad, 'Adam': optim.Adam,
                     'AdamW': optim.AdamW, 'Adamax': optim.Adamax, 'LBFGS': optim.LBFGS, 'NAdam': optim.NAdam, 'RAdam': optim.RAdam,
                     'RMSprop': optim.RMSprop, 'Rprop': optim.Rprop,'SGD': optim.SGD, 'SparseAdam': optim.SparseAdam}
    
    optimizer = optimizer_map[args.OPTIMIZER](train_params)
    loss_function = nn.CrossEntropyLoss()





    train_dataloader_all = [None] * world_size
    dev_dataloader_all = [None] * world_size
    dist.all_gather_object(train_dataloader_all, train_dataloader)
    dist.all_gather_object(dev_dataloader_all, dev_dataloader)
    if rank == 0:
        train_dataloader_all = [data for data_list in train_dataloader_all for data in data_list]
        dev_dataloader_all = [data for data_list in dev_dataloader_all for data in data_list]
        print(f'Number of samples in train set: {len(train_data)}')
        print(f'Number of samples in dev set: {len(dev_data)}')
        print(f'Number of train batches: {len(train_dataloader_all)}')
        print(f'Number of dev batches: {len(dev_dataloader_all)}')
        print(f'Labels to ids: {labels_to_ids}')
        print(f'Ids to labels: {ids_to_labels}\n')
    dist.barrier()

    
    for epoch in range(args.EPOCHS):
        if rank == 0:
            print(f'Epoch {epoch}')
        train_sampler.set_epoch(epoch)
        prompts_contexts_plm.train()
        responses_plm.train()
        cls.train()
        loss_total = 0
        for batch_index, data in enumerate(train_dataloader):
            prompts_contexts_input_ids = data['prompts_contexts_input_ids'].to(rank)
            prompts_contexts_attention_mask = data['prompts_contexts_attention_mask'].to(rank)
            responses_input_ids = data['responses_input_ids'].to(rank)
            responses_attention_mask = data['responses_attention_mask'].to(rank)
            labels = data['labels'].to(rank)
            prompts_contexts_logit = prompts_contexts_plm(input_ids=prompts_contexts_input_ids, attention_mask=prompts_contexts_attention_mask).last_hidden_state
            responses_logit = responses_plm(input_ids=responses_input_ids, attention_mask=responses_attention_mask).last_hidden_state
            logit = cls(prompts_contexts_logit, responses_logit)[:, 0, :]
            loss = loss_function(logit, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss
        loss_total_all = [torch.zeros_like(loss_total)] * world_size
        dist.all_gather(loss_total_all, loss_total)
        if rank == 0:
            print(f'Total loss: {sum(loss_total_all):.5f}')


        prompts_contexts_plm.eval()
        responses_plm.eval()
        cls.eval()
        labels_dev_true, labels_dev_pred = [], []
        with torch.inference_mode():
            for batch_index, data in enumerate(dev_dataloader):
                prompts_contexts_input_ids = data['prompts_contexts_input_ids'].to(rank)
                prompts_contexts_attention_mask = data['prompts_contexts_attention_mask'].to(rank)
                responses_input_ids = data['responses_input_ids'].to(rank)
                responses_attention_mask = data['responses_attention_mask'].to(rank)
                labels = data['labels'].to(rank)
                prompts_contexts_logit = prompts_contexts_plm(input_ids=prompts_contexts_input_ids, attention_mask=prompts_contexts_attention_mask).last_hidden_state
                responses_logit = responses_plm(input_ids=responses_input_ids, attention_mask=responses_attention_mask).last_hidden_state
                logit = cls(prompts_contexts_logit, responses_logit)[:, 0, :]
                labels_dev_true.extend(labels.tolist())
                labels_dev_pred.extend(logit.argmax(dim=-1).tolist())

        labels_dev_true_all = [None] * world_size
        labels_dev_pred_all = [None] * world_size
        dist.all_gather_object(labels_dev_true_all, labels_dev_true)
        dist.all_gather_object(labels_dev_pred_all, labels_dev_pred)
        dist.barrier()
        
        if rank == 0:
            labels_dev_true_all = [label for label_list in labels_dev_true_all for label in label_list]
            labels_dev_pred_all = [label for label_list in labels_dev_pred_all for label in label_list]
            cls_report = metrics.classification_report(labels_dev_true_all, labels_dev_pred_all, target_names=labels_to_ids,
                                                       labels=tuple(labels_to_ids.values()), zero_division=0.0, digits=5)
            print(cls_report, '\n')

        dist.barrier()


        
        


            

        




    # Synchronize processes after directory creation
    cleanup_distributed()

def main():
    # Get number of GPUs
    world_size = torch.cuda.device_count()
    # Spawn processes
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()