import torch
import pandas as pd
from torch.utils.data import Dataset

def preprocess_data(path, dataset, tokenizer):
    df = pd.read_csv(path / f'{dataset}.csv')
    responses = df['response'].tolist()
    responses_output = tokenizer(responses, padding='max_length',
                                 truncation=True, max_length=100)
    data_dict = {'ids': df['id'].tolist(),
                 'responses_input_ids': responses_output.input_ids,
                 'responses_attention_mask': responses_output.attention_mask,
                 'labels': df[[column for column in df.columns if 'label' in column][0]].tolist()}
    return pd.DataFrame(data_dict)

def get_labels(df):
    labels = df['labels'].unique()
    labels_to_ids = {label:index for index, label in enumerate(labels)}
    ids_to_labels = {index:label for label, index in labels_to_ids.items()}
    return labels_to_ids, ids_to_labels

class LLMHallucinationDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        data = {'responses_input_ids': torch.tensor(self.df['responses_input_ids'].iloc[index]),
                'responses_attention_mask': torch.tensor(self.df['responses_attention_mask'].iloc[index]),
                'labels': self.df['labels'].iloc[index]}
        return data