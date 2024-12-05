import os
import numpy as np
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset


# Update the datasets module
def get_data(P):
    """
    Prepare datasets using MarBERT-compatible inputs.
    """
    tokenizer = AutoTokenizer.from_pretrained('UBC-NLP/MARBERT')
    label_map = build_label_map([P['train_file']])  # Build label map from train file
    source_data = load_nadi_data(P['train_file'], P['test_file'], label_map)
    
    split_idx = generate_split(
        len(source_data['train']['texts']),
        P['val_frac'],
        np.random.RandomState(P['split_seed'])
    )
    
    train_ds = ds_multilabel(
        source_data['train']['texts'][split_idx[0]],
        source_data['train']['labels'][split_idx[0]],
        tokenizer
    )
    val_ds = ds_multilabel(
        source_data['train']['texts'][split_idx[1]],
        source_data['train']['labels'][split_idx[1]],
        tokenizer
    )
    test_ds = ds_multilabel_test(
        source_data['test']['texts'],
        tokenizer
    )
    return {'train': train_ds, 'val': val_ds, 'test': test_ds}

# Metadata for datasets
def get_metadata(dataset_name):
    if dataset_name == 'nadi':
        meta = {
            'num_classes': 18,  # Based on the country labels
            'path_to_dataset': 'data/nadi',
            'path_to_images': None  # No images, just text
        }
    else:
        raise NotImplementedError('Metadata dictionary not implemented.')
    return meta

# Split data into train and val
def generate_split(num_ex, frac, rng):
    n_2 = int(np.round(frac * num_ex))
    n_1 = num_ex - n_2
    idx_rand = rng.permutation(num_ex)
    idx_1 = np.sort(idx_rand[:n_1])
    idx_2 = np.sort(idx_rand[-n_2:])
    return (idx_1, idx_2)

# Build label map (only for country labels)
def build_label_map(file_paths):
    label_set = set()
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'):  # Skip header
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 3:  # Ensure valid data
                    country_label = parts[2].strip()  # #3_label (country)
                    label_set.add(country_label)
    label_map = {label: idx for idx, label in enumerate(sorted(label_set))}
    return label_map

# Load NADI dataset
def load_nadi_data(train_path, test_path, label_map):
    """
    Load the NADI dataset.
    For the train dataset, load both text and country labels.
    For the test dataset, load only the text.
    """
    def load_file(file_path, is_test=False):
        texts = []
        labels = []  # Only used for training data
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if is_test == False and idx < 40770:
                    continue
                if line.startswith('sentence'):  # Skip header for test dataset
                    continue
                parts = line.strip().split('\t')
                if is_test:
                    # For test dataset, extract only the `sentence` column
                    text = parts[0].strip()  # Assuming `sentence` is the first column
                    # print(text)
                    texts.append(text)
                else:
                    # For train/validation datasets, process normally
                    if len(parts) >= 3:
                        text = parts[1].strip()  # Extract the text content
                        # print(text)
                        texts.append(text)
                        country_label = parts[2].strip()  # Extract the country label
                        label_vector = np.zeros(len(label_map), dtype=int)
                        if country_label in label_map:
                            label_vector[label_map[country_label]] = 1
                        labels.append(label_vector)

        if is_test:
            return {'texts': np.array(texts), 'labels': None}  # No labels for test
        return {'texts': np.array(texts), 'labels': np.array(labels)}
    
    train_data = load_file(train_path, is_test=False)
    test_data = load_file(test_path, is_test=True)
    return {'train': train_data, 'test': test_data}




# Multilabel dataset class
class multilabel:
    def __init__(self, P, tokenizer, label_map):
        self.train_path = P['train_file']
        self.test_path = P['test_file']
        
        # Load data
        source_data = load_nadi_data(self.train_path, self.test_path, label_map)
        
        # Generate train/val split
        split_idx = {}
        (split_idx['train'], split_idx['val']) = generate_split(
            len(source_data['train']['texts']),
            P['val_frac'],
            np.random.RandomState(P['split_seed'])
        )
        
        # Train and validation datasets
        self.train = ds_multilabel(
            source_data['train']['texts'][split_idx['train']],
            source_data['train']['labels'][split_idx['train']],
            tokenizer
        )
        self.val = ds_multilabel(
            source_data['train']['texts'][split_idx['val']],
            source_data['train']['labels'][split_idx['val']],
            tokenizer
        )
        # Test dataset (only text, no labels)
        self.test = ds_multilabel_test(
            source_data['test']['texts'],
            tokenizer
        )
        
        # Dataset lengths
        self.lengths = {'train': len(self.train), 'val': len(self.val), 'test': len(self.test)}
    
    def get_datasets(self):
        return {'train': self.train, 'val': self.val, 'test': self.test}


class ds_multilabel(Dataset):
    def __init__(self, texts, label_matrix, tokenizer):
        self.texts = texts
        self.label_matrix = label_matrix
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.label_matrix[idx]
        tokenized = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': tokenized['input_ids'].squeeze(0),  # Treat input_ids as 'image'
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'label_vec_obs': torch.FloatTensor(labels),  # Observed labels
            'label_vec_true': torch.FloatTensor(labels),  # True labels
            'idx': idx  # Index
        }

# Dataset class for test data
class ds_multilabel_test(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokenized = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'image': tokenized['input_ids'].squeeze(0),  # Treat input_ids as 'image'
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'idx': idx  # Index
        }
# Dataset class for train/validation data
# class ds_multilabel(Dataset):
#     def __init__(self, texts, label_matrix, tokenizer):
#         self.texts = texts
#         self.label_matrix = label_matrix
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         labels = self.label_matrix[idx]
#         tokenized = self.tokenizer(
#             text,
#             max_length=128,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
        
#         return {
#             'input_ids': tokenized['input_ids'].squeeze(0),
#             'attention_mask': tokenized['attention_mask'].squeeze(0),
#             'label_vec_true': torch.FloatTensor(labels)
#         }
    

# # Dataset class for test data (text only)
# class ds_multilabel_test(Dataset):
#     def __init__(self, texts, tokenizer):
#         self.texts = texts
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         tokenized = self.tokenizer(
#             text,
#             max_length=128,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
#         return {
#             'input_ids': tokenized['input_ids'].squeeze(0),
#             'attention_mask': tokenized['attention_mask'].squeeze(0),
#         }


# Get dataset based on parameters
# def get_data(P, tokenizer, label_map):
#     if P['dataset'] == 'nadi':
#         ds = multilabel(P, tokenizer, label_map).get_datasets()
#     else:
#         raise ValueError('Unknown dataset.')
#     return ds

# Example usage
P = {
    'dataset': 'nadi',
    'split_seed': 42,
    'val_frac': 0.1,
    'train_file': '/home/ali.mekky/Documents/NLP/Project/NADI2024/subtask1/train/NADIcombined_cleaned.tsv',
    'test_file': '/home/ali.mekky/Documents/NLP/Project/NADI2024/subtask1/dev/NADI2024_subtask1_dev2.tsv'
}

# Build label map
label_map = build_label_map([P['train_file']])

# Initialize MarBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('UBC-NLP/MARBERT')

datasets = {}

# Load datasets
# ds = get_data(P, tokenizer, label_map)
datasets  = get_data(P)
# # Access splits
train_ds = datasets['train']
val_ds = datasets['val']
test_ds = datasets['test']

# Print dataset stats
print(f"Number of labels: {len(label_map)}")
print(f"Number of training examples: {len(train_ds)}")
print(f"Number of validation examples: {len(val_ds)}")
print(f"Number of test examples: {len(test_ds)}")

# Print label map
print("Label map:", label_map)

# Print a sample
sample = train_ds[5000]
print("train 0")
print("Sample input_ids:", sample['image'])
print("Sample attention_mask:", sample['attention_mask'])
print("Sample label vector:", sample['label_vec_true'])

print("train 1")
sample = train_ds[1]
print("Sample input_ids:", sample['image'])
print("Sample attention_mask:", sample['attention_mask'])
print("Sample label vector:", sample['label_vec_true'])

print("train 2")
sample = train_ds[2]
print("Sample input_ids:", sample['image'])
print("Sample attention_mask:", sample['attention_mask'])
print("Sample label vector:", sample['label_vec_true'])

print("test 0")
sample = test_ds[0]
print("Sample input_ids:", sample['image'])
print("Sample attention_mask:", sample['attention_mask'])
# print("Sample label vector:", sample['label_vec_true'])

print("test 1")
sample = test_ds[1]
print("Sample input_ids:", sample['image'])
print("Sample attention_mask:", sample['attention_mask'])
# print("Sample label vector:", sample['label_vec_true'])

print("test 2")
sample = test_ds[2]
print("Sample input_ids:", sample['image'])
print("Sample attention_mask:", sample['attention_mask'])
# print("Sample label vector:", sample['label_vec_true'])



# Example DataLoader
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
for batch in train_loader:
    print("Batch input_ids:", batch['image'].shape)
    print("Batch attention_mask:", batch['attention_mask'].shape)
    print("Batch label vectors:", batch['label_vec_true'].shape)
    break

test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
for batch in test_loader:
    print("Test Batch input_ids:", batch['image'].shape)
    print("Test Batch attention_mask:", batch['attention_mask'].shape)
    break
