import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import models
from instrumentation import compute_metrics,prob_stat,scatter1,hist1,cal
import losses
import numpy as np

P = {
    "model_name": "/home/ali.mekky/Documents/NLP/Project/Cross-Country-Dialectal-Arabic-Identification/GR_LOSS_EXP1",
    "dataset_path": "/home/ali.mekky/Documents/NLP/Project/NADI2024/subtask1/train/NADIcombined_cleaned.tsv",  # Path to dataset file
    "labels": ['Algeria', 'Bahrain', 'Egypt', 'Iraq', 'Jordan', 'Kuwait',
       'Lebanon', 'Libya', 'Morocco', 'Oman', 'Palestine', 'Qatar',
       'Saudi_Arabia', 'Sudan', 'Syria', 'Tunisia', 'UAE', 'Yemen'],
    "bsize": 8,
    "num_workers": 4,
    "lr": 2e-5,
    "num_epochs": 10,
    "beta": [1.0, 2.0, 0.5, 1.0],  # Example beta parameters
    "alpha": [0.5, 0.1, 1.0, 0.5],  # Example alpha parameters
    "q2q3": [0.5, 0.5],
    "save_path": "marbert_gr_loss_model.pth",
    "num_classes": 18
}


class TweetDataset(torch.utils.data.Dataset):
    """Custom Dataset class for Tweet data."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # print(self.labels[idx])
        # item['labels'] = torch.tensor(self.labels[idx], dtype=torch.str)
        return item

    @staticmethod
    def create_dataset(df, tokenizer, max_len=128):
        """Tokenize the text and create the dataset."""
        encodings = tokenizer(
            df['text'].tolist(), truncation=True, padding=True, max_length=max_len
        )
        return TweetDataset(encodings, df['label'].values)

    @staticmethod
    def split_data(df, test_size=0.1):
        """Split data into train and validation sets."""
        train_df, val_df = train_test_split(
            df, test_size=test_size, random_state=42
        )
        train_df['text'] = train_df['text'].astype(str)
        val_df['text'] = val_df['text'].astype(str)
        return train_df, val_df

    @staticmethod
    def prepare_data(dataset, content_col="sentence", label_cols=None):
        """Convert labels to one-hot encoding and prepare the data."""
        labels = ['Algeria', 'Egypt', 'Jordan', 'Palestine', 'Sudan', 'Syria', 'Tunisia', 'Yemen']
        # df_with_dummies = pd.get_dummies(dataset, columns=['#3_label'], prefix='', prefix_sep='', dtype=int)
        # df_with_dummies = df_with_dummies.dropna(subset=['#2_content']).reset_index(drop=True)
        # df_with_dummies = df_with_dummies[labels + content_col]
        label_cols = dataset.columns.difference([content_col])

        processed_df = pd.DataFrame({
            'text': dataset[content_col],
            'label': dataset[label_cols].values.tolist()
        })
        return processed_df

def infer_model(P, dev_dataset_path, output_file="predictions.txt"):
    print("Loading and preprocessing development dataset...")
    # Load the development dataset
    if '.tsv' in dev_dataset_path:
        dev_dataset = pd.read_csv(dev_dataset_path, sep='\t')
    else:
        dev_dataset = pd.read_csv(dev_dataset_path)

    # # Prepare the development data
    processed_dev_df = TweetDataset.prepare_data(dev_dataset, content_col='sentence', label_cols=P['labels'])
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(P['model_name'])

    # Create the development dataset
    dev_dataset = TweetDataset.create_dataset(processed_dev_df, tokenizer)

    # Wrap in a DataLoader
    dev_dataloader = DataLoader(dev_dataset, batch_size=P['bsize'], shuffle=False)

    print("Loading trained model...")
    # Load the trained model
    model = AutoModelForSequenceClassification.from_pretrained(
        P['model_name'], num_labels=P['num_classes']
    )
    # model.load_state_dict(torch.load(P['save_path']))
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    print("Starting inference on development dataset...")
    all_preds = []
    # all_labels = []

    with torch.no_grad():
        for batch in tqdm(dev_dataloader, desc="Inference Progress"):
            print(batch['input_ids'])
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # labels = batch['labels'].to(device)  # True labels for evaluation

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            preds = torch.sigmoid(logits)  # Convert to probabilities
            # print(preds)
            # Collect predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            # all_labels.extend(labels.cpu().numpy())

    # Convert predictions to binary for saving (optional)
    threshold = 0.5
    binarized_preds = (np.array(all_preds) > threshold).astype(int)

    print("Saving predictions to file...")
    with open(output_file, "w") as f:
        for pred in binarized_preds:
            pred_line = ",".join(map(str, pred))  # Convert list of predictions to comma-separated string
            f.write(pred_line + "\n")

    print(f"Predictions saved to {output_file}")

    print("Evaluating performance on the development dataset...")
    # metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
    # print(f"Development mAP: {metrics['map']:.4f}")

    return all_preds


# Inference Parameters
dev_dataset_path = "/home/ali.mekky/Documents/NLP/Project/NADI2024/subtask1/dev/NADI2024_subtask1_dev2.tsv"
output_file = "dev_predictions.txt"
all_preds = infer_model(P, dev_dataset_path, output_file=output_file)

