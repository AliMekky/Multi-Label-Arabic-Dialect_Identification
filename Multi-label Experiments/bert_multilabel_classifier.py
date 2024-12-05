import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    TrainerCallback,
)

from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support
import models
from instrumentation import compute_metrics,prob_stat,scatter1,hist1,cal
import losses

class BertTrainer:
    def __init__(self, training_dataset_path, labels, exp_num, model_name="UBC-NLP/MARBERT"):
        self.labels = labels
        self.label2id = {label: idx for idx, label in enumerate(labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.model_name = model_name
        self.exp_num = exp_num


        if '.tsv' in training_dataset_path:
            training_dataset = pd.read_csv(training_dataset_path, sep = '\t')
        else:
            training_dataset = pd.read_csv(training_dataset_path)

        self.df_reduced = self.prepare_data(training_dataset)


        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.train_df, self.val_df = self.split_data()
        self.train_dataset = self.create_dataset(self.train_df)
        self.val_dataset = self.create_dataset(self.val_df)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()

        
    def prepare_data(self, dataset):
        """Convert labels to one-hot encoding and prepare the data."""

        df_with_dummies = dataset[self.labels + ['tweet']]

        content_col = 'tweet'
        label_cols = df_with_dummies.columns.difference([content_col])

        processed_df = pd.DataFrame({
            'text': df_with_dummies[content_col],
            'label': df_with_dummies[label_cols].values.tolist()
        })

        return processed_df

    def split_data(self):
        """Split data into train and validation sets."""
        train_df, val_df = train_test_split(
            self.df_reduced, test_size=0.1, random_state=42,
        )
        train_df['text'] = train_df['text'].astype(str)
        val_df['text'] = val_df['text'].astype(str)
        return train_df, val_df

    def create_dataset(self, df):
        """Tokenize the text and create the dataset."""
        encodings = self.tokenizer(
            df['text'].tolist(), truncation=True, padding=True, max_length=128
        )
        return TweetDataset(encodings, df['label'].values)

    def load_model(self):
        """Load the MarBERT model with the correct configuration."""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id,
            problem_type="multi_label_classification"
        )
        model.to(self.device)
        return model

    def compute_metrics(self, p: EvalPrediction):
        """Compute evaluation metrics."""
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = self.multi_label_metrics(preds, p.label_ids)
        return result

    @staticmethod
    def multi_label_metrics(predictions, labels, threshold=0.5):
        """Calculate metrics for multi-label classification."""
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1

        f1 = f1_score(labels, y_pred, average='micro')
        roc_auc = roc_auc_score(labels, y_pred, average='micro')
        accuracy = accuracy_score(labels, y_pred)

        return {'f1': f1, 'roc_auc': roc_auc, 'accuracy': accuracy}
    

    def predict(self, texts):
        
        encodings = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=128, 
            return_tensors="pt"
        )

   
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

       
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        
        probabilities = torch.sigmoid(logits).cpu().numpy()

       
        predictions = (probabilities >= 0.3).astype(int)

        return predictions, probabilities

    
    def evaluate(self, dev_path):
        if '.tsv' in dev_path:
            dev = pd.read_csv(dev_path, sep = '\t')
        else:
            dev = pd.read_csv(dev_path)

        df_replaced = dev.replace({'y': 1, 'n': 0})


        country_columns = df_replaced.columns.difference(['sentence'])

        df_replaced['label'] = df_replaced[country_columns].values.tolist()

 
        df_final = df_replaced[['sentence', 'label']]

        predictions, probabilities = self.predict(df_final['sentence'].tolist())

        output_file = 'out.txt'
        with open(output_file, 'w') as f:
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
           
                pred_str = ','.join(map(str, pred))

                f.write(f'{pred_str}\n')
            

        indexes = [0, 2, 4, 10, 13, 14, 15, 17]

        predictions = [output[indexes] for output in predictions]


        subset_accuracy = accuracy_score(df_final['label'].tolist(), predictions)
        print(f"Subset Accuracy: {subset_accuracy:.4f}")

        
        hamming = hamming_loss(df_final['label'].tolist(), predictions)
        print(f"Hamming Loss: {hamming:.4f}")

        
        precision, recall, f1, _ = precision_recall_fscore_support(
            df_final['label'].tolist(), predictions, average='micro'  
        )
        print(f"Micro Precision: {precision:.4f}")
        print(f"Micro Recall: {recall:.4f}")
        print(f"Micro F1-Score: {f1:.4f}")

      
        precision_per_label, recall_per_label, f1_per_label, _ = precision_recall_fscore_support(
            df_final['label'].tolist(), predictions, average=None 
        )
        print(f"Precision per label: {precision_per_label}")
        print(f"Recall per label: {recall_per_label}")
        print(f"F1-Score per label: {f1_per_label}")
        multilabel_check = [np.sum(np.array(prediction)) for prediction in predictions]
        print(set(multilabel_check))



    def train(self):
        """Train the model using HuggingFace Trainer."""
        training_args = TrainingArguments(
            output_dir='./exp_' + str(self.exp_num) + '/results',
            num_train_epochs=100,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./exp_' + str(self.exp_num) + '/logs',
            logging_steps=500,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            fp16=True
        )

        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        trainer.train()

        self.model.save_pretrained('./exp_' + str(self.exp_num) + '/marbert_finetuned', safe_serialization= False)
        self.tokenizer.save_pretrained('./exp_' + str(self.exp_num) + '/marbert_finetuned')


class TweetDataset(torch.utils.data.Dataset):
    """Custom Dataset class for Tweet data."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class CustomTrainer(Trainer):
    """Custom Trainer class to ensure tensors are contiguous."""
    def save_model(self, output_dir=None, **kwargs):
        if output_dir is None:
            output_dir = self.args.output_dir
        for param in self.model.parameters():
            param.data = param.data.contiguous()
        super().save_model(output_dir, **kwargs)
