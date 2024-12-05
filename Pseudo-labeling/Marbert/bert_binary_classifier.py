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
import matplotlib.pyplot as plt
from IPython.display import clear_output

from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support

def get_training_data(df, dialect):
    positive = df[df['#3_label'] == dialect].copy()
    positive['#3_label'] = 1

    negative = df[df['#3_label'] != dialect].copy()
    negative['#3_label'] = 0

    sample_size = positive.shape[0]
    sampled_negative = negative.sample(n=sample_size, random_state=42)

    training_data = pd.concat([positive, sampled_negative], axis=0).reset_index(drop=True)

    training_data = training_data[['#2_content', '#3_label']].copy()

    training_data = training_data.rename(columns={'#2_content': 'text', '#3_label': 'label'})
    train_df, val_df = train_test_split(training_data, test_size=0.1, random_state=42, shuffle=True, stratify=training_data['label'] )

    train_df = train_df.dropna(subset=['text']).reset_index(drop=True)
    val_df = val_df.dropna(subset=['text']).reset_index(drop=True)

    return train_df, val_df

class Bert_Binary_Classifier:

    def __init__(self, dataset_path, exp_num, dialect, model_name="UBC-NLP/MARBERT"):
        self.model_name = model_name
        self.exp_num = exp_num
        self.dialect = dialect
        self.dir = dialect + '_exp' + str(exp_num)

        if '.tsv' in dataset_path:
            dataset = pd.read_csv(dataset_path, sep = '\t')
        else:
            dataset = pd.read_csv(dataset_path)

        self.train_df, self.val_df = get_training_data(dataset, dialect)


        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.train_dataset = self.create_dataset(self.train_df)
        self.val_dataset = self.create_dataset(self.val_df)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()



    def create_dataset(self, df):
        """Tokenize the text and create the dataset."""
        encodings = self.tokenizer(
            df['text'].tolist(), truncation=True, padding=True, max_length=128
        )
        
        return TweetDataset(encodings, df['label'].values)

    def load_model(self):
        """Load the MarBERT model for binary classification."""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
        )
        model.to(self.device)
        return model
    
    def compute_metrics(self, pred):
        logits, labels = pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train(self):
        training_args = TrainingArguments(
            output_dir='./' + self.dir + '/results',
            num_train_epochs=20,              
            per_device_train_batch_size=16,  
            per_device_eval_batch_size=64,   
            warmup_steps=500,               
            weight_decay=0.01,              
            logging_dir='./' + self.dir + '/logs',
            logging_steps=10,
            evaluation_strategy="epoch",  
            save_strategy="epoch",         
            load_best_model_at_end=True,     
            metric_for_best_model="f1",
            greater_is_better=True,

        )
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[PlotMetricsCallback()]  
        )
        trainer.train()

        self.model.save_pretrained('./' + self.dir + '/marbert_finetuned', safe_serialization= False)
        self.tokenizer.save_pretrained('./' + self.dir + '/marbert_finetuned')

    

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

        predictions = torch.argmax(logits, dim=1).cpu().numpy()

        return predictions


    def evaluate(self, dev_path):
        if '.tsv' in dev_path:
            dev = pd.read_csv(dev_path, sep = '\t')
        else:
            dev = pd.read_csv(dev_path)

        dev_df = pd.DataFrame({
            'text': dev['sentence'],
            'label': dev[self.dialect]
        })

        dev_df = dev_df.replace({'y': 1, 'n': 0})
        dev_df = dev_df.dropna(subset=['text']).reset_index(drop=True)
        predictions = self.predict(dev_df['text'].tolist())

        accuracy = accuracy_score(dev_df['label'].values, predictions)
        print(f"Accuracy: {accuracy:.4f}")
        f1 = f1_score(dev_df['label'], predictions, average='binary')
        print(f"F1-Score: {f1:.4f}")

    
        precision, recall, _, _ = precision_recall_fscore_support(
            dev_df['label'], predictions, average='binary'
        )
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

    
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

  



class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
class CustomTrainer(Trainer):
    def save_model(self, output_dir=None, **kwargs):
        """Make tensors contiguous before saving the model."""
        if output_dir is None:
            output_dir = self.args.output_dir

       
        for param in self.model.parameters():
            param.data = param.data.contiguous()


        super().save_model(output_dir, **kwargs)



class PlotMetricsCallback(TrainerCallback):
    """Custom callback to plot training and evaluation metrics."""
    def __init__(self):
        super().__init__()
        self.train_losses = [] 
        self.eval_losses = []   
        self.epochs = []        

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called at each logging step."""
        if logs is None:
            logs = {}

       
        if "epoch" in logs and logs["epoch"] not in self.epochs:
            self.epochs.append(logs["epoch"])

       
        if "loss" in logs:
            self.train_losses.append(logs["loss"])

        
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])

    
        min_length = min(len(self.epochs), len(self.train_losses), len(self.eval_losses))

        epochs = self.epochs[:min_length]
        train_losses = self.train_losses[:min_length]
        val_losses = self.eval_losses[:min_length]

        
        clear_output(wait=True)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label='Training Loss', marker='o')
        if val_losses:
            plt.plot(epochs, val_losses, label='Validation Loss', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
