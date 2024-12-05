import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import numpy as np
import pickle
import ast
from transformers import AutoTokenizer, AutoModel
import torch
import fasttext


class Binary_Classifier:
    def __init__(self, dataset_path, exp_num, dialect, model_name = 'svm', load_model = None):
        self.exp_num = exp_num
        self.dialect = dialect
        self.vectorizer = TfidfVectorizer(max_features=5000)  


  
        if '.tsv' in dataset_path:
            dataset = pd.read_csv(dataset_path, sep='\t')
        else:
            dataset = pd.read_csv(dataset_path)

 
        self.train_df = get_training_data(dataset, dialect)




        self.train_df = self.train_df.dropna(subset=['text']).reset_index(drop=True)
        self.vectorizer.fit(self.train_df['text'])
        self.train_df['embedding'] = list(self.vectorizer.transform(self.train_df['text']).toarray())


  
        self.X_train = np.array(self.train_df['embedding'].tolist())
        self.y_train = self.train_df['label']


        if load_model:
            try:
                self.model = pickle.load(open(load_model, 'rb'))
            except Exception as e:
                print(e)
        elif model_name == 'svm':
            self.model = SVC(kernel='linear', probability=False)
        elif model_name == 'lr':
            self.model = LogisticRegression(max_iter=500)  
        elif model_name == 'rf':
            self.model = RandomForestClassifier(n_estimators=20, random_state=42)
        elif model_name == 'ensemble':
            estimator = [] 
            estimator.append(('LR',LogisticRegression(max_iter = 200))) 
            estimator.append(('SVC', SVC(kernel='rbf', probability = False))) 
            estimator.append(('DTC', RandomForestClassifier(n_estimators=20, max_depth=10,random_state=42))) 
            self.model = VotingClassifier(estimators = estimator, voting ='hard') 
        else:
            print('Error: Choose a valid model_name')

    def get_sentence_embedding(self, text):
        words = text.split()
        embeddings = [self.fasttext_model.get_word_vector(word) for word in words]
        return np.mean(embeddings, axis=0)
    
    def train(self):
        """Train the SVM model on the training data."""
        print("Training the model...")
        self.model.fit(self.X_train, self.y_train)


    def get_arabert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
       
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return cls_embedding.flatten()  
        
    def evaluate(self, dev_path):
        """Evaluate the SVM model on the validation data."""
        print("Evaluating the model...")
        if '.tsv' in dev_path:
            dev = pd.read_csv(dev_path, sep = '\t')
        else:
            dev = pd.read_csv(dev_path)

        try:

            dev_df = pd.DataFrame({
                'text': dev['sentence'],
                'label': dev[self.dialect]
            })
            dev_df = dev_df.replace({'y': 1, 'n': 0})
            dev_df = dev_df.dropna(subset=['text']).reset_index(drop=True)

            X_test = self.vectorizer.transform(dev_df['text'])

     

            predictions = self.model.predict(X_test)
            # Calculate metrics
            accuracy = accuracy_score(dev_df['label'].tolist(), predictions)
            f1 = f1_score(dev_df['label'].tolist(), predictions, average='binary')
            precision, recall, _, _ = precision_recall_fscore_support(dev_df['label'].tolist(), predictions, average='binary')

            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")



            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        except Exception as e:
            print("Dialect not in the dev file")

    def predict(self, texts):
        """Make predictions on new texts."""
        X_texts = self.vectorizer.transform(texts)
        predictions = self.model.predict(X_texts)
        return predictions


def get_training_data(df, dialect):
    labels = ['Algeria', 'Bahrain', 'Egypt', 'Iraq', 'Jordan', 'Kuwait',
       'Lebanon', 'Libya', 'Morocco', 'Oman', 'Palestine', 'Qatar',
       'Saudi_Arabia', 'Sudan', 'Syria', 'Tunisia', 'UAE', 'Yemen']
    other_dialects = [col for col in labels if col != dialect]
    positive = df[df['#3_label'] == dialect].copy()
    positive['#3_label'] = 1
    sample_size = positive.shape[0] + 500

    negative_pool = df[(df['#3_label'] != dialect)]
    negatives = pd.DataFrame()

    per_dialect_sample_size = sample_size // len(other_dialects)
    for other_dia in other_dialects:
       
        current_negatives = negative_pool[negative_pool['#3_label'] == other_dia].sample(
            n=min(per_dialect_sample_size, negative_pool[negative_pool['#3_label'] != dialect].shape[0]),
            random_state=42,
            replace=True  
        )
        negatives = pd.concat([negatives, current_negatives], axis=0)

    negatives['#3_label'] = 0



    training_data = pd.concat([positive, negatives], axis=0).reset_index(drop=True)
    training_data = training_data[['#2_content', '#3_label', 'embedding']].copy()
    training_data = training_data[['#2_content', '#3_label', 'embedding']].copy()
    training_data = training_data.rename(columns={'#2_content': 'text', '#3_label': 'label'})


    training_data = training_data.dropna(subset=['text']).reset_index(drop=True)

    temp = pd.read_csv('/home/ali.mekky/Documents/NLP/Project/NADI2024/subtask1/multilabel/NADIcombined_cleaned_MULTI_LABEL_MODIFIED_FINAL.csv')
    temp = temp[temp['Computed'] == 'yes']
    temp = temp[['tweet'] + labels]
    content_col = 'tweet'
    label_cols = temp.columns.difference([content_col])
    temp = pd.DataFrame({
        'text': temp[content_col],
        'label': temp[label_cols].values.tolist()
    })
    temp['sum'] = [np.sum(np.array(label)) for label in temp['label']]
    temp = temp[temp['sum'] == 18]
    temp = temp[['text', 'label']]
    temp['label'] = 1

    training_data = pd.concat([training_data, temp], axis=0).reset_index(drop=True)


    return training_data

def save_model(self, path):
    pickle.dump(self.model, open(path, 'wb')) 

