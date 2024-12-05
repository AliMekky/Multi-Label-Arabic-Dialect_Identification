import pandas as pd


countries = ['Iraq', 'Egypt', 'Morocco', 'Libya', 'UAE', 'Saudi_Arabia',
             'Bahrain', 'Syria', 'Lebanon', 'Oman', 'Palestine', 'Algeria',
             'Jordan', 'Tunisia', 'Kuwait', 'Yemen', 'Sudan', 'Qatar']

path = "../NADI2024_subtask1/subtask1/train/NADIcombined_cleaned.tsv"
train_dataset = pd.read_csv(path, sep='\t')

train_dataset.rename(columns={'#2_content': 'tweet', '#3_label': 'label', '#1_id':'id'}, inplace=True)
train_dataset.drop(columns=['#4_province_label', 'id'], inplace=True)
train_dataset = train_dataset.reset_index().rename(columns={'index': 'id'})

df_one_hot = pd.get_dummies(train_dataset['label'], prefix='', prefix_sep='').astype(int)
df_final = pd.concat([train_dataset.drop('label', axis=1), df_one_hot], axis=1)
df_final.to_csv("../NADI2024_subtask1/subtask1/train/NADIcombined_cleaned_ONE_HOT_SINGLE_LABEL.tsv", index=False)

df = pd.read_csv("../NADI2024_subtask1/subtask1/train/NADIcombined_cleaned_ONE_HOT_SINGLE_LABEL.tsv")
if 'Computed' not in df.columns:
    df['Computed'] = 'no'

df.to_csv("../NADI2024_subtask1/subtask1/train/NADIcombined_cleaned_MULTI_LABEL.csv", index=False)
path = "../NADI2024_subtask1/subtask1/train/NADIcombined_cleaned_MULTI_LABEL.csv"


