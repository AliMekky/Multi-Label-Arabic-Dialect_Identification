import pandas as pd
from camel_tools.utils.dediac import AR_DIAC_CHARSET
import os 
import re
import  nltk
from nltk.corpus import stopwords
import pyarabic.araby as araby
import string
from sklearn import preprocessing
arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations = arabic_punctuations + english_punctuations
diacritics_pattern = f"[{''.join(AR_DIAC_CHARSET)}]"
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002500-\U00002BEF"  # chinese char
                           u"\U00002702-\U000027B0"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           "]+", flags=re.UNICODE)

def remove_at_https(df):
    # Remove 'https' from the 2021 df
    df['#2_content'] = df['#2_content'].str.replace(r'https', '', regex=True)
    
    # Remove '@' from the 2021 df
    df['#2_content'] = df['#2_content'].str.replace(r'@', '', regex=True)
    
    return df

def remove_stopwords(text):
    stop = stopwords.words('arabic')
    return " ".join(word for word in text.split() if word not in stop)

def cleaning(text):
    # Remove diacritics
    text = re.sub(diacritics_pattern, '', text)
    text = araby.strip_diacritics(text)
    text = araby.strip_shadda(text)
    text = araby.strip_tashkeel(text)
    # Remove repeated letters
    text = text.strip()
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)

    # This removes repeated characters (NOT USED AS IT MAY REMOVE IMPORT 7OROF GAR EX: للارا would be just لارا so this changed the meaning)
    # text = re.sub(r'(.)\1+', r'\1', text)

    text = re.sub(r'(ا{2,})', 'ا', text)  # Replace 2 or more 'ا' with a single 'ا'
    text = re.sub(r'(و{2,})', 'و', text)  # Replace 2 or more 'و' with a single 'و'
    text = re.sub(r'(ي{2,})', 'ي', text)  # Replace 2 or more 'ي' with a single 'ي'
    text = re.sub(r'(ء{2,})', 'ء', text)  # Replace 2 or more 'ء' with a single 'ء'

    # Remove punctuation using string class
    remove_punct = str.maketrans('', '', punctuations)
    text = text.translate(remove_punct)

    # Remove rest of punctuation
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'ஜ', ' ', text)

    # Remove emojis
    text = emoji_pattern.sub(r'', text)

    # This regex matches a sequence where Arabic and English characters are adjacent without a space
    text = re.sub(r'([\u0600-\u06FF])([A-Za-z])|([A-Za-z])([\u0600-\u06FF])', r'\1 \2\3 \4', text)

    # Replace https with URL and @ with USER
    text = re.sub(r'https?://\S+', 'URL', text)  # Replace URLs (http and https)
    text = re.sub(r'@\w+', 'USER', text)         # Replace @usernames

    # Remove English letters, numbers, but preserve 'URL' and 'USER'
    # The negative lookahead ensures 'URL' and 'USER' are not removed
    text = re.sub(r'\b(?!URL\b)(?!USER\b)[A-Za-z0-9_]+\b|[^\w\s]|#', ' ', text)

    # Remove escape sequences
    text = re.sub(r'\\u[A-Za-z0-9\\]+', ' ', text)
    text = remove_stopwords(text)
    return text


def final_eliminations(df, column_name = '#2_content'):
    df.loc[:, column_name] = df[column_name].apply(cleaning)
    return df



# Load the TSV file into a DataFrame
df_2021 = pd.read_csv("/home/ali.mekky/Documents/NLP/Project/NADI2024/subtask1/train/NADI2021-TWT.tsv", sep='\t')

# Clean the DataFrame
df_cleaned = remove_at_https(df_2021)

# Save the cleaned DataFrame to a new TSV file
df_cleaned.to_csv("/home/ali.mekky/Documents/NLP/Project/NADI2024/subtask1/train/NADI2021-TWT_cleaned.tsv", sep='\t', index=False)


# Combine the 3 datasets for training
df_1 = pd.read_csv("/home/ali.mekky/Documents/NLP/Project/NADI2024/subtask1/train/NADI2020-TWT.tsv", sep='\t')
df_2 = pd.read_csv("/home/ali.mekky/Documents/NLP/Project/NADI2024/subtask1/train/NADI2021-TWT_cleaned.tsv", sep='\t')
df_3 = pd.read_csv("/home/ali.mekky/Documents/NLP/Project/NADI2024/subtask1/train/NADI2023-TWT.tsv", sep='\t')
dataset = pd.concat([df_1, df_2, df_3], ignore_index=True)
dataset = final_eliminations(dataset)

# Save the final DataFrame to a new TSV file
dataset.to_csv("/home/ali.mekky/Documents/NLP/Project/NADI2024/subtask1/train/NADIcombined_cleaned.tsv", sep='\t', index=False)


