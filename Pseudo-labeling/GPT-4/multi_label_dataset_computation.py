from dotenv import load_dotenv
import json
import os
from openai import OpenAI
import pandas as pd
import argparse


load_dotenv()

countries = ['Iraq', 'Egypt', 'Morocco', 'Libya', 'UAE', 'Saudi_Arabia',
             'Bahrain', 'Syria', 'Lebanon', 'Oman', 'Palestine', 'Algeria',
             'Jordan', 'Tunisia', 'Kuwait', 'Yemen', 'Sudan', 'Qatar']

path = "../NADI2024_subtask1/subtask1/train/NADIcombined_cleaned_MULTI_LABEL.csv"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_dialect_classification(tweet):
    prompt = f"""You are a native Arabic Speaker and Linguist with a vast knowledge of Arabic dialectal constituents. We have a sentence and we want to know if it's possibly within one of the following dialects: 'Iraq, Egypt, Morocco, Libya, UAE, Saudi_Arabia, Bahrain, Syria, Lebanon, Oman, Palestine, Algeria, Jordan, Tunisia, Kuwait, Yemen, Sudan, Qatar'. 

Run the prompt independently for every dialect of the listed above and return a JSON object with 0 or 1 for every dialect. Return nothing else but the JSON object.

Input sentence: {tweet}"""

    try:
        stream = client.chat.completions.create(
            model="gpt-4",  
            messages=[
                {"role": "system", "content": "You are a native Arabic speaker and dialect expert."},
                {"role": "user", "content": prompt}
            ],
            stream=True  
        )

        result_content = ""
        for chunk in stream:
            content = getattr(chunk.choices[0].delta, 'content', None)
            if content is not None:
                result_content += content
        
        # it differs from gpt-4o and gpt4 so make sure to print first and check
        result = json.loads(result_content)
        return result

    except Exception as e:
        print(f"Error processing tweet: {tweet}. Error: {e}")
        return None
    

def process_tweets_in_batches(path, batch_size, save_threshold):
    chunk_size = batch_size
    rows_processed = 0
    processed = 0
    i = 0
    rows_to_update = []

    for chunk in pd.read_csv(path, chunksize=chunk_size):
        print("In Batch: ", i)
        for idx, row in chunk.iterrows():
            if row['Computed'] == 'no':  # Process only if not already computed
                result = get_dialect_classification(row['tweet'])
                print("Processed tweet: ", row['tweet'])
                processed += 1
                if result: 
                    for country in countries:
                        chunk.at[idx, country] = result.get(country, 0)
                    chunk.at[idx, 'Computed'] = 'yes'
                    rows_to_update.append(row['id'])
                    if len(rows_to_update) >= save_threshold:
                        df_original = pd.read_csv(path)  
                        df_original.update(chunk[chunk['id'].isin(rows_to_update)])  
                        df_original.to_csv(path, index=False) 
                        rows_processed += len(rows_to_update)
                        print(f"Saved after processing {rows_processed} updated rows.")
                        rows_to_update = []
    
        if rows_processed >= 2000:
            break

        # experimental purposes
        if len(rows_to_update) > 0:
            print(f"Processed {len(rows_to_update)} rows.")
        i += 1
    
    print("All batches processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tweets in batches from a CSV file.")

    parser.add_argument('--path', type=str, default="../NADI2024_subtask1/subtask1/train/NADIcombined_cleaned_MULTI_LABEL.csv", help='The path to the CSV file.')
    parser.add_argument('batch_size', type=int, help='The number of rows to process in each batch.')
    parser.add_argument('save_threshold', type=int, help='The number of rows to update before saving.')

    args = parser.parse_args()

    process_tweets_in_batches(args.path, args.batch_size, args.save_threshold)