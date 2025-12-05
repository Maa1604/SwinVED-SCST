import pandas as pd
import re
from collections import Counter

def generate_vocab_file(input_files, output_file):
    """
    Reads multiple CSV files, extracts unique words from 'question' and 'answer'
    columns, converts them to lowercase, and writes a vocabulary file.
    
    Args:
        input_files (list): A list of paths to the input CSV files.
        output_file (str): The path to the output vocabulary file.
    """
    
    # 1. Define the required special tokens
    special_tokens = [
        "[CLS]",
        "[PAD]",
        "[SEP]",
        "[UNK]",
        "[MASK]"
    ]

    # Set to store all unique words
    unique_words = set()

    # 2. Process each input file
    print(f"Processing input files: {input_files}...")
    for file_path in input_files:
        try:
            # Read the CSV file. Assuming it has columns 'question' and 'answer'.
            df = pd.read_csv(file_path)
            
            # Ensure the required columns exist
            if 'question' not in df.columns or 'answer' not in df.columns:
                print(f"Warning: File {file_path} must contain 'question' and 'answer' columns. Skipping.")
                continue

            # Concatenate all text from 'question' and 'answer' columns
            all_text = pd.concat([df['question'], df['answer']]).astype(str).tolist()

            # Process the text to extract words
            for line in all_text:
                # Convert to lowercase
                line = line.lower()
                
                # Use regex to split the line into words, keeping only alphanumeric characters.
                # This pattern:
                # - [a-z0-9]+: matches one or more letters or numbers (e.g., 'word', '123')
                # - | : OR
                # - [^\w\s]+: matches one or more non-word characters that aren't whitespace (i.e., punctuation like ',', '?', '.')
                # Note: You might want to adjust this regex based on how you define a "word".
                words = re.findall(r'[a-z0-9]+|[^\w\s]+', line)
                
                # Add all extracted words to the set
                unique_words.update(words)
                
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    # 3. Prepare the final vocabulary list
    
    # Sort the unique words for consistent ordering (optional but recommended)
    sorted_words = sorted(list(unique_words))
    
    # The final vocabulary list starts with special tokens, followed by the unique words
    final_vocab = special_tokens + sorted_words
    
    # 4. Write the vocabulary to the output file
    print(f"Writing {len(final_vocab)} unique tokens to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in final_vocab:
            f.write(word + '\n')

    print(f"Successfully created vocabulary file: {output_file}")


# --- Configuration ---
input_csv_files = [
    'generated_questions_answers_test_all.csv',
    'generated_questions_answers_train_all.csv'
]
output_vocab_file = 'vocab-mimic-cxr-vqa.tgt'

# --- Run the function ---
# Ensure you have the 'pandas' library installed: pip install pandas
if __name__ == "__main__":
    generate_vocab_file(input_csv_files, output_vocab_file)