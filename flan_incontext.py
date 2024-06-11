import numpy as np
import pandas as pd
import random
import csv 
import os

 #read in leuven and flan data
leuven_data = pd.read_csv('/mnt/dv/wid/projects3/Rogers-muri-human-ai/mia/data/leuven_data.csv')
flan_data = pd.read_csv('/mnt/dv/wid/projects3/Rogers-muri-human-ai/mia/data/flan_prompts.csv')

# Generates true and false examples from leuven data for the FLAN model to answer
def generate_examples():
    #Counters for true and false examples
    true_count = 0
    false_count = 0

    #store three false and three true examples from leuven data in context_examples
    context_examples = ''

    while true_count < 3 or false_count < 3:
        # Get random row index and column index from leuven data 
        random_row_index = random.randint(1, leuven_data.shape[0] - 1)
        random_column_index = random.randint(1, leuven_data.shape[1] - 1)  # Exclude the 'item' column

        # Get the row and column name (property name) from the random column (excluding the first column)
        row_name = leuven_data.iloc[random_row_index, 0]
        column_name = leuven_data.columns[random_column_index]

        # Get the value from the random row and column and convert 1/0 value to True/False
        value = leuven_data.iloc[random_row_index, random_column_index]
        value = (lambda val: 'True' if val == 1 else 'False')(value)

        # create question for FLAN to answer
        question = f'Q: Is the property [{row_name}] true for the concept [{column_name}]?\nA: {value}\n'

        if value == 'True' and true_count < 3:
            context_examples += question
            true_count += 1
        elif value == 'False' and false_count < 3:
            context_examples += question
            false_count += 1

    return context_examples

#Using the generate_prompt function to generate a prompt for the FLAN model

from transformers import T5Tokenizer, T5ForConditionalGeneration

CACHE_DIR = '/mnt/dv/wid/projects3/Rogers-muri-human-ai/mia'

tokenizer = T5Tokenizer.from_pretrained("philschmid/flan-t5-xxl-sharded-fp16", cache_dir=CACHE_DIR)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", cache_dir=CACHE_DIR)

 # Function to write a row to a CSV file
def write_to_csv(filename, row):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)

filename = 'flan_incontext_underscore.csv'

header = ['FLAN Answer', 'FLAN Query', 'Example 1', 'Example 2', 'Example 3', 'Example 4', 'Example 5', 'Example 6']
write_to_csv(filename, header)

for i in range(len(flan_data)):
    underscore = '_'.join(flan_data['feature'][i].split())
    query = "Q: Is the property [" + underscore + "] true for the concept [" + flan_data['stimulus'][i] + "]?\n"
    input_text = generate_examples() + query
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(input_ids, max_new_tokens=5)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = decoded_output.split("A: ")[-1].strip()

    cols = input_text.strip().split('\n')
    row = [answer, cols[-1]] + [' '.join(cols[0:2])] + [' '.join(cols[2:4])] + [' '.join(cols[4:6])] + [' '.join(cols[6:8])] + [' '.join(cols[8:10])] + [' '.join(cols[10:12])]
    write_to_csv(filename, row)

