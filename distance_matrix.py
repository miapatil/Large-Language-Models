import numpy as np
import pandas as pd
import pickle
import random
import fasttext
from scipy.spatial.distance import cdist
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Step 1: Load and preprocess the Leuven data and the Tyler data
leuven_data_path = '/mnt/dv/wid/projects3/Rogers-muri-human-ai/mia/data/leuven_data.csv'  # Update this to your actual path
leuven_df = pd.read_csv(leuven_data_path)
leuven_concepts = leuven_df.columns[1:].tolist()  # Exclude the first column which is features
leuven_features_df = leuven_df.set_index('Item')

tyler_data_path = '/mnt/dv/wid/projects3/Rogers-muri-human-ai/mia/data/tyler_data_grouped_keys_mia.pkl'  # Update this to your actual path
with open(tyler_data_path, 'rb') as file:
    tyler_data = pickle.load(file)

# Extract Tyler concepts and initialize the feature vectors
tyler_concepts = list(set(item[0] for item in tyler_data))

# Step 2: Load FastText model
model_path = '/mnt/dv/wid/projects3/Rogers-muri-human-ai/mia/cc.en.300.bin'  # Update this to your actual path to the FastText binary file
fasttext_model = fasttext.load_model(model_path)

def get_fasttext_embedding(word, model):
    return model.get_word_vector(word)

# Convert concepts to embeddings using FastText
leuven_embeddings = np.array([get_fasttext_embedding(concept, fasttext_model) for concept in leuven_concepts])
tyler_embeddings = np.array([get_fasttext_embedding(concept, fasttext_model) for concept in tyler_concepts])

# Compute the distance matrix using cosine distance
distance_matrix = cdist(tyler_embeddings, leuven_embeddings, metric='cosine')

# Convert to df for better readability
distance_df = pd.DataFrame(distance_matrix, index=tyler_concepts, columns=leuven_concepts)

# Step 3: Define utility functions to find closest concepts and generate prompts

# Function to find top n closest concepts
def get_top_n_closest(concept, n=3):
    if concept in distance_df.index:
        distances = distance_df.loc[concept]
    elif concept in distance_df.columns:
        distances = distance_df[concept]
    else:
        raise ValueError(f"Concept '{concept}' not found in distance matrix.")
    
    closest_concepts = distances.nsmallest(n + 1).iloc[1:n + 1]
    return closest_concepts.index.tolist(), closest_concepts.values.tolist()

# Step 4: Create CSV structure
csv_data = {}
csv_data['Prompt'] = []

# Initialize tokenizer and model
CACHE_DIR = '/mnt/dv/wid/projects3/Rogers-muri-human-ai/mia'
tokenizer = T5Tokenizer.from_pretrained("philschmid/flan-t5-xxl-sharded-fp16", cache_dir=CACHE_DIR)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", cache_dir=CACHE_DIR)

# Process each Leuven concept
for leuven_concept in leuven_concepts:
    # Find the 3 most similar Tyler concepts
    top_closest_concepts, _ = get_top_n_closest(leuven_concept, n=3)
    
    # Create the prompt for the Leuven concept
    prompt = f"Concepts related to '{leuven_concept}' are {', '.join(top_closest_concepts)}."
    csv_data['Prompt'].append(prompt)
    
    # Add each feature and its True/False value to the csv_data
    for feature in leuven_features_df.index:
        if feature not in csv_data:
            csv_data[feature] = []
        
        query = f"Q: Is the property [{feature}] true for the concept [{leuven_concept}]? True/False?"
        input_text = prompt + "\n" + query
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(input_ids, max_new_tokens=10)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = decoded_output.replace("A: ", "").strip()
        
        # Append the answer (True/False) to the csv_data
        csv_data[feature].append(answer)

# Convert csv_data to a DataFrame and save as CSV
csv_df = pd.DataFrame(csv_data, index=leuven_concepts)
csv_df.to_csv('/mnt/dv/wid/projects3/Rogers-muri-human-ai/mia/concept_prompts.csv')

print("CSV file has been created")
