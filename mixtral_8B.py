import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
CACHE_DIR = 'path/to/shared_models'
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

custom_vocab = ["Yes", "No"]
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = CACHE_DIR, load_in_4bit=True, vocabulary=custom_vocab,
                                          vocab_size=len(custom_vocab))

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", cache_dir = CACHE_DIR, load_in_4bit=True)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# read in prompts data
import pandas as pd
csv_path = 'path/to/mia_prompts.csv'
df = pd.read_csv(csv_path)

prompts = df['prompt'].head(100)

counter = 1 
for prompt in prompts:
    user_entry = dict(role="user", content=prompt) 
    input_ids = tokenizer.apply_chat_template([user_entry], return_tensors="pt").to("cuda") 

    print(f"Prompt {counter}: {prompt}")
    counter += 1
    
    print("Mixtral: ", end="")
    result = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        streamer=streamer,
        do_sample=False,
        temperature=0.0,
        max_new_tokens=1,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True
        )



