from huggingface_hub import login
login(token='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')
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
device = torch.device("cuda:2")
model.eval()
model.to(device)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
tokenizer.truncation_side = 'right'
instruction="Please complete the given text in the German language : "
from datasets import load_dataset


datawiki_de  = load_dataset('wikipedia', '20220301.de')
human_de = []
for i in range(1000):
    human_de.append(str(datawiki_de['train'][i]['text']))
ai_text_de =[]
count = 0
for item in human_de:
    prefix_input_ids=tokenizer([f"{instruction} {item}"],max_length=30,padding='max_length',truncation=True,return_tensors="pt")
    prefix_input_ids={k:v.to("cuda:2") for k,v in prefix_input_ids.items()}
    outputs = model.generate(
        **prefix_input_ids,
        max_new_tokens = 512,
        do_sample = True,
        temperature = 0.6,
        top_p = 0.9,
        pad_token_id=tokenizer.pad_token_id
    )
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    AI_texts=[
        item.replace("Please complete the given text in the German language : ","") for item in output_text
    ]
    ai_text_de.append(AI_texts)
    count+=1
    print(count)
    
file_fr = "llama_ai_de.txt"
with open(file_fr,'a+') as f:
    for text in ai_text_de:
        f.write('start : '+str(text)+'\n')
