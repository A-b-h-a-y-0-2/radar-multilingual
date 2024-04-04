import transformers
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch.nn.functional as F


datawiki_de = load_dataset('wikipedia','20220301.de')

human_de = []
for i in range(1000):
    human_de.append(datawiki_de['train'][i]['text'])

print("human dataset gathered")


tokenizer = transformers.AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.1")
model = transformers.AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.1")
model.eval()
model.to(torch.device('cuda'))


tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
tokenizer.truncation_side = 'right'
instruction="Please complete the given text in the german language : "

ai_text =[]
count = 0
for item in human_de:
    prefix_input_ids=tokenizer([f"{instruction} {item}"],max_length=30,padding='max_length',truncation=True,return_tensors="pt")
    prefix_input_ids={k:v.to("cuda") for k,v in prefix_input_ids.items()}
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
        item.replace("Please complete the given text in the french language : ","") for item in output_text
    ]
    ai_text.append(AI_texts)
    count+=1
    print(count)
print('ai text done')

with open('ai_de.txt', 'w') as f:
    for item in ai_text:
        f.write("%s\n" % item[0])
