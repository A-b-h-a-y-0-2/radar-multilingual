import argparse
from datasets import load_dataset
import transformers
import torch
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, help='choose language to test in', choices=['French', 'German', 'Italian', 'Spanish'])
    parser.add_argument('--model', type=str, help='choose model to test with', choices=['llama', 'vicuna-7b'])
    parser.add_argument('--device', type=str, help='cuda/cpu')
    parser.add_argument('--samples', type=int, help='number of samples to generate.')
    parser.add_argument('--output_ai', type=str, help='path to the output file for ai text')
    return parser.parse_args()

def load_data(language, samples,):
    human_text = []
    if language == 'French':
        datawiki_fr = load_dataset('wikipedia', '20220301.fr')
        for i in range(samples):
            human_text.append(datawiki_fr['train'][i]['text'])
    elif language == 'German':
        datawiki_de = load_dataset('wikipedia', '20220301.de')
        for i in range(samples):
            human_text.append(datawiki_de['train'][i]['text'])
    elif language == 'Italian':
        datawiki_it = load_dataset('wikipedia', '20220301.it')
        for i in range(samples):
            human_text.append(datawiki_it['train'][i]['text'])
    else:
        print('Invalid language')
    return human_text
    

def llama(human, language, output_ai, device):
    tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    device = torch.device(device)
    model.eval()
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'right'
    instruction= "Please complete the given text in the " + language + " language : "
    ai_text = []
    for item in tqdm(human):
        prefix_input_ids=tokenizer([f"{instruction} {item}"],max_length=30,padding='max_length',truncation=True,return_tensors="pt")
        prefix_input_ids={k:v.to(device) for k,v in prefix_input_ids.items()}
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
            item.replace(instruction,"") for item in output_text
        ]
        ai_text.append(AI_texts)
    
    with open(output_ai, 'w') as f:
        for item in ai_text:
            f.write("%s\n" % item)


def vicuna(human, language, output_ai, device):
    tokenizer = transformers.AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.1")
    model = transformers.AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.1")
    device = torch.device(device)
    model.eval()
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'right'
    instruction= "Please complete the given text in the " + language + " language : "
    ai_text = []
    for item in tqdm(human):
        prefix_input_ids=tokenizer([f"{instruction} {item}"],max_length=30,padding='max_length',truncation=True,return_tensors="pt")
        prefix_input_ids={k:v.to(device) for k,v in prefix_input_ids.items()}
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
            item.replace(instruction,"") for item in output_text
        ]
        ai_text.append(AI_texts)
    
    with open(output_ai, 'w') as f:
        for item in ai_text:
            f.write("%s\n" % item)

if __name__ == '__main__':
    args = get_args()
    human = load_data(args.language, args.samples)
    if args.model == 'llama':
        llama(human, args.language, args.output_ai, args.device)
    elif args.model == 'vicuna-7b':
        vicuna(human, args.language, args.output_ai, args.device)
    else:
        print('Invalid model')
    
