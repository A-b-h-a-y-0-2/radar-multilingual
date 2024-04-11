import argparse
from datasets import load_dataset
import transformers
import torch
from tqdm import tqdm
from transformers import pipeline
import torch.nn.functional as F
import csv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, help='choose language to test in', choices=['French', 'German', 'Italian', 'Spanish'])
    parser.add_argument('--model', type=str, help='choose model to test with', choices=['RADAR', 'RoBERTa', 'logrank', 'logp', 'entropy', 'all'])
    parser.add_argument('--tr', type=bool, help='choose translation mode', default=False,)
    parser.add_argument('--samples', type=int, help='number off samples to take.')
    parser.add_argument('--ai', type=str, help='path to the ai file')
    parser.add_argument('--output_h', type=str, help='path to the output file for human preds')
    parser.add_argument('--output_ai', type=str, help='path to the output file for ai preds')
    return parser.parse_args()

def load_data(language, ai, samples):
    human_text = [] 
    ai_text = []
    if language == 'French':
        datawiki_fr = load_dataset('wikipedia', '20220301.fr')
        for i in range(samples):
            human_text.append(datawiki_fr['train'][i]['text'])
        with open(ai,'r') as f:
            line = f.readline()
            while line:
                line = line.strip()
                line = line.replace('start : ',' ')
                line = line.replace('[\"',' ')
                line = line.replace('[\'',' ')
                line = line.replace('\"]',' ')
                line = line.replace('\']',' ')
                ai_text.append(line)
                line = f.readline()
    elif language == 'German':
        datawiki_de = load_dataset('wikipedia', '20220301.de')
        for i in range(samples):
            human_text.append(datawiki_de['train'][i]['text'])
        with open(ai,'r') as f:
            line = f.readline()
            while line:
                line = line.strip()
                line = line.replace('start : ',' ')
                line = line.replace('[\"',' ')
                line = line.replace('[\'',' ')
                line = line.replace('\"]',' ')
                line = line.replace('\']',' ')
                ai_text.append(line)
                line = f.readline()
    elif language == 'Italian':
        datawiki_it = load_dataset('wikipedia', '20220301.it')
        for i in range(samples):
            human_text.append(datawiki_it['train'][i]['text'])
        with open(ai,'r') as f:
            line = f.readline()
            while line:
                line = line.strip()
                line = line.replace('start : ',' ')
                line = line.replace('[\"',' ')
                line = line.replace('[\'',' ')
                line = line.replace('\"]',' ')
                line = line.replace('\']',' ')
                ai_text.append(line)
                line = f.readline()
    else:
        print('Language not supported')
    ai_text = ai_text[:samples]
    return human_text, ai_text

def analyse_radar(human, ai, output_h, output_ai):
    print('Analyse RADAR')
    device = "cuda"# example: cuda:0
    detector_path_or_id = "TrustSafeAI/RADAR-Vicuna-7B"
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(detector_path_or_id)
    detector_tokenizer = transformers.AutoTokenizer.from_pretrained(detector_path_or_id)
    detector.eval()
    detector.to(device)

    Text_input = human
    # error = 0
    output_probs_list =[]
    # Use detector to deternine wehther the text_input is ai-generated.
    with torch.no_grad():
        for i  in tqdm(Text_input):
            inputs = detector_tokenizer(i, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k:v.to(device) for k,v in inputs.items()}
            output_probs = F.log_softmax(detector(**inputs).logits,-1)[:,0].exp().tolist()
            output_probs_list.append(output_probs)
    with open(output_h, 'w', newline= '\n') as f:
        writer = csv.writer(f)
        for item in output_probs_list:
            writer.writerow(item)
    
    Text_input = ai
    output_probs_list =[]
    # Use detector to deternine wehther the text_input is ai-generated.
    with torch.no_grad():
        for i  in tqdm(Text_input):
            inputs = detector_tokenizer(i, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k:v.to(device) for k,v in inputs.items()}
            output_probs = F.log_softmax(detector(**inputs).logits,-1)[:,0].exp().tolist()
            output_probs_list.append(output_probs)
    with open(output_ai, 'w', newline= '\n') as f:
        writer = csv.writer(f)
        for item in output_probs_list:
            writer.writerow(item)

def analyse_roberta(human, ai, output_h, output_ai):
    pipe = pipeline("text-classification", model="openai-community/roberta-large-openai-detector")
    output_probs = []
    for i in tqdm(human):
        output = pipe(i)
        output_probs.append(output)
    with open(output_h, 'w', newline= '\n') as f:
        writer = csv.writer(f)
        for item in output_probs:
            writer.writerow(item)
    
    output_probs = []
    for i in tqdm(ai):
        output = pipe(i)
        output_probs.append(output)
    with open(output_ai, 'w', newline= '\n') as f:
        writer = csv.writer(f)
        for item in output_probs:
            writer.writerow(item)

def get_rank(text, base_model, base_tokenizer,log=False ):
    DEVICE = 'cuda'
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        labels = tokenized.input_ids[:,1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:,-1], matches[:,-2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1 # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()

def get_ll(text, base_tokenizer, base_model):
    DEVICE = 'cuda'    
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        return -base_model(**tokenized, labels=labels).loss.item()

def get_entropy(text, base_tokenizer, base_model):
    DEVICE = 'cuda'

    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        neg_entropy = torch.nn.functional.softmax(logits, dim=-1) * torch.nn.functional.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()

def analyse_logrank(human, ai, output_h, output_ai):
    base_tokenizer = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
    base_model = transformers.AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")
    base_model.eval()
    base_model.to('cuda')
    output_probs = []
    for i in tqdm(human):
        output = get_rank(i, base_model, base_tokenizer)
        output_probs.append(output)
    with open(output_h, 'w', newline= '\n') as f:
        writer = csv.writer(f)
        for item in output_probs:
            writer.writerow(item)

    output_probs = []
    for i in tqdm(ai):
        output = get_rank(i, base_model, base_tokenizer)
        output_probs.append(output)
    with open(output_ai, 'w', newline= '\n') as f:
        writer = csv.writer(f)
        for item in output_probs:
            writer.writerow(item)

def analyse_logp(human, ai, output_h, output_ai):
    base_tokenizer = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
    base_model = transformers.AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")
    base_model.eval()
    base_model.to('cuda')
    output_probs = []
    for i in tqdm(human):
        output = get_ll(i, base_tokenizer, base_model)
        output_probs.append(output)
    with open(output_h, 'w', newline= '\n') as f:
        writer = csv.writer(f)
        for item in output_probs:
            writer.writerow(item)

    output_probs = []
    for i in tqdm(ai):
        output = get_ll(i, base_tokenizer, base_model)
        output_probs.append(output)
    with open(output_ai, 'w', newline= '\n') as f:
        writer = csv.writer(f)
        for item in output_probs:
            writer.writerow(item)

def analyse_entropy(human, ai, output_h, output_ai):
    base_tokenizer = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
    base_model = transformers.AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")
    base_model.eval()
    base_model.to('cuda')
    output_probs = []
    for i in tqdm(human):
        output = get_entropy(i, base_tokenizer, base_model)
        output_probs.append(output)
    with open(output_h, 'w', newline= '\n') as f:
        writer = csv.writer(f)
        for item in output_probs:
            writer.writerow(item)

    output_probs = []
    for i in tqdm(ai):
        output = get_entropy(i, base_tokenizer, base_model)
        output_probs.append(output)
    with open(output_ai, 'w', newline= '\n') as f:
        writer = csv.writer(f)
        for item in output_probs:
            writer.writerow(item)
            
def translate(text, language):
    tokenizer = None 
    model = None
    if language == 'French':
        tokenizer = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    elif language == 'German':
        tokenizer = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    elif language == 'Italian':
        tokenizer = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-it-en")
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-it-en")

    model.eval()
    model.to(torch.device('cuda'))

    for i in range(len(text)):
        text[i] = text[i][:1024]
    
    text_tr = []
    for text in tqdm(text):
        #inp = text[:512]\
        inp = text
        input_ids = tokenizer(inp, return_tensors="pt").input_ids
        input_ids = input_ids.to(torch.device('cuda'))
        outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=1)
        text_tr.append(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        del input_ids
        del outputs
        torch.cuda.empty_cache()
    
    return text_tr

if __name__ == '__main__':
    args = get_args()
    human, ai = load_data(args.language, args.ai, args.samples)
    if args.tr:
        human = translate(human, args.language)
        ai = translate(ai, args.language)
    if args.model == 'RADAR':
        output_h = os.path.join(args.output_h, 'RADAR_'+args.language+'_human.csv')
        output_ai = os.path.join(args.output_ai, 'RADAR_'+ args.language +'_ai.csv')
        analyse_radar(human, ai, output_h, output_ai)
    elif args.model == 'RoBERTa':
        output_h = os.path.join(args.output_h, 'RoBERTa_'+args.language+'_human.csv')
        output_ai = os.path.join(args.output_ai, 'RoBERTa_'+ args.language +'_ai.csv')
        analyse_roberta(human, ai, output_h, output_ai)
    elif args.model == 'logrank':
        output_h = os.path.join(args.output_h, 'logrank_'+args.language+'_human.csv')
        output_ai = os.path.join(args.output_ai, 'logrank_'+ args.language +'_ai.csv')
        analyse_logrank(human, ai, args.output_h, args.output_ai)
    elif args.model == 'logp':
        output_h = os.path.join(args.output_h, 'logp_'+args.language+'_human.csv')
        output_ai = os.path.join(args.output_ai, 'logp_'+ args.language +'_ai.csv')
        analyse_logp(human, ai, args.output_h, args.output_ai)
    elif args.model == 'entropy':
        output_h = os.path.join(args.output_h, 'entropy_'+args.language+'_human.csv')
        output_ai = os.path.join(args.output_ai, 'entropy_'+ args.language +'_ai.csv')
        analyse_entropy(human, ai, args.output_h, args.output_ai)
    elif args.model == 'all':
        output_h = os.path.join(args.output_h, 'RADAR_'+args.language+'_human.csv')
        output_ai = os.path.join(args.output_ai, 'RADAR_'+ args.language +'_ai.csv')
        analyse_radar(human, ai, output_h, output_ai)
        output_h = os.path.join(args.output_h, 'RoBERTa_'+args.language+'_human.csv')
        output_ai = os.path.join(args.output_ai, 'RoBERTa_'+ args.language +'_ai.csv')
        analyse_roberta(human, ai, output_h, output_ai)
        output_h = os.path.join(args.output_h, 'logrank_'+args.language+'_human.csv')
        output_ai = os.path.join(args.output_ai, 'logrank_'+ args.language +'_ai.csv')
        analyse_logrank(human, ai, args.output_h, args.output_ai)
        output_h = os.path.join(args.output_h, 'logp_'+args.language+'_human.csv')
        output_ai = os.path.join(args.output_ai, 'logp_'+ args.language +'_ai.csv')
        analyse_logp(human, ai, args.output_h, args.output_ai)
        output_h = os.path.join(args.output_h, 'entropy_'+args.language+'_human.csv')
        output_ai = os.path.join(args.output_ai, 'entropy_'+ args.language +'_ai.csv')
        analyse_entropy(human, ai, args.output_h, args.output_ai)
    
    
