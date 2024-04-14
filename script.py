import argparse
from datasets import load_dataset
import transformers
import torch
from tqdm import tqdm
from transformers import pipeline
import torch.nn.functional as F
import csv
from .ai_generate import *
import os
import pandas as pd
import csv
from sklearn.metrics import accuracy_score, auc, roc_curve



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, help='choose language to test in', choices=['French', 'German', 'Italian', 'Spanish'])
    parser.add_argument('--model', type=str, help='choose model to test with', choices=['RADAR', 'RoBERTa', 'logrank', 'logp', 'entropy', 'all'])
    parser.add_argument('--tr', type=bool, help='choose translation mode', default=False,)
    parser.add_argument('--samples', type=int, help='number off samples to take.')
    parser.add_argument('--ai', type=str, help='path to the ai file')
    parser.add_argument('--output', type=str, help='path to the output file for human preds')
    parser.add_argument('--dataset', type=str, help='choose dataset', choices=['wikipedia', 'multitude'])
    return parser.parse_args()
def analyse(pred_ai, pred_human, output, model):
    ## Calculate the ROC metrics
    fpr, tpr, roc_auc = get_roc_metrics(pred_human, pred_ai)
    ## calculate confusion matrix
    tnr, fpr, tpr, fnr = accuracy(pred_human, pred_ai)
    ## calaculate accuracy score
    accuracy = accuracy_score([0] * len(pred_human) + [1] * len(pred_ai), pred_human + pred_ai)
    ## save to outpt file
    res = {'model': model, 'roc_auc': roc_auc, 'tnr': tnr, 'fpr': fpr, 'tpr': tpr, 'fnr': fnr, 'accuracy': accuracy}
    with open(output, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=res.keys())
        writer.writeheader()
        writer.writerow(res)

def get_roc_metrics(human_preds, ai_preds):
    # human_preds is the ai-generated probabiities of human-text
    # ai_preds is the ai-generated probabiities of AI-text
    if not human_preds or not ai_preds:
            # Handle empty arrays to avoid the IndexError
            return None    # Rest of your code
    fpr, tpr, _ = roc_curve([0] * len(human_preds) + [1] * len(ai_preds), human_preds + ai_preds, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def accuracy(human_pred, ai_pred):
    count_correct_human = 0
    count_incorrect_human = 0
    count_correct_ai = 0
    count_incorrect_ai = 0
    for i in human_pred:
        if i < 0.5:
            count_correct_human+=1
        else:
            count_incorrect_human +=1
    for i in ai_pred:
        if i > 0.5:
            count_correct_ai +=1
        else:
            count_incorrect_ai +=1
    return  (count_correct_human/len(human_pred)), (count_incorrect_human)/len(human_pred), (count_correct_ai/len(ai_pred)), (count_incorrect_ai)/len(ai_pred)

def load_data_multi(model, output):
    df_main = pd.read_csv('multitude.csv')
    df = df_main[['label', 'text','language']]
    languages = ['de', 'en', 'es', 'nl', 'pt', 'ru', 'zh', 'ar', 'uk', 'cs', 'ca']
    for language in languages:
        df_test = df[df['language'] == language]
        df_ai = df_test[df['label'] == 1]
        df_human = df_test[df['label'] == 0]
        human = df_human['text'].tolist()
        ai = df_ai['text'].tolist()
        if model == 'RADAR':
            pred_h, pred_ai = analyse_radar(human, ai)
        elif args.model == 'RoBERTa':
            pred_h, pred_ai = analyse_roberta(human, ai)
        elif args.model == 'logrank':
            pred_h, pred_ai = analyse_logrank(human, ai)
        elif args.model == 'logp':
            pred_h, pred_ai = analyse_logp(human, ai)

        analyse(pred_ai, pred_h, output, model)


    

def load_data_wiki(language, samples):
    human_text = [] 
    ai_text = []
    ai_gen = False
    model = input('Select model to generate ai corpus (Vicuna, llama2, None)')
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
        print('Language not supported')
    
    
    if model == 'Vicuna':
        ai_gen = True
    elif model == 'llama2':
        ai_gen = True
    filepath = model+'_ai.txt'
    if ai_gen:
        if model == 'Vicuna':
            vicuna(human, language, filepath, 'cuda')
        elif model == 'llama2':
            llama(human, language, filepath, 'cuda')
    
    
    if language == 'French':
        with open(filepath,'r') as f:
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
        with open(filepath,'r') as f:
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
        with open(filepath,'r') as f:
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
    ai_text = ai_text[:samples]
    return human_text, ai_text

def analyse_radar(human, ai):
    print('Analyse RADAR')
    device = "cuda"# example: cuda:0
    detector_path_or_id = "TrustSafeAI/RADAR-Vicuna-7B"
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(detector_path_or_id)
    detector_tokenizer = transformers.AutoTokenizer.from_pretrained(detector_path_or_id)
    detector.eval()
    detector.to(device)

    Text_input = human
    # error = 0
    output_probs_list_human =[]
    # Use detector to deternine wehther the text_input is ai-generated.
    with torch.no_grad():
        for i  in tqdm(Text_input):
            inputs = detector_tokenizer(i, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k:v.to(device) for k,v in inputs.items()}
            output_probs = F.log_softmax(detector(**inputs).logits,-1)[:,0].exp().tolist()
            output_probs_list_human.append(output_probs)
    
    Text_input = ai
    output_probs_list_ai =[]
    # Use detector to deternine wehther the text_input is ai-generated.
    with torch.no_grad():
        for i  in tqdm(Text_input):
            inputs = detector_tokenizer(i, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k:v.to(device) for k,v in inputs.items()}
            output_probs = F.log_softmax(detector(**inputs).logits,-1)[:,0].exp().tolist()
            output_probs_list_ai.append(output_probs)

    return output_probs_list_human, output_probs_list_ai

def analyse_roberta(human, ai):
    pipe = pipeline("text-classification", model="openai-community/roberta-large-openai-detector")
    output_probs_human = []
    for i in tqdm(human):
        output = pipe(i)
        output_probs_human.append(output)

    
    output_probs_ai = []
    for i in tqdm(ai):
        output = pipe(i)
        output_probs_ai.append(output)

    return output_probs_human, output_probs_ai
    

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
    output_probs_human = []
    for i in tqdm(human):
        output = get_rank(i, base_model, base_tokenizer)
        output_probs_human.append(output)


    output_probs_ai = []
    for i in tqdm(ai):
        output = get_rank(i, base_model, base_tokenizer)
        output_probs_ai.append(output)

    return output_probs_human, output_probs_ai

def analyse_logp(human, ai, output_h, output_ai):
    base_tokenizer = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
    base_model = transformers.AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")
    base_model.eval()
    base_model.to('cuda')
    output_probs_human = []
    for i in tqdm(human):
        output = get_ll(i, base_tokenizer, base_model)
        output_probs_human.append(output)
    
    output_probs_ai = []
    for i in tqdm(ai):
        output = get_ll(i, base_tokenizer, base_model)
        output_probs_ai.append(output)
    return output_probs_human, output_probs_ai

def analyse_entropy(human, ai, output_h, output_ai):
    base_tokenizer = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
    base_model = transformers.AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")
    base_model.eval()
    base_model.to('cuda')
    output_probs_human = []
    for i in tqdm(human):
        output = get_entropy(i, base_tokenizer, base_model)
        output_probs_human.append(output)
 

    output_probs_ai = []
    for i in tqdm(ai):
        output = get_entropy(i, base_tokenizer, base_model)
        output_probs_ai.append(output)
    return output_probs_human, output_probs_ai
            
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
    if args.dataset == 'multitude':
        load_data_multi(args.model, args.output)
    elif args.dataset == 'wikipedia':
        human, ai = load_data_wiki(args.language, args.ai, args.samples)
    if args.tr:
        human = translate(human, args.language)
        ai = translate(ai, args.language)
    if args.model == 'RADAR':
        pred_h, pred_ai = analyse_radar(human, ai)
        analyse(pred_ai, pred_h, args.output, args.model)
    elif args.model == 'RoBERTa':
        pred_h, pred_ai = analyse_roberta(human, ai)
        analyse(pred_ai, pred_h, args.output, args.model)
    elif args.model == 'logrank':
        pred_h, pred_ai = analyse_logrank(human, ai)
        analyse(pred_ai, pred_h, args.output, args.model)
    elif args.model == 'logp':
        pred_h, pred_ai = analyse_logp(human, ai)
        analyse(pred_ai, pred_h, args.output, args.model)
    elif args.model == 'entropy':
        pred_h, pred_ai = analyse_entropy(human, ai)
        analyse(pred_ai, pred_h, args.output, args.model)
    elif args.model == 'all':
        radar_pred_h, radar_pred_ai = analyse_radar(human, ai)
        roberta_pred_h, roberta_pred_ai = analyse_roberta(human, ai)
        logrank_pred_h, logrank_pred_ai =analyse_logrank(human, ai)
        logp_pred_h, logp_pred_ai = analyse_logp(human, ai)
        entropy_pred_h, entropy_pred_ai = analyse_entropy(human, ai)
        analyse(radar_pred_ai, radar_pred_h, args.output, 'RADAR')
        analyse(roberta_pred_ai, roberta_pred_h, args.output, 'RoBERTa')
        analyse(logrank_pred_ai, logrank_pred_h, args.output, 'logrank')
        analyse(logp_pred_ai, logp_pred_h, args.output, 'logp')
        analyse(entropy_pred_ai, entropy_pred_h, args.output, 'entropy')



    
    
