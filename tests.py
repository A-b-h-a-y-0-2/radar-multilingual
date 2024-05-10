import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline
import torch
from tqdm import tqdm
import torch.nn.functional as F
import csv
from .ai_generate import *
import os
import pandas as pd
import csv
import numpy as np
import time 
from .binoculars import Binoculars

def check_radar(input_df):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  detector = AutoModelForSequenceClassification.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
  tokenizer = AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
  detector.eval()
  detector.to(device)
  for input_text in tqdm(input_df['text']):
    with torch.no_grad():
      inputs = tokenizer(input_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
      inputs = {k:v.to(device) for k,v in inputs.items()}
      output_probs = F.log_softmax(detector(**inputs).logits,-1)[:,0].exp().tolist()
      df_radar = df_radar.append({'text':output_probs},ignore_index=True)
  return df_radar

def predict(df, classifier):
  preds = ['unknown'] * len(df)
  scores = [0] * len(df)
  for index, row in tqdm(df.iterrows(), total=len(df)):
    tokenizer_kwargs = {'truncation':True,'max_length':512}
    pred = classifier(row['text'], **tokenizer_kwargs)
    preds[index] = pred[0]['label']
    scores[index] = pred[0]['score']
  return preds, scores

def checkRoberta(input_df):
    pipe = pipeline("text-classification", model="openai-community/roberta-large-openai-detector")
    preds = []
    for text in tqdm(input_df['text']):
        pred = pipe(text)
        preds.append(pred)
    results = input_df.copy()
    results['preds'] = preds
    return results
  
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

def analyse_logrank(input_df):
    base_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
    base_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")
    base_model.eval()
    base_model.to('cuda')
    output_probs = []
    for i in tqdm(input_df['text']):
        output = get_rank(i, base_model, base_tokenizer)
        output_probs.append(output)
    input_df['logrank'] = output_probs
    return input_df

def analyse_logp(input_df):
    base_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
    base_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")
    base_model.eval()
    base_model.to('cuda')
    output_probs = []
    for i in tqdm(input_df['text']):
        output = get_ll(i, base_tokenizer, base_model)
        output_probs.append(output)
    input_df['logp'] = output_probs
    return input_df

def analyse_entropy(input_df):
    base_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
    base_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")
    base_model.eval()
    base_model.to('cuda')
    output_probs = []
    for i in tqdm(input_df['text']):
        output = get_entropy(i, base_tokenizer, base_model)
        output_probs.append(output)
    input_df['entropy'] = output_probs
    return input_df

def checkBino(input_df):
    bino = Binoculars()
    output_probs = []
    output_preds = []
    for i in tqdm(input_df['text']):
        output_pred = bino.predict(i)
        output_prob = bino.compute_score(i)
        output_preds.append(output_pred)
        output_probs.append(output_prob)
    input_df['preds'] = output_preds
    input_df['score'] = output_probs
    return input_df

def checkStat(input_df):
    logp_df = analyse_logp(input_df)
    logrank_df = analyse_logrank(input_df)
    entropy_df = analyse_entropy(input_df)
    result_df = pd.concat([logp_df, logrank_df, entropy_df], axis=1)
    return result_df

def check_custom(input_df, output_model):
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    classifier = pipeline("text-classification", model=output_model, device=device, torch_dtype=torch.float16)
    results = predict(input_df, classifier)
    results_df = input_df.copy()
    results_df['label-pred'] = results[0]
    results_df['score'] = results[1]
    return results_df

def checkMultitude(input_df):
    mdeberta = "finetuned_models/mdeberta-v3-base-all-all"
    xlm_roberta = "finetuned_models/xlm-roberta-large-all-all"
    openai_roberta = "finetuned_models/roberta-large-openai-detector-all-all"
    bert_base = "finetuned_models/bert-base-multilingual-cased-all-all"
    mdb_df = check_custom(input_df, mdeberta)
    mdb_df['pred_model'] = len(mdb_df) * ['mdeberta']
    xlmr_df = check_custom(input_df, xlm_roberta)
    xlmr_df['pred_model'] = len(xlmr_df) * ['xlm-roberta']
    oar_df = check_custom(input_df, openai_roberta)
    oar_df['pred_model'] = len(oar_df) * ['openai-roberta']
    bert_df = check_custom(input_df, bert_base)
    bert_df['pred_model'] = len(bert_df) * ['bert-base']
    results_df = pd.concat([mdb_df, xlmr_df, oar_df, bert_df])
    return results_df

def translate(input_df):
    languages = set()
    for i in input_df['language']:
        languages.append(i)
    translators = []
    for lang in languages:
        if lang == 'en' or lang == 'pt':
            continue
        translator = pipeline('translation', model='Helsinki-NLP/opus-mt-'+lang+'-en', device=0)
        translators.append([translator, lang])
    text = []
    for input_text in tqdm(input_df['text']):
        for translator, lang in translators:
            if lang == 'en' :
                text.append(input_text)
            if lang == 'pt':
                text.append("No translator available for Portuguese.")
            if lang == input_df['language']:
                translated_text = translator(input_text, max_length=512)
                text.append(translated_text[0]['translation_text'])
    results_df = input_df.copy()
    results_df['text'] = text
    return results_df

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tr', type=bool, help='choose translation mode', default=False,)
    parser.add_argument('--custom', type=str, help='choose custom model to test with')
    parser.add_argument('--dataset', type=str, help='Path to dataset')
    parser.add_argument('--output', type=str, help='Path to output')
    return parser.parse_args()

if __name__ == "__main__":
    input_df = pd.read_csv(args.dataset)
    if args.tr is True:
        trans_df = translate(input_df)
        if args.custom is not None:
            output_model = args.custom
            results_custom = check_custom(trans_df, output_model)
            results_custom.to_csv(args.output+f'_custom_{args.custom}_tr(en).csv')
        else:
            results_multi = checkMultitude(trans_df)
            results_multi.to_csv(args.output+'_multi_tr(en).csv')
            results_stat = checkStat(trans_df)
            results_stat.to_csv(args.output+'_stat_tr(en).csv')
            results_bino = checkBino
            results_bino.to_csv(args.output+'_bino_tr(en).csv')
            results_roberta = checkRoberta(trans_df)
            results_roberta.to_csv(args.output+'_roberta_tr(en).csv')
            results_radar = check_radar(trans_df)
            results_radar.to_csv(args.output+'_radar_tr(en).csv')
    else :
        if args.custom is not None:
            output_model = args.custom
            results_custom = check_custom(input_df, output_model)
            results_custom.to_csv(args.output+f'_custom_{args.custom}.csv')
        else:
            results_multi = checkMultitude(input_df)
            results_multi.to_csv(args.output+'_multi.csv')
            results_stat = checkStat(input_df)
            results_stat.to_csv(args.output+'_stat.csv')
            results_bino = checkBino(input_df)
            results_bino.to_csv(args.output+'_bino.csv')
            results_roberta = checkRoberta(input_df)
            results_roberta.to_csv(args.output+'_roberta.csv')
            results_radar = check_radar(input_df)
            results_radar.to_csv(args.output+'_radar.csv')
