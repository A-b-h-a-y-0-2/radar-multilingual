import argparse
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, pipeline, T5Tokenizer, T5ForConditionalGeneration, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
#from .binoculars import Binoculars
import time
import torch.nn.functional as F
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, help='path to dataset' )
    parser.add_argument('--output', type=str, help='path to output file')
    parser.add_argument('--custom', type=str,help='Path to custom model')
    parser.add_argument('--mode', type=str, help='Mhich way to paeaphrase', choices= ['backtranslation', 'transformer', 'translation'])
    return parser.parse_args()
args = get_args()
def psp_paraphrase(input_df,num_return_sequences=1,num_beams=1):
  model_name = 'tuner007/pegasus_paraphrase'
  torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
  tokenizer = PegasusTokenizer.from_pretrained(model_name)
  model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
  df_ai = input_df[input_df['label'] == '1']
  df_ai = df_ai['text']
  df_para = pd.DataFrame(columns=['text'])
  for input_text in tqdm(df_ai):
    batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    df_para = df_para.append({'text':tgt_text[0]},ignore_index=True)
  return df_para



def dp_paraphrase( input_df, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
    """Paraphrase a text using the DIPPER model.

    Args:
        input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
        lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
        order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
        **kwargs: Additional keyword arguments like top_p, top_k, max_length.
    """
    model="kalpeshk2011/dipper-paraphraser-xxl"
    prefix = "A Article writer is writing an article on a topic.He wants to paraphrase the following sentences."

    verbose=True
    time1 = time.time()
    tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
    model = T5ForConditionalGeneration.from_pretrained(model)
    if verbose:
        print(f"{model} model loaded in {time.time() - time1}")
    model.cuda()
    model.eval()
    assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
    assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

    lex_code = int(100 - lex_diversity)
    order_code = int(100 - order_diversity)
    df_ai =  input_df[input_df['label ']==1]
    df_ai = df_ai['text']
    df_para = pd.DataFrame(columns=['text'])
    for input_text in tqdm(df_ai):
        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = model.generate(**final_input, **kwargs)
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]
        df_para = df_para.append({'text':output_text},ignore_index=True)
    return df_para

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
# def checkBino(input_df):
#     bino = Binoculars()
#     output_probs = []
#     output_preds = []
#     for i in tqdm(input_df['text']):
#         output_pred = bino.predict(i)
#         output_prob = bino.compute_score(i)
#         output_preds.append(output_pred)
#         output_probs.append(output_prob)
#     input_df['preds'] = output_preds
#     input_df['score'] = output_probs
#     return input_df
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
def backtranslate(input_df,target_lang):
    translator = pipeline('translation', model='Helsinki-NLP/opus-mt-en-'+target_lang, device=0)
    back_translator = pipeline('translation', model='Helsinki-NLP/opus-mt-'+target_lang+'-en', device=0)
    text = []
    for input_text in tqdm(input_df['text']):
        translated_text = translator(input_text, max_length=512)
        back_translated_text = back_translator(translated_text[0]['translation_text'], max_length=512)
        text.append(back_translated_text[0]['translation_text'])
    results_df = input_df.copy()
    results_df['bt-text'] = text
    return results_df
def translate(input_df,target_lang):
    translator = pipeline('translation', model='Helsinki-NLP/opus-mt-en-'+target_lang, device=0)
    text = []
    for input_text in tqdm(input_df['text']):
        translated_text = translator(input_text, max_length=512)
        text.append(translated_text[0]['translation_text'])
    results_df = input_df.copy()
    results_df['t-text'] = text
    return results_df
if __name__== "__main__":
    args  = get_args()
    if args.mode == 'backtranslation':
        bt_df_zh = backtranslate(args.dataset, 'zh')
        bt_df_ru = backtranslate(args.dataset, 'ru')
        bt_df_es = backtranslate(args.dataset, 'es')
        bt_df_hi = backtranslate(args.dataset, 'hi')
        bt_df_ar = backtranslate(args.dataset, 'ar')
        if args.model == 'custom':
            res_zh = check_custom(bt_df_zh, args.custom)
            res_ru = check_custom(bt_df_ru, args.custom)
            res_es = check_custom(bt_df_es, args.custom)
            res_hi = check_custom(bt_df_hi, args.custom)
            res_ar = check_custom(bt_df_ar, args.custom)
            res_zh.to_csv(args.output + 'custom_bt_zh.csv', index=False)
            res_ru.to_csv(args.output + 'custom_bt_ru.csv', index=False)
            res_es.to_csv(args.output + 'custom_bt_es.csv', index=False)
            res_hi.to_csv(args.output + 'custom_bt_hi.csv', index=False)
            res_ar.to_csv(args.output + 'custom_bt_ar.csv', index=False)
        else:
            radar_zh = check_radar(bt_df_zh)
            radar_ru = check_radar(bt_df_ru)
            radar_es = check_radar(bt_df_es)
            radar_hi = check_radar(bt_df_hi)
            radar_ar = check_radar(bt_df_ar)
            radar_zh.to_csv(args.output + 'radar_bt_zh.csv', index=False)
            radar_ru.to_csv(args.output + 'radar_bt_ru.csv', index=False)
            radar_es.to_csv(args.output + 'radar_bt_es.csv', index=False)
            radar_hi.to_csv(args.output + 'radar_bt_hi.csv', index=False)
            radar_ar.to_csv(args.output + 'radar_bt_ar.csv', index=False)
            stat_zh = checkStat(bt_df_zh)
            stat_ru = checkStat(bt_df_ru)
            stat_es = checkStat(bt_df_es)
            stat_hi = checkStat(bt_df_hi)
            stat_ar = checkStat(bt_df_ar)
            stat_zh.to_csv(args.output + 'stat_bt_zh.csv', index=False)
            stat_ru.to_csv(args.output + 'stat_bt_ru.csv', index=False)
            stat_es.to_csv(args.output + 'stat_bt_es.csv', index=False)
            stat_hi.to_csv(args.output + 'stat_bt_hi.csv', index=False)
            stat_ar.to_csv(args.output + 'stat_bt_ar.csv', index=False)
            multi_zh = checkMultitude(bt_df_zh)
            multi_ru = checkMultitude(bt_df_ru)
            multi_es = checkMultitude(bt_df_es)
            multi_hi = checkMultitude(bt_df_hi)
            multi_ar = checkMultitude(bt_df_ar)
            multi_zh.to_csv(args.output + 'multi_bt_zh.csv', index=False)
            multi_ru.to_csv(args.output + 'multi_bt_ru.csv', index=False)
            multi_es.to_csv(args.output + 'multi_bt_es.csv', index=False)
            multi_hi.to_csv(args.output + 'multi_bt_hi.csv', index=False)
            multi_ar.to_csv(args.output + 'multi_bt_ar.csv', index=False)
            roberta_zh = checkRoberta(bt_df_zh)
            roberta_ru = checkRoberta(bt_df_ru)
            roberta_es = checkRoberta(bt_df_es)
            roberta_hi = checkRoberta(bt_df_hi)
            roberta_ar = checkRoberta(bt_df_ar)
            roberta_zh.to_csv(args.output + 'roberta_bt_zh.csv', index=False)
            roberta_ru.to_csv(args.output + 'roberta_bt_ru.csv', index=False)
            roberta_es.to_csv(args.output + 'roberta_bt_es.csv', index=False)
            roberta_hi.to_csv(args.output + 'roberta_bt_hi.csv', index=False)
            roberta_ar.to_csv(args.output + 'roberta_bt_ar.csv', index=False)
            # bino_zh = checkBino(bt_df_zh)
            # bino_ru = checkBino(bt_df_ru)
            # bino_es = checkBino(bt_df_es)
            # bino_hi = checkBino(bt_df_hi)
            # # bino_ar = checkBino(bt_df_ar)
            # bino_zh.to_csv(args.output + 'bino_bt_zh.csv', index=False)
            # bino_ru.to_csv(args.output + 'bino_bt_ru.csv', index=False)
            # bino_es.to_csv(args.output + 'bino_bt_es.csv', index=False)
            # bino_hi.to_csv(args.output + 'bino_bt_hi.csv', index=False)
            # bino_ar.to_csv(args.output + 'bino_bt_ar.csv', index=False)
        pass
    elif args.mode == 'transformer':
        dp_para = dp_paraphrase(args.dataset, lex_diversity=60, order_diversity=0, do_sample=True, top_p=0.75, top_k=None, max_length=512)
        psp_para = psp_paraphrase(args.dataset,1,1)
        if args.model == 'custom':
            dp_results_para = check_custom(dp_para, args.custom)
            psp_results_para = check_custom(psp_para, args.custom)
            dp_results_para.to_csv(args.output + 'custom_dp.csv', index=False)
            psp_results_para.to_csv(args.output + 'custom_psp.csv', index=False)
        else:
            res_radar_psp = check_radar(dp_para)
            res_radar_dp = check_radar(psp_para)
            res_radar_psp.to_csv(args.output + 'radar_psp.csv', index=False)
            res_radar_dp.to_csv(args.output + 'radar_dp.csv', index=False)
            stat_psp = checkStat(dp_para)
            stat_dp = checkStat(psp_para)
            stat_psp.to_csv(args.output + 'stat_psp.csv', index=False)
            stat_dp.to_csv(args.output + 'stat_dp.csv', index=False)
            multi_dp = checkMultitude(dp_para)
            multi_psp = checkMultitude(psp_para)
            multi_dp.to_csv(args.output + 'multi_dp.csv', index=False)
            multi_psp.to_csv(args.output + 'multi_psp.csv', index=False)
            roberta_dp = checkRoberta(dp_para)
            roberta_psp = checkRoberta(psp_para)
            roberta_dp.to_csv(args.output + 'roberta_dp.csv', index=False)
            roberta_psp.to_csv(args.output + 'roberta_psp.csv', index=False)
            # bino_dp = checkBino(dp_para)
            # bino_psp = checkBino(psp_para)
            # bino_dp.to_csv(args.output + 'bino_dp.csv', index=False)
            # bino_psp.to_csv(args.output + 'bino_psp.csv', index=False)
    elif args.mode == 'translation':
        t_df_zh = translate(args.dataset, 'zh')
        t_df_ru = translate(args.dataset, 'ru')
        t_df_es = translate(args.dataset, 'es')
        if args.model == 'custom':
            res_zh = check_custom(t_df_zh, args.custom)
            res_ru = check_custom(t_df_ru, args.custom)
            res_es = check_custom(t_df_es, args.custom)
            res_zh.to_csv(args.output + 'custom_t_zh.csv', index=False)
            res_ru.to_csv(args.output + 'custom_t_ru.csv', index=False)
            res_es.to_csv(args.output + 'custom_t_es.csv', index=False)
        else:
            radar_zh = check_radar(t_df_zh)
            radar_es = check_radar(t_df_es)
            radar_ru = check_radar(t_df_ru)
            radar_zh.to_csv(args.output + 'radar_t_zh.csv', index=False)
            radar_es.to_csv(args.output + 'radar_t_es.csv', index=False)
            radar_ru.to_csv(args.output + 'radar_t_ru.csv', index=False)
            stat_zh = checkStat(t_df_zh)
            stat_es = checkStat(t_df_es)
            stat_ru = checkStat(t_df_ru)
            stat_zh.to_csv(args.output + 'stat_t_zh.csv', index=False)
            stat_es.to_csv(args.output + 'stat_t_es.csv', index=False)
            stat_ru.to_csv(args.output + 'stat_t_ru.csv', index=False)
            multi_zh = checkMultitude(t_df_zh)
            multi_es = checkMultitude(t_df_es)
            multi_ru = checkMultitude(t_df_ru)
            multi_zh.to_csv(args.output + 'multi_t_zh.csv', index=False)
            multi_es.to_csv(args.output + 'multi_t_es.csv', index=False)
            multi_ru.to_csv(args.output + 'multi_t_ru.csv', index=False)
            roberta_zh = checkRoberta(t_df_zh)
            roberta_es = checkRoberta(t_df_es)
            roberta_ru = checkRoberta(t_df_ru)
            roberta_zh.to_csv(args.output + 'roberta_t_zh.csv', index=False)
            roberta_es.to_csv(args.output + 'roberta_t_es.csv', index=False)
            roberta_ru.to_csv(args.output + 'roberta_t_ru.csv', index=False)
            # bino_zh = checkBino(t_df_zh)
            # bino_es = checkBino(t_df_es)
            # bino_ru = checkBino(t_df_ru)
            # bino_zh.to_csv(args.output + 'bino_t_zh.csv', index=False)
            # bino_es.to_csv(args.output + 'bino_t_es.csv', index=False)
            # bino_ru.to_csv(args.output + 'bino_t_ru.csv', index=False)
    else:
        raise ValueError("Invalid mode. Choose from 'backtranslation', 'transformer', 'translation'")
    