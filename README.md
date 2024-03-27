
## Summary
### PipeLine 1 : human corpus  -> AI corpus -> Detectors (shown in the image below) ::
![image](https://github.com/A-b-h-a-y-0-2/radar-multilingual/assets/143434285/4e5d0268-1162-4777-94fb-4bad7d4529bb)

### Pipeline 2 : Human corpus -> AI corpus -> Translators  -> Detctors (shown in the image below) ::
![image](https://github.com/A-b-h-a-y-0-2/radar-multilingual/assets/143434285/83a5df69-c5fa-42ae-b0ec-0e3b2cb42679)



<div class="block-language-tx"><table>
<caption id="prototypetable">TABLE-III: Results of Multilingual analysis over different MGT detectors. </caption>
<thead>
<tr>
<th></th>
<th style="text-align:center" colspan="3">Italian</th>
<th style="text-align:center" colspan="3">German</th>
<th style="text-align:center" colspan="3">French</th>
</tr>
<tr>
<th>Detector used</th>
<th style="text-align:center">OpenAi's RoBERTa</th>
<th style="text-align:right">RADAR</th>
<th style="text-align:right">logrank</th>
<th style="text-align:center">OpenAi's RoBERTa</th>
<th style="text-align:right">RADAR</th>
<th style="text-align:right">Logrank</th>
<th style="text-align:center">OpenAi's RoBERTa</th>
<th style="text-align:right">RADAR</th>
<th style="text-align:right">Logrank</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>Pipeline 1</td>
    <td style="text-align:center">TPR: 98.30%<br/>FPR: 99.8% 
    <td style="text-align:center">TPR: 56.34%<br/>FPR: 20.42% 
    <td style="text-align:center">TPR: 36.16%<br/>FPR: 1.8%
    <td style="text-align:center">TPR: 98.70%<br/>FPR: 99.4%
    <td style="text-align:center">TPR: 53.70%<br/>FPR: 9.53%
    <td style="text-align:center">TPR: 62.70%<br/>FPR: 21.8%
    <td style="text-align:center">TPR: 98.7%<br/>FPR: 99.7% 
    <td style="text-align:center">TPR: 62.70%<br/>FPR: 14.00%
    <td style="text-align:center">TPR: 64.20%<br/>FPR: 24.3%
</tr>
      <tr>
    <td>Pipeline 2</td>
    <td style="text-align:center">TPR: 50.94%<br/>FPR: 81.5% 
    <td style="text-align:center">TPR: 95.90%<br/>FPR: 56.41%
    <td style="text-align:center">TPR: 85.11%<br/>FPR: 48.00%
    <td style="text-align:center">TPR: 56.20%<br/>FPR: 76.30% 
    <td style="text-align:center">TPR: 94.60%<br/>FPR: 33.60%
    <td style="text-align:center">TPR: 88.60%<br/>FPR: 56.30%
    <td style="text-align:center">TPR: 53.40%<br/>FPR: 76.70% 
    <td style="text-align:center">TPR: 95.90%<br/>FPR: 58.30%
    <td style="text-align:center">TPR: 91.20%<br/>FPR: 58.70%
</tr>
    </tbody>
    <tbody>

</tbody>
</table>
</div>


# Detailed Analysis
False Positive : human text as AI
True Positive : AI text as AI
False Negative : AI text as human
True Negative : Human texts as human 

## Results for German
- true Positive rate : 90.47 %
- False Positive Rate: 09.53 %
- true Positive rate : 53.70 % 
- AUROC score : 84.30

## Results for Italian
- true Negative rate : 79.59 %
- False Positive Rate: 20.41 %
- true Positive rate : 56.34 %
- AUROC score : 79.21

## Results for French
- true Negative rate : 86.00 %
- False Positive Rate: 14.00 %
- true Positive rate : 62.70  %
- AUROC score : 80.14

## Results for German Texts after Translations
- true Negative rate : 66.31 %
- false Positve rate : 33.68 %
- true Positive rate : 94.6 %
- AUROC acore : 91.69

## Results for Italian Texts after Translations
- true Negative rate : 43.58 %
- false Positve rate : 56.41 %
- true Positive rate : 95.90 %
- AUROC score : 83.58


## Results for French after Translations 
- true Negative rate : 41.82 %
- False Positive Rate: 58.17 %
- true Positive rate : 95.9%
- AURUC : 86.23


# ROBERTA analysis

## Results over French
- True Negative rate ::  0.3 %
- False Positive Rate::  99.7 %
- True Positive rate ::  98.7 %
  
## Results over French after Translations
- True Negative rate ::  23.3 %
- False Positive Rate::  76.7 %
- True Positive rate ::  46.6 %

## Results over german
- True Negative rate ::  0.60 %
- False Positive Rate::  99.4 %
- True Positive rate ::  98.7 %

## Results over german after translations
- True Positive rate ::  43.8 %
- True Negative rate ::  23.7 %
- False Positive Rate:: 76.3 %

## Results over Italian
- True Negative rate ::  00.20 %
- False Positive Rate::  99.8 %
- True Positive rate ::  98.30 %

## Results over Italian after translations 
- True Negative rate ::  18.5 %
- False Positive Rate::  81.5 %
- True Positive rate ::  50.94 %

# Log Rank analysis

## Results over French
- True Negative rate ::  57.8 %
- False Positive Rate::  24.3 %
- True Positive rate ::  64.2 %
  
## Results over French after Translations
- True Negative rate ::  29.8 %
- False Positive Rate::  58.7 %
- True Positive rate ::  91.2 %

## Results over german
- True Negative rate ::  65.4 %
- False Positive Rate::  21.8 %
- True Positive rate ::  62.7 %

## Results over german after translations
- True Positive rate ::  88.6 %
- True Negative rate ::  35.1 %
- False Positive Rate::  56.3 %

## Results over Italian
- True Negative rate ::  96.9 %
- False Positive Rate::  1.8 %
- True Positive rate ::  36.16 %

## Results over Italian after translations 
- True Negative rate ::  43.00 %
- False Positive Rate:: 48.00 %
- True Positive rate ::  85.11 %


