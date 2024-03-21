# radar : Multilingual analysis
all the pipelines have been computed over 1000 samples of human and ai corpus
The True positve rate for detecting texts as Humans and True Negative rate on AI.
Also detected the AUROC score over multilingual dataset.

## Results for German
- true Positve rate : 90.47 %
- false Positive rate : 09.53 %
- true Negative rate : 53.70 % 
- AUROC score : 84.30

## Results for Italian
- true Positive rate : 79.59 %
- false Positive rate : 20.41 %
- true Negative rate : 56.34 %
- AUROC score : 79.21

## Results for French
- true Positive rate : 86.00 %
- false Positive rate : 14.00 %
- true Negative rate : 62.70  %
- AUROC score : 80.14

## Results for German Texts after Translations
- true Positive rate : 66.31 %
- false Positve rate : 33.68 %
- true Negative rate : 94.6 %
- AUROC acore : 91.69

## Results for Italian Texts after Translations
- true Positive rate : 43.58 %
- false Positve rate : 56.41 %
- true Negative rate : 95.90 %
- AUROC score : 83.58


## Results for French after Translations 
- true Positive rate : 41.82 %
- false Positive rate : 58.17 %
- true Negative rate : 95.9%
- AURUC : 86.23


# ROBERTA analysis

## Results over French
- True Positive rate ::  0.3 %
- False Positive rate ::  99.7 %
- True Negative rate ::  98.7 %
  
## Results over French after Translations
- True Positive rate ::  23.3 %
- False Positive rate ::  76.7 %
- True Negative rate ::  46.6 %

## Results over german
- True Positive rate ::  0.60 %
- False Positive rate ::  99.4 %
- True Negative rate ::  98.7 %

## Results over german after translations
- True Negative rate ::  43.8 %
- True Positive rate ::  23.7 %
- False Positive rate :: 76.3 %

## Results over Italian
- True Positive rate ::  00.20 %
- False Positive rate ::  99.8 %
- True Negative rate ::  98.30 %

## Results over Italian after translations 
- True Positive rate ::  18.5 %
- False Positive rate ::  81.5 %
- True Negative rate ::  50.94 %

# Log Rank analysis

## Results over French
- True Positive rate ::  57.8 %
- False Positive rate ::  24.3 %
- True Negative rate ::  64.2 %
  
## Results over French after Translations
- True Positive rate ::  29.8 %
- False Positive rate ::  58.7 %
- True Negative rate ::  91.2 %

## Results over german
- True Positive rate ::  65.4 %
- False Positive rate ::  21.8 %
- True Negative rate ::  62.7 %

## Results over german after translations
- True Negative rate ::  88.6 %
- True Positive rate ::  35.1 %
- False Positive rate ::  56.3 %

## Results over Italian
- True Positive rate ::  96.9 %
- False Positive rate ::  1.8 %
- True Negative rate ::  36.16 %

## Results over Italian after translations 
- True Positive rate ::  43.00 %
- False Positive rate :: 48.00 %
- True Negative rate ::  85.11 %



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
    <td style="text-align:center"> True Positive 50% <\br>  True Negative 50
    <td style="text-align:center">95.3 &plusmn 12
    <td style="text-align:center">84.5 &plusmn 22.1
    <td style="text-align:center">92.97 &plusmn 8.95
    <td style="text-align:center">93.18 &plusmn 11.82
    <td style="text-align:center">91.4 &plusmn 17.1
    <td style="text-align:center">85.94 &plusmn 13.75
    <td style="text-align:center">88.58 &plusmn 12.43
    <td style="text-align:center">85.94 &plusmn 13.75
</tr>
      <tr>
    <td>Pipeline 2</td>
    <td style="text-align:center">92.2 &plusmn 8.84
    <td style="text-align:center">95.3 &plusmn 12
    <td style="text-align:center">84.5 &plusmn 22.1
    <td style="text-align:center">92.97 &plusmn 8.95
    <td style="text-align:center">93.18 &plusmn 11.82
    <td style="text-align:center">91.4 &plusmn 17.1
    <td style="text-align:center">85.94 &plusmn 13.75
    <td style="text-align:center">88.58 &plusmn 12.43
    <td style="text-align:center">85.94 &plusmn 13.75
</tr>
      <tr>
    <td>Pipeline 3</td>
    <td style="text-align:center">92.2 &plusmn 8.84
    <td style="text-align:center">95.3 &plusmn 12
    <td style="text-align:center">84.5 &plusmn 22.1
    <td style="text-align:center">92.97 &plusmn 8.95
    <td style="text-align:center">93.18 &plusmn 11.82
    <td style="text-align:center">91.4 &plusmn 17.1
    <td style="text-align:center">85.94 &plusmn 13.75
    <td style="text-align:center">88.58 &plusmn 12.43
    <td style="text-align:center">85.94 &plusmn 13.75
</tr>
      <tr>
    <td>Pipeline 4</td>
    <td style="text-align:center">92.2 &plusmn 8.84
    <td style="text-align:center">95.3 &plusmn 12
    <td style="text-align:center">84.5 &plusmn 22.1
    <td style="text-align:center">92.97 &plusmn 8.95
    <td style="text-align:center">93.18 &plusmn 11.82
    <td style="text-align:center">91.4 &plusmn 17.1
    <td style="text-align:center">85.94 &plusmn 13.75
    <td style="text-align:center">88.58 &plusmn 12.43
    <td style="text-align:center">85.94 &plusmn 13.75
</tr>
    </tbody>
    <tbody>

</tbody>
</table>
</div>

