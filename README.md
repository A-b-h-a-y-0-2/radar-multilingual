
## Summary
### PipeLine 1 : human corpus  -> AI corpus -> Detectors (shown in the image below) ::
![image](https://github.com/A-b-h-a-y-0-2/radar-multilingual/assets/143434285/4e5d0268-1162-4777-94fb-4bad7d4529bb)

### Pipeline 2 : Human corpus -> AI corpus -> Translators  -> Detctors (shown in the image below) ::
![image](https://github.com/A-b-h-a-y-0-2/radar-multilingual/assets/143434285/83a5df69-c5fa-42ae-b0ec-0e3b2cb42679)

## Results
[Link to sheet.](https://docs.google.com/spreadsheets/d/1AKM0zlMQZoomOVhyPTVduYcDxLA_gdeOGynHypB8jQQ/edit?usp=sharing)

![image](https://github.com/A-b-h-a-y-0-2/radar-multilingual/assets/143434285/0e42d46b-4c84-46f1-a4cf-b11678c4aa82)

## Results using llama2-chat
![image](https://github.com/A-b-h-a-y-0-2/radar-multilingual/assets/143434285/cc97e4f8-d413-4352-90b8-cbe7cf3da8bb)

## Results over Multitude dataset.
![image](https://github.com/A-b-h-a-y-0-2/radar-multilingual/assets/143434285/a075edf9-bb98-467b-a450-b6e8a5b0f3eb)



## Instructions 
instructions to run script.py
- example

```rb
python3 script.py --language French --tr True --model RADAR --ai ai_fr.txt --output_h human_test.csv --output_ai ai_test.csv --samples 512
```
- language French, German, Italian
- model RADAR, RoBERTa, logrank, logp, entropy
- --tr to use translations
- ai_fr.txt #path/to/ai_sample_file
- human_test.csv #path/to/output/res
- ai_test.csv #path/to/output/res
- number of samples

  Instructions to generate ai corpus
-example code

```rb
python3 ai_generate.py --language French --model llama --device cuda --samples 10 --output_ai ai_test.txt
```
- language
- model {llama, vicuna-7b}
- device
- samples
- output_ai
  
