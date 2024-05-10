## Instructions 
Instructions to run test.py
- example

```
python3 tests.py --tr True --custom path/to/model --dataset path/to/dataset --output path/to/output_file
```
Instructions to paraphrase and test 

```
python3 paraphrase.py --mode x --custom path/to/model --dataset path/to/dataset --output path/to/output_file
```
here x can be any of the chices = ['backtranslation', 'transformer', 'translation']

```
python3 ai_generate.py --language French --model llama --device cuda --samples 10 --output_ai ai_test.txt
```
- language
- model {llama, vicuna-7b}
- device
- samples
- output_ai
  
