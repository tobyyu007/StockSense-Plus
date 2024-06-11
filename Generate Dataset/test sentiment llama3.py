import ollama
from datasets import load_dataset
import tqdm


stocksense_modelfile='''
FROM /Users/jackchen/Downloads/Stocksense-Plus-Full-GGUF-unsloth.Q4_K_M.gguf
SYSTEM What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}. Don't say other things.
TEMPLATE "{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"
PARAMETER num_keep 24
PARAMETER stop <|start_header_id|>
PARAMETER stop <|end_header_id|>
PARAMETER stop <|eot_id|>
'''

original_llama3_modelfile='''
FROM llama3:latest
SYSTEM What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.
TEMPLATE "{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"
PARAMETER num_keep 24
PARAMETER stop <|start_header_id|>
PARAMETER stop <|end_header_id|>
PARAMETER stop <|eot_id|>
'''

mistral_modelfile = '''
FROM mistral:latest
SYSTEM What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.
TEMPLATE [INST] {{ if .System }}{{ .System }} {{ end }}{{ .Prompt }} [/INST]

PARAMETER stop [INST]
PARAMETER stop [/INST]
'''

# change the model to the model you want to use
model = 'mistral'
modelfile = {'stocksense': stocksense_modelfile, 'original_llama3': original_llama3_modelfile, 'mistral': mistral_modelfile}
ollama.create(model='stocksense-plus-test', modelfile=modelfile[model])

# Load json data from disk
dataset = load_dataset("json", data_files="test sentiment.json", split="train")
userPrompts = []
answers = []
generatedAnswers = []
correctResponse = 0

for d in dataset:
    userPrompts.append([{'role': 'user', 'content': d['Input']}])
    answers.append(d['Output'])

for idx, prompts in enumerate(tqdm.tqdm(userPrompts)):
    print(prompts)
    response = ollama.chat(model='stocksense-plus-test', messages=prompts)
    generatedAns = response['message']['content'].lower().strip()
    tqdm.tqdm.write(f"Generated Answer: {generatedAns:<10} Expected Answer: {answers[idx]}")
    if answers[idx] == generatedAns:
        print("Correct")
        correctResponse += 1

print(correctResponse)
print(correctResponse/len(answers))