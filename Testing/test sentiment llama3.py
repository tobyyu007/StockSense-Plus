import ollama
from datasets import load_dataset
import tqdm


modelfile='''
FROM /Users/toby/stocksense.gguf
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

# modelfile='''
# FROM llama3:latest
# SYSTEM What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.
# TEMPLATE "{{ if .System }}<|start_header_id|>system<|end_header_id|>

# {{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

# {{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

# {{ .Response }}<|eot_id|>"
# PARAMETER num_keep 24
# PARAMETER stop <|start_header_id|>
# PARAMETER stop <|end_header_id|>
# PARAMETER stop <|eot_id|>
# '''

ollama.create(model='stocksense-plus-test', modelfile=modelfile)

# Load json data from disk
dataset = load_dataset("json", data_files="sentiment.json", split="train")
userPrompts = []
answers = []
generatedAnswers = []
correctResponse = 0

for d in dataset:
    index_of_open_brace = d['messages'][0]['content'].index('{')
    index_of_close_brace = d['messages'][0]['content'].index('}')
    if d['messages'][0]['content'][index_of_open_brace+1:index_of_close_brace] == 'negative/neutral/positive':
        userPrompts.append([{'role': 'user', 'content': d['messages'][1]['content']+ d['messages'][0]['content'] + "Don't say other things, just give me answer in {negative/neutral/positive} for your response"}])
        groundTruth = d['messages'][2]['content'].split(' ')
        answers.append((groundTruth[0]))

for idx, prompts in enumerate(tqdm.tqdm(userPrompts)):
    print(prompts)
    response = ollama.chat(model='stocksense-plus-test', messages=prompts)
    generatedAns = response['message']['content'].lower()
    tqdm.tqdm.write(f"Generated Answer: {generatedAns:<10} Expected Answer: {answers[idx]}")
    if answers[idx] == generatedAns:
        correctResponse += 1

print(correctResponse)
print(correctResponse/len(answers))