import ollama
from datasets import load_dataset


# modelfile='''
# FROM /Users/toby/Downloads/Stocksense-Plus-Prediction-Q4_K_M.gguf
# SYSTEM You are a seasoned stock market analyst. Your task is to predict the companies' stock price movement for this week based on this week's positive headlines and negative headlines. Give me answer in the format of {increased/decreased/flat} in {X}%
# TEMPLATE "{{ if .System }}<|start_header_id|>system<|end_header_id|>

# {{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

# {{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

# {{ .Response }}<|eot_id|>"
# PARAMETER num_keep 24
# PARAMETER stop <|start_header_id|>
# PARAMETER stop <|end_header_id|>
# PARAMETER stop <|eot_id|>
# '''

modelfile='''
FROM llama3:latest
SYSTEM You are a seasoned stock market analyst. Your task is to predict the companies' stock price movement for this week based on this week's positive headlines and negative headlines. Just me answer in the format of {increased/decreased/flat} in {X}%. Don't say other things.
TEMPLATE "{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"
PARAMETER num_keep 24
PARAMETER stop <|start_header_id|>
PARAMETER stop <|end_header_id|>
PARAMETER stop <|eot_id|>
'''

# modelfile ='''
# FROM mistral:latest
# SYSTEM You are a seasoned stock market analyst. Your task is to predict the companies' stock price movement for this week based on this week's positive headlines and negative headlines. Just me answer in the format of {increased/decreased/flat} in {X}%. Don't say other things.
# TEMPLATE [INST] {{ if .System }}{{ .System }} {{ end }}{{ .Prompt }} [/INST]

# PARAMETER stop [INST]
# PARAMETER stop [/INST]
# '''

ollama.create(model='stocksense-plus-test', modelfile=modelfile)

# Load jsonl data from disk
dataset = load_dataset("json", data_files="test_dataset.json", split="train")
userPrompts = []
answers = []
generatedAnswers = []
correctResponse = 0

for d in dataset:
    userPrompts.append([{'role': 'user', 'content': d['messages'][1]['content']+ "Just me answer in the format of {increased/decreased/flat} in {X}%. Don't say other things."}])
    # userPrompts.append([{'role': 'user', 'content': d['messages'][1]['content']}])
    groundTruth = d['messages'][2]['content'].split(' ')
    answers.append((groundTruth[0], groundTruth[2]))

for _ in range(3):
    for idx, prompts in enumerate(userPrompts):
        response = ollama.chat(model='stocksense-plus-test', messages=prompts)
        print(response['message']['content'], answers[idx])
        upDown = response['message']['content'].split(' ')[0].lower()
        percentage = response['message']['content'].split(' ')[2]
        # print(upDown.lower(), percentage)
        if answers[idx][0] == upDown:
            correctResponse += 1

# print(answers)
print(correctResponse/len(answers) * 3 * 100, '%')