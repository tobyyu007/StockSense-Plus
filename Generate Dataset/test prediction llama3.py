import ollama
from datasets import load_dataset


stocksense_modelfile='''
FROM /Users/jackchen/Downloads/Stocksense-Plus-Prediction-Q4_K_M.gguf
SYSTEM You are a seasoned stock market analyst. Your task is to predict the companies' stock price movement for this week based on this week's positive headlines and negative headlines. Give me answer in the format of {increased/decreased/flat} in {X}%
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

mistral_modelfile = '''
FROM mistral:latest
SYSTEM You are a seasoned stock market analyst. Your task is to predict the companies' stock price movement for this week based on this week's positive headlines and negative headlines. Just give me the prediction only in the format of {increased/decreased/flat} in {X}% and don't say any other things such as the stock ticker in the response.
TEMPLATE [INST] {{ if .System }}{{ .System }} {{ end }}{{ .Prompt }} [/INST]

PARAMETER stop [INST]
PARAMETER stop [/INST]
'''

# change the model to the model you want to use
model = 'stocksense'
modelfile = {'stocksense': stocksense_modelfile, 'original_llama3': original_llama3_modelfile, 'mistral': mistral_modelfile}

ollama.create(model='stocksense-plus-test', modelfile=modelfile[model])

# Load jsonl data from disk
dataset = load_dataset("json", data_files="test_dataset.json", split="train")
userPrompts = []
answers = []
generatedAnswers = []
correct_upDown_Response = 0
correct_Response = 0
accuracies = []

for d in dataset:
    if model == 'stocksense' or model == 'original_llama3':
        # stocksense and llama3
        userPrompts.append([{'role': 'user', 'content': d['messages'][1]['content']+ "Just me answer in the format of {increased/decreased/flat} in {X}%. Don't say other things."}])
    elif model == 'mistral':
        # mistral
        userPrompts.append([{'role': 'user', 'content': d['messages'][1]['content']+ "Just give me the prediction only in the format of {increased/decreased/flat} in {X}% and don't say any other things such as the stock ticker in the response."}])

    # userPrompts.append([{'role': 'user', 'content': d['messages'][1]['content']}])
    groundTruth = d['messages'][2]['content'].split(' ')
    answers.append((groundTruth[0], groundTruth[2]))

for i in range(3):
    for idx, prompts in enumerate(userPrompts):
        response = ollama.chat(model='stocksense-plus-test', messages=prompts)
        print(response['message']['content'], answers[idx])
        if len(response['message']['content'].split(' ')) < 7:
            if model == 'stocksense' or model == 'original_llama3':
                upDown = response['message']['content'].split(' ')[0].lower()
            elif model == 'mistral':
                upDown = response['message']['content'].split(' ')[1].lower()
            if model == 'stocksense' or model == 'original_llama3':
                percentage = response['message']['content'].split(' ')[-1]
            elif model == 'mistral':
                if '{' in response['message']['content']:
                    # parse the response to get the percentage {x-x}% to x-x%
                    percentage = response['message']['content'].split(' ')[-1].replace('{', '').replace('}', '')
                else:
                    # select the one with the percentage
                    for word in response['message']['content'].split(' '):
                        if '%' in word:
                            percentage = word
                            break
        else:
            upDown = 'Neither'
            percentage = '9999%'
        
        if '-' in percentage:
            lower_percentage = float(percentage.split('-')[0])
            upper_percentage = float(percentage.split('-')[1].replace('%', '').replace('.', ''))
        elif percentage[-1] == '.' and percentage[-2] == '%':
            lower_percentage = float(percentage[:-2])
            upper_percentage = float(percentage[:-2])
        elif percentage[-1] == '%':
            lower_percentage = float(percentage[:-1])
            higher_percentage = float(percentage[:-1])
        elif percentage[-1] == '.':
            lower_percentage = float(percentage[:-1])
            higher_percentage = float(percentage[:-1])
        
        if answers[idx][0] == upDown:
            print('Correct upDown')
            correct_upDown_Response += 1
        
            if answers[idx][0] == upDown and ((abs(float(answers[idx][1][:-1]) - lower_percentage) <= 1.0) or (abs(higher_percentage - float(answers[idx][1][:-1])) <= 1.0)):
                print('Correct upDown and percentage')
                correct_Response += 1
        print('\n')
    
    upDown_acc = correct_upDown_Response/len(answers) * 100
    percentage_acc = correct_Response/correct_upDown_Response * 100
    print(f"Correct Up/Down epoch {i}: {upDown_acc}%")
    print(f"Both Correct epoch {i}: {percentage_acc}%")
    accuracies.append((upDown_acc, percentage_acc))
    correct_Response = 0
    correct_upDown_Response = 0

print(accuracies)
# print Average of upDown and percentage accuracy
upDown_acc = sum([acc[0] for acc in accuracies])/len(accuracies)
percentage_acc = sum([acc[1] for acc in accuracies])/len(accuracies)
print(f"Average Up/Down: {upDown_acc}%")
print(f"Average Both: {percentage_acc}%")

# 1st
# [(54.0, 34.0)]

# 2nd
# [(56.00000000000001, 26.0)]

# 3rd
# [(50.0, 22.0)]

# Avg
# Average Up/Down: 53.333333333333336%
# Average Both: 27.333333333333336%