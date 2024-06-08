import pandas as pd
from datasets import load_dataset

# Load jsonl data from disk
dataset = load_dataset("json", data_files="sentiment.json", split="train")

# Prepare a list to collect rows, which later convert into a DataFrame
output = []

for d in dataset:
    # Extract the content of the first and second message
    first_message_content = d['messages'][0]['content']
    second_message_content = d['messages'][1]['content']
        
    # Find the indices of the curly braces
    index_of_open_brace = first_message_content.index('{')
    index_of_close_brace = first_message_content.index('}')

    if first_message_content[index_of_open_brace+1:index_of_close_brace] == 'negative/neutral/positive':
        # check if the second message contains space at the end; if so remove it
        if second_message_content[-1] == ' ':
            second_message_content = second_message_content[:-1]
        # check if the second message contains space + period at the last two characters; if so remove them
        if second_message_content[-2:] == ' .':
            second_message_content = second_message_content[:-2]
        # check if the second message contains period at the end or not
        if second_message_content[-1] != '.':
            second_message_content += '.'
        
        input_text = second_message_content + " Don't say other things, just give me answer in  {negative/neutral/positive} for your response."
        output.append({
            'Instruction': first_message_content, 
            'Input': input_text, 
            'Output': d['messages'][2]['content']
        })

# Convert the list of dictionaries to a DataFrame
output = pd.DataFrame(output)

print(output.shape)
output.to_csv('./dataset/output/sentiment_output.csv', index=False)