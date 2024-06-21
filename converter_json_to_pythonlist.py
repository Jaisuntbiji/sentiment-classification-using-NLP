import json

file_path = 'E:\\NPL\\Sarcasm.json'

sentences = []
labels = []
urls = []

with open(file_path, 'r') as f:
    for line in f:
        if line.strip():  # Make sure the line isn't empty
            item = json.loads(line)
            sentences.append(item['headline'])
            labels.append(item['is_sarcastic'])
            urls.append(item['article_link'])

# print(sentences)
# print(labels)
# print(urls)

