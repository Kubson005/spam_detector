import csv
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import re

rows = []

with open("emails.csv", "r") as file:
    data = csv.reader(file)
    counter = 0
    for i in data:
        if counter == 0:
            counter += 1
            continue
        rows.append(i)


text = []
label = []

for i in range(len(rows)):
    text.append(rows[i][0])
    label.append(int(rows[i][1]))

text, test_text, label, test_label = train_test_split(
    text,
    label,
    test_size=int(0.1*len(text)),
    random_state=42,
    stratify=label
)


def clean():
    url = r'https?://\S+|www\.\S+'
    mail = r'\S+@\S+'

    for i in range(len(text)):
        text[i] = text[i].lower()
        text[i] = re.sub(' +', ' ', text[i])
        text[i] = re.sub(url, '', text[i])
        text[i] = re.sub(mail, '', text[i])
        text[i] = re.sub(r'[^a-zA-Z0-9 ]', '', text[i])
        text[i] = re.sub(r' +', ' ', text[i]).strip()
    
    for i in range(len(test_text)):
        test_text[i] = test_text[i].lower()
        test_text[i] = re.sub(' +', ' ', test_text[i])
        test_text[i] = re.sub(url, '', test_text[i])
        test_text[i] = re.sub(mail, '', test_text[i])
        test_text[i] = re.sub(r'[^a-zA-Z0-9 ]', '', test_text[i])
        test_text[i] = re.sub(r' +', ' ', test_text[i]).strip()


tokens = []
test_tokens = []
tokens_dict = {"<PAD>": 0, "<UNK>": 1}
sequences = []
test_sequences = []

def tokenize():
    counter = Counter()

    for i in range(len(text)):
        tokens.append(text[i].split())

    for i in range(len(test_text)):
        test_tokens.append(test_text[i].split())
    
    for mail in tokens:
        counter.update(mail)

    for word, _ in counter.most_common():
        tokens_dict[word] = len(tokens_dict)
    
    for mail in tokens:
        seq = [tokens_dict.get(word, tokens_dict["<UNK>"]) for word in mail]
        sequences.append(seq)

    for mail in test_tokens:
        seq = [tokens_dict.get(word, tokens_dict["<UNK>"]) for word in mail]
        test_sequences.append(seq)

    return (
        pad_sequences(sequences, maxlen=100, padding="post", truncating="post"),
        pad_sequences(test_sequences, maxlen=100, padding="post", truncating="post"),
    )
    
    

clean()
X, test_X = tokenize()

print(X[0])
