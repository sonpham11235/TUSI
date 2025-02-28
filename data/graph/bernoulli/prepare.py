import os
import requests
import tiktoken
import random
import numpy as np

def read_file(path):
    try:
        data = []
        with open(path, 'r') as f:
            for line in f:
                line.strip()
                if line:
                    data.append(line)
            return data
    except FileNotFoundError:
        print(f'Invalid file path: {path}')
        return []
    except Exception as e:
        print(f'An error occured while reading file path: {path}')
        return []

# positive samples
possitive_path = os.path.join(os.path.dirname(__file__), 'positive.txt')
# negative samples
negative_path = os.path.join(os.path.dirname(__file__), 'negative.txt')
# edges
edges_path = os.path.join(os.path.dirname(__file__), 'edges.txt')

print('Reading precomputed graph paths')
edges = read_file(edges_path)
positives = read_file(possitive_path)
negatives = read_file(negative_path)

n_edges = len(edges)
n_positive = len(positives)
n_negative = len(negatives)
print(f'Number of edges: {n_edges}')
print(f'Number of positive samples: {n_positive}')
print(f'Number of negative samples: {n_negative}')

print('Preparing train data')
# add all edges and a fraction of simple path in the train set
train = edges + random.sample(positives, int(n_positive*0.15))
# number of current train set
n_train = len(train)
# add void edges into the train set. the number match with positive samples
train += random.sample(negatives, n_train)
# shuffle the train set
random.shuffle(train)
n_train = len(train)
print(f'Number of train samples {n_train}')

print('Preparing validation data')
validation = random.sample(edges, int(n_edges*0.1)) + random.sample(positives, int(n_positive*0.015))
# number of validation
n_validation = len(validation)
# add negative samples to validation set
validation += random.sample(negatives, n_validation)
# shuffle the validation set
random.shuffle(validation)
n_validation = len(validation)
print(f'Number of validation samples {n_validation}')

# # encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary_batch(train)
val_ids = enc.encode_ordinary_batch(validation)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.concatenate([np.array(sample, dtype=np.uint16) for sample in train_ids])
val_ids = np.concatenate([np.array(sample, dtype=np.uint16) for sample in val_ids])
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 18,054 tokens
# val.bin has 1,804 tokens
