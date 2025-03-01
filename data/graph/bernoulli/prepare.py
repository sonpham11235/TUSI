import os
import requests
import tiktoken
import random
import numpy as np

special_tokens = {'X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23','X24','X25','X26','X27','X28','X29','X30','X31','X32','X33','X34','X35','X36','X37','X38','X39','X40','X41','X42','X43','X44','X45','X46','X47','X48','X49','X50','X51','X52','X53','X54','X55','X56','X57','X58','X59','X60','X61','X62','X63','X64','X65','X66','X67','X68','X69','X70','X71','X72','X73','X74','X75','X76','X77','X78','X79','X80','X81','X82','X83','X84','X85','X86','X87','X88','X89','X90','X91','X92','X93','X94','X95','X96','X97','X98','X99','X100','X101','X102','X103','X104','X105','X106','X107','X108','X109','X110','X111','X112','X113','X114','X115','X116','X117','X118','X119','X120','X121','X122','X123','X124','X125','X126','X127','X128','X129','X130','X131','X132','X133','X134','X135','X136','X137','X138','X139','X140','X141','X142','X143','X144','X145','X146','X147','X148','X149','X150','X151','X152','X153','X154','X155','X156','X157','X158','X159','X160','X161','X162','X163','X164','X165','X166','X167','X168','X169','X170','X171','X172','X173','X174','X175','X176','X177','X178','X179','X180','X181','X182','X183','X184','X185','X186','X187','X188','X189','X190','X191','X192','X193','X194','X195','X196','X197','X198','X199','X200','p0','p1','<|endoftext|>','<|question|>','<|answer|>'}

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
train_ids = []
val_ids = []
for sample in train:
    train_ids.append(enc.encode(sample, allowed_special=special_tokens))
for sample in validation:
    val_ids.append(enc.encode(sample, allowed_special=special_tokens))

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.concatenate([np.array(sample, dtype=np.uint16) for sample in train_ids])
val_ids = np.concatenate([np.array(sample, dtype=np.uint16) for sample in val_ids])
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 18,054 tokens
# val.bin has 1,804 tokens
