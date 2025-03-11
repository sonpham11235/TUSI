"""
Sample from a trained model
"""
import streamlit as st
import os
import networkx as nx
from contextlib import nullcontext
import torch
import tiktoken
import graphviz
from model import GPTConfig, GPT

# python sample.py --device=cpu --out_dir=out-graph --start="X130 X1" --num_samples=1 --max_new_tokens=100

class model:
    def __init__(self, out_dir):
        self.special_tokens = {'X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23','X24','X25','X26','X27','X28','X29','X30','X31','X32','X33','X34','X35','X36','X37','X38','X39','X40','X41','X42','X43','X44','X45','X46','X47','X48','X49','X50','X51','X52','X53','X54','X55','X56','X57','X58','X59','X60','X61','X62','X63','X64','X65','X66','X67','X68','X69','X70','X71','X72','X73','X74','X75','X76','X77','X78','X79','X80','X81','X82','X83','X84','X85','X86','X87','X88','X89','X90','X91','X92','X93','X94','X95','X96','X97','X98','X99','X100','X101','X102','X103','X104','X105','X106','X107','X108','X109','X110','X111','X112','X113','X114','X115','X116','X117','X118','X119','X120','X121','X122','X123','X124','X125','X126','X127','X128','X129','X130','X131','X132','X133','X134','X135','X136','X137','X138','X139','X140','X141','X142','X143','X144','X145','X146','X147','X148','X149','X150','X151','X152','X153','X154','X155','X156','X157','X158','X159','X160','X161','X162','X163','X164','X165','X166','X167','X168','X169','X170','X171','X172','X173','X174','X175','X176','X177','X178','X179','X180','X181','X182','X183','X184','X185','X186','X187','X188','X189','X190','X191','X192','X193','X194','X195','X196','X197','X198','X199','X200','p0','p1','<|endoftext|>','<|question|>','<|answer|>'}
        self.start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
        self.num_samples = 1 # number of samples to draw
        self.max_new_tokens = 32 # number of tokens generated in each sample
        self.temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        self.top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
        self.seed = 1337
        self.device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
        self.compile = False # use PyTorch 2.0 to compile the model to be faster
        exec(open('configurator.py').read()) # overrides from command line or config file

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        device_type = 'cuda' if 'cuda' in self.device else 'cpu' # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        self.model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

        # let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        self.encode = lambda s: enc.encode(s, allowed_special=self.special_tokens)
        self.decode = lambda l: enc.decode(l)

    def infer(self, prompt):
        start_ids = self.encode(prompt)
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])
        with torch.no_grad():
            with self.ctx:
                for k in range(self.num_samples):
                    y = self.model.generate(x, self.max_new_tokens, temperature=self.temperature, top_k=self.top_k)
                    model_output = str(self.decode(y[0].tolist()))
                    meaningfull = model_output.split('<|endoftext|>')[0]
                    return meaningfull

@st.cache_resource
def load_model():
    return model('stepwise-ckpt')

@st.cache_resource
def load_graph() -> nx.DiGraph:
    return nx.read_adjlist('./data/graph/bernoulli/bernoulli.adjlist', create_using=nx.DiGraph)

def build_subgraph(paths):
    subgraph = graphviz.Digraph(strict=True)
    for path in paths:
        for i in range(len(path)-1):
            subgraph.edge(path[i], path[i+1])
    return subgraph

col1, col2 = st.columns(2)
with col1:
    source = st.text_input('Source node')

with col2:
    target = st.text_input('Target node')

if source and target:
    # infer
    prompt = '<|question|>' + target + ' ' + source
    model = load_model()
    answer = model.infer(prompt)
    # answer
    st.write(answer)
    # subgraph involved
    graph = load_graph()
    paths = nx.all_simple_paths(graph, source, target, 6)
    subgraph = build_subgraph(paths)
    st.graphviz_chart(subgraph)