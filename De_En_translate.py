
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import spacy, re
import random, math
import dill as pickle
import time, copy
import os


import spacy
import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset

# REFER FOLLOWING TUTORIALS

# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

# https://towardsdatascience.com/how-to-use-torchtext-for-neural-machine-translation-plus-hack-to-make-it-5x-faster-77f3884d95

# https://gmihaila.medium.com/better-batches-with-pytorchtext-bucketiterator-12804a545e2a

# https://www.youtube.com/watch?v=KRgq4VnCr7I

# https://torchtext.readthedocs.io/en/latest/index.html

print('\n')
print('Credits: Samuel Lynn-Evans  https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec')
print('\n')
print('Credits: Samuel Lynn-Evans  https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec')
print('\n')



if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# In[]

# download the data from http://www.statmt.org/europarl/.
# Above link corresponds to the proceedings at the European Parliament 

europarl_en = open('data//de-en//europarl-v7.de-en_short.en', encoding='utf-8').read().split('\n')
europarl_de = open('data//de-en//europarl-v7.de-en_short.de', encoding='utf-8').read().split('\n')

en = spacy.load('en_core_web_sm')
de = spacy.load('de_core_news_sm')


def tokenize_en(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]
    
def tokenize_de(sentence):
    return [tok.text for tok in de.tokenizer(sentence)]
    
DE_TEXT = Field(tokenize=tokenize_de)
EN_TEXT = Field(tokenize=tokenize_en, init_token = "<sos>", eos_token = "<eos>")

raw_data = {'German' : [line for line in europarl_de], 'English': [line for line in europarl_en]}
df = pd.DataFrame(raw_data, columns=["German", "English"])

df['de_len'] = df['German'].str.count(' ')
df['eng_len'] = df['English'].str.count(' ')


# remove sentences where translations are not of roughly equal length.
df = df.query('eng_len < de_len * 1.5 & eng_len * 1.5 > de_len')

# Remove sentences longer than 512. This is a sufficinetly large number. It is highly unlikely that a sentence contains more than 512 words
df = df.query('eng_len < 512 & de_len < 512')

_MAX_SEQ_LEN_ = torch.tensor(512)
_MAX_SEQ_LEN_ = _MAX_SEQ_LEN_.to(device)

# create train and validation set 
train, val = train_test_split(df, test_size=0.1)
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)

data_fields = [('German', DE_TEXT), ('English', EN_TEXT)]
train,val = TabularDataset.splits(path='./', train='train.csv', validation='val.csv', format='csv', fields=data_fields)

EN_TEXT.build_vocab(train, val)
DE_TEXT.build_vocab(train, val)

train_iter = BucketIterator(train, batch_size=3, sort_key=lambda x: len(x.English), shuffle=True)

# batch = next(iter(train_iter))
# print(EN_TEXT.vocab.itos[:99]) 
# print(batch.English)
# print(batch.French)

#In[]

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


# There seems to be a bug in the code. Read through the comments in the medium link.
# Also my comments with the keyword 'correction' in this script

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = _MAX_SEQ_LEN_):   # this code allows only a max of _MAX_SEQ_LEN_=96 words in a sentence
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, d_model)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))

                # Correction: Seems this is an error in the medium code. It should be 2*i instead of 2(i+1)
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i)/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)

        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False).cuda()
        return x


#In[]

# Following class implements Fig. 2 (right) in Vaswani et al.
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads  # floor operator
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):        
        bs = q.size(0)
        
        # perform linear operation and split into h heads        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model/h       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)    
        return output

#In[]

# Following class implements Fig. 2 (left) in Vaswani et al.

def attention(q, k, v, d_k, mask=None, dropout=None):    
    scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

#In[]

# build an encoder layer with one multi-head attention layer and one # feed-forward layer
# Following class implements Fig. 1 (left) in Vaswani et al.

# Here is some ambiguity. In Vaswani et al, norm is after the MultiHeadAttention. However, in this code, 
# it is placed before the MultiHeadAttention as pointed out by many readers in the medium. I modified and
# tried running but the training loss was not converging. Feel free to change the order and try as follows.

# x = x + self.dropout_1(self.attn_1(x, x, x, trg_mask))
# x = self.norm_1(x)

# x = x + self.dropout_2(self.attn_2(x, e_outputs, e_outputs, src_mask))
# x = self.norm_2(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()

        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and one feed-forward layer
# Following class implements Fig. 1 (right) in Vaswani et al.
# same order problem of Norm vs MultiHeadAttn is present here as well

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()

        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).cuda()
        
    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
        
# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

#In[]

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()

        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()

        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()

        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)

        # we don't perform softmax on the output since crossentropy loss function automatically uses logSoftmax
        return output

#In[]

def create_masks(input_seq, target_seq):

    # creates mask with 0s wherever there is padding in the input
    input_pad = torch.tensor(DE_TEXT.vocab.stoi['<pad>'])
    input_pad = input_pad.to(device)
    input_msk = (input_seq != input_pad).unsqueeze(1)

    # create mask as before
    target_pad = torch.tensor(EN_TEXT.vocab.stoi['<pad>'])
    target_pad = target_pad.to(device)
    target_msk = (target_seq != target_pad).unsqueeze(1)

    size = target_seq.size(1)  # get seq_len for matrix
    nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8') 

    nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0)
    nopeak_mask = nopeak_mask.to(device)
    target_msk = target_msk & nopeak_mask

    return input_msk, target_msk, target_pad    

#In[]

def translate(model, src, max_len = _MAX_SEQ_LEN_, custom_string=False):
    
    model.eval()

    if custom_string:
        src = tokenize_de(src)
        src = Variable(torch.LongTensor([[DE_TEXT.vocab.stoi[tok] for tok in src]])).cuda()

    input_pad = torch.tensor(DE_TEXT.vocab.stoi['<pad>'])
    input_pad = input_pad.to(device)    
    src_mask = (src != input_pad).unsqueeze(-2)
    src_mask.to(device)
    e_outputs = model.encoder(src, src_mask)
    
    outputs = torch.zeros(max_len).type_as(src.data)
    outputs[0] = torch.LongTensor([EN_TEXT.vocab.stoi['<sos>']])
    
    for i in range(1, max_len):                
        trg_mask = np.triu(np.ones((1, i, i)), k=1).astype('uint8')
        trg_mask = Variable(torch.from_numpy(trg_mask) == 0).cuda()
        out = model.out(model.decoder(outputs[:i].unsqueeze(0), e_outputs, src_mask, trg_mask))

        # In validation/testing, softmax is explicitly required. We did not perform softmax during trainig because
        # cross entropy loss function implicitly performs logsoftmax 
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)        
        outputs[i] = ix[0][0]

        # This is the stopping criteria. When the predicted token corresponds to '<eos>', prediction
        # is completed. Hence exit
        if ix[0][0] == EN_TEXT.vocab.stoi['<eos>']:
            break

    return ' '.join([EN_TEXT.vocab.itos[ix] for ix in outputs[:i]])

#In[]

d_model = 512  # dimension used throughout the different layers
heads = 8  # No: of parallel attention layers in Multihead attention layer
N = 6  # No: of stacked identical Encoder and Decoder layers

src_vocab = len(DE_TEXT.vocab)
trg_vocab = len(EN_TEXT.vocab)

model = Transformer(src_vocab, trg_vocab, d_model, N, heads)
model = model.to(device)


for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

 
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

epochs = 30
print_every = 1

model.train()

start = time.time()
temp = start
total_loss = 0

for epoch in range(epochs):    
    for i, batch in enumerate(train_iter): 

        optim.zero_grad()        
        src = batch.German.transpose(0,1)
        trg = batch.English.transpose(0,1)

        src = src.to(device)
        trg = trg.to(device)
        
        # the English sentence we input has all words except the last, as it is using each word to predict the next
        trg_input = trg[:, :-1]
        
        # the words we are trying to predict            
        targets = trg[:, 1:].contiguous().view(-1)
        
        # create function to make masks using mask code above
        
        src_mask, trg_mask, target_pad = create_masks(src, trg_input)
        
        preds = model(src, trg_input, src_mask, trg_mask)
                
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), targets, ignore_index=target_pad)
        loss.backward()
        optim.step()        

        total_loss += loss.item()

        if (i + 1) % print_every == 0:
            loss_avg = total_loss / print_every
            print("time = %dm, epoch %d, iter = %d, loss = %.3f,\
            %ds per %d iters" % ((time.time() - start) // 60,\
            epoch + 1, i + 1, loss_avg, time.time() - temp,\
            print_every))
            total_loss = 0
            temp = time.time()

#In[] 
    # Following is the validation/testing API. 

    # val_iter = BucketIterator(train, batch_size=1, sort_key=lambda x: len(x.English), shuffle=True)
    
    # for id, val_batch in enumerate(val_iter): 
    #     val_src = val_batch.German.transpose(0,1)
    #     val_true_trg = val_batch.English.transpose(0,1) 
        
    #     val_src = val_src.to(device)
    #     val_true_trg = val_true_trg.to(device)

    #     trg_pred = translate(model, val_src)


print('\n')
print('Credits: Samuel Lynn-Evans  https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec')
print('\n')
print('Credits: Samuel Lynn-Evans  https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec')
print('\n')
