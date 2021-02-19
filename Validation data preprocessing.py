import tensorflow as tf
import tensorflow_datasets
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from collections import Counter
import numpy as np
import pickle

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

with open('word_tokenizer.pkl', 'rb') as handle:
  word_tokenizer = pickle.load(handle)

with open('char_tokenizer.pkl', 'rb') as handle:
  char_tokenizer = pickle.load(handle)

ds = tensorflow_datasets.load('squad/v1.1')
dataset = tensorflow_datasets.as_numpy(ds)
def get_tokens(txt):
    return [r.text for r in tokenizer(txt)]

def get_char(token):
    ans = []
    for i in token:
        ans.append(list(i))
    return ans

con_w, con_c, que_w, que_c,y1, y2 = [], [], [], [], [], []
break_flag = 0
answer_ary = []
for i in dataset['validation']:
            
    context = i['context'].decode('utf-8')
    query = i['question'].decode('utf-8')

    c_token = get_tokens(context)
    con_w.append(c_token)

    c_tmp = get_char(c_token)
    con_c.append(c_tmp)
   
  
    q_token = get_tokens(query)
    que_w.append(q_token)
    
    q_tmp = get_char(q_token)
    que_c.append(q_tmp)
    
    start_pos = len(get_tokens(context[:i['answers']['answer_start'][0]]))
    end_pos = start_pos + len(get_tokens(i['answers']['text'][0].decode('utf-8'))) - 1
    answer_ary.append([i['answers']['text'][0].decode('utf-8')])

    y1.append(start_pos)
    y2.append(end_pos)

token_context = [[i for i in sen if i != ' '] for sen in context_token]
token_context = equalize_length(token_context, 809)
def get_chartoken(x):
    ch_token = []
    for sen in x:
        tmp1 = []
        for words in sen:
            tmp = []
            for i in words:
                if i in char_tokenizer:
                    tmp.append(char_tokenizer[i])
                elif i != ' ':
                    tmp.append(char_tokenizer['OOV'])
                
            tmp1.append(tmp)
        ch_token.append(tmp1)
    return ch_token

def get_wordtoken(x):
    w_token = []
    for sen in x:
        tmp1 = []
        for words in sen:
            if words in word_tokenizer:
                tmp1.append(word_tokenizer[words])
            elif words != ' ':
                tmp1.append(11)
                
        w_token.append(tmp1)
    return w_token


query_ch_token = get_chartoken(que_c)
context_ch_token = get_chartoken(con_c)
query_w_token = get_wordtoken(que_w)
context_w_token = get_wordtoken(con_w)


from tensorflow.keras.preprocessing.sequence import pad_sequences
max_word_que, max_word_con, max_ch_con, max_ch_que = 60,809,37,26
def equalize_length(txt, m_len):
    return pad_sequences(txt, maxlen = m_len, padding = 'post')
query_ch = []
for i in range(len(query_ch_token)):
    query_ch.append(equalize_length(query_ch_token[i], max_ch_que))
    
query_ch = equalize_length(query_ch, max_word_que)

context_ch = []
for i in range(len(context_ch_token)):
    context_ch.append(equalize_length(context_ch_token[i], max_ch_con))
context_ch = equalize_length(context_ch, max_word_con)
query_w_token = equalize_length(query_w_token, max_word_que)
context_w_token = equalize_length(context_w_token, max_word_con)


with open('val_query_char.pkl', 'wb') as handle:
    pickle.dump(query_ch, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('val_context_char.pkl', 'wb') as handle:
    pickle.dump(context_ch, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('val_query_word_token.pkl', 'wb') as handle:
    pickle.dump(query_w_token, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('val_context_word_token.pkl', 'wb') as handle:
    pickle.dump(context_w_token, handle, protocol=pickle.HIGHEST_PROTOCOL)

query_ch, context_ch, query_w_token, context_w_token =  [], [], [], [] 

with open('val_y1_start.pkl', 'wb') as handle:
    pickle.dump(y1, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('val_y2_end.pkl', 'wb') as handle:
    pickle.dump(y2, handle, protocol=pickle.HIGHEST_PROTOCOL)