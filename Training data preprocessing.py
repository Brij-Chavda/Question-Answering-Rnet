import tensorflow as tf
import tensorflow_datasets
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from collections import Counter
import numpy as np
import pickle
nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

ds = tensorflow_datasets.load('squad/v1.1')
dataset = tensorflow_datasets.as_numpy(ds)

def get_tokens(txt):
    return [r.text for r in tokenizer(txt)]

def get_char(token):
    ans = []
    mlen = 0
    for i in token:
        ans.append(list(i))
        if mlen < len(list(i)):
            mlen = len(list(i))
    return ans, mlen

char_counter, token_counter = Counter(' '), Counter(' ')
max_ch_con, max_word_con, max_ch_que ,max_word_que = 0, 0, 0, 0
y1, y2, context_token, query_token, context_ch, query_ch = [], [], [], [], [], []

for i in dataset['train']:
    
    context = i['context'].decode('utf-8')
    #context = context.replace("'", "`")
    char_counter = char_counter + Counter(context)
    query = i['question'].decode('utf-8')
    char_counter = char_counter + Counter(query)
    c_token = get_tokens(context)
    token_counter = token_counter + Counter(c_token)
    context_token.append(c_token)
    c_tmp, mlenc = get_char(c_token)
    char_counter = char_counter + Counter(sum(map(Counter, c_tmp), Counter())) 
    context_ch.append(c_tmp)
   
    q_token = get_tokens(query)
    token_counter = token_counter + Counter(q_token)
    query_token.append(q_token)
    q_tmp, mlenq = get_char(q_token)
    char_counter = char_counter + Counter(sum(map(Counter, q_tmp), Counter())) 
    query_ch.append(q_tmp)
    max_ch_con = max(max_ch_con, mlenc)
    max_word_con = max(max_word_con, len(c_token))
    max_word_que = max(max_word_que, len(q_token))
    max_ch_que = max(max_ch_que, mlenq)    
    
    start_pos = len(get_tokens(context[:i['answers']['answer_start'][0]]))
    end_pos = start_pos + len(get_tokens(i['answers']['text'][0].decode('utf-8'))) - 1
   
    y1.append(start_pos)
    y2.append(end_pos)
    
    

del char_counter[' ']
del token_counter[' ']

count = 2
word_tokenizer = {}
for i in token_counter.keys():
    word_tokenizer[i] = count
    count = count + 1
token_counter = {val: key for key,val in word_tokenizer.items()}
vocab_size = count
    
count = 2
char_tokenizer = {}
for i in char_counter.keys():
    char_tokenizer[i] = count
    count = count + 1
char_counter = {val: key for key,val in char_tokenizer.items()}
char_size = count

word_tokenizer['NULL'], word_tokenizer['OOV'] = 0, 1
char_tokenizer['NULL'], char_tokenizer['OOV'] = 0, 1

def load_wordvector(gloveFile):
    f = open(gloveFile,'r', encoding='utf8')
    model = {}
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = list(map(float, splitLine[1:]))
        model[word] = embedding
    print('Glove loaded successful')
    return model

with open('word_tokenizer.pkl', 'wb') as handle:
    pickle.dump(word_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('char_tokenizer.pkl', 'wb') as handle:
    pickle.dump(char_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('char_counter.pkl', 'wb') as handle:
    pickle.dump(char_counter, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('token_counter.pkl', 'wb') as handle:
    pickle.dump(token_counter, handle, protocol=pickle.HIGHEST_PROTOCOL)

char_counter, token_counter = {}, {}

query_ch_token = [[[char_tokenizer[i] for i in words if i != ' '] for words in sen] for sen in query_ch]
context_ch_token = [[[char_tokenizer[i] for i in words if i != ' '] for words in sen] for sen in context_ch]
query_w_token = [[word_tokenizer[i] for i in sen if i != ' '] for sen in query_token]
context_w_token = [[word_tokenizer[i] for i in sen if i != ' '] for sen in context_token]

char_tokenizer = {}

query_ch, context_ch, query_token, context_token = [], [], [], [] 


from tensorflow.keras.preprocessing.sequence import pad_sequences
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


with open('query_ch.pkl', 'wb') as handle:
    pickle.dump(query_ch, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('context_ch.pkl', 'wb') as handle:
    pickle.dump(context_ch, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('query_w_token.pkl', 'wb') as handle:
    pickle.dump(query_w_token, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('context_w_token.pkl', 'wb') as handle:
    pickle.dump(context_w_token, handle, protocol=pickle.HIGHEST_PROTOCOL)

query_ch, context_ch, query_w_token, context_w_token =  [], [], [], [] 

with open('y1.pkl', 'wb') as handle:
    pickle.dump(y1, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('y2.pkl', 'wb') as handle:
    pickle.dump(y2, handle, protocol=pickle.HIGHEST_PROTOCOL)

glove = load_wordvector('glove.840B.300d.txt')
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in word_tokenizer.items():
    if glove.get(word):
        embedding_matrix[i] = glove.get(word)
    elif glove.get(word.lower()):
        embedding_matrix[i] = glove.get(word.lower())
    elif glove.get(word.upper()):
        embedding_matrix[i] = glove.get(word.upper())



with open('embedding.pkl', 'wb') as handle:
    pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

