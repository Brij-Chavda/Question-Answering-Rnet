from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Masking

from collections import Counter
import numpy as np
import pickle

import tensorflow as tf
#import tensorflow_datasets
#tf.compat.v1.disable_eager_execution()

class encoder(tf.keras.Model):

    def __init__(self, embedding_matrix, num_units, vocab_size):
        super(encoder, self).__init__()
        
        self.embedding_matrix = embedding_matrix
        #self.char_size = char_len
        self.vocab_size = vocab_size
        self.ch_gru_con = Bidirectional(GRU(100, return_sequences = True, return_state = False, dropout=0.1))
        self.ch_gru_que = Bidirectional(GRU(100, return_sequences = True, return_state = False, dropout=0.1))
        self.gru_layer_con1 = Bidirectional(GRU(num_units, return_sequences = True, return_state = False, dropout=0.1))
        self.gru_layer_con2 = Bidirectional(GRU(num_units, return_sequences = True, return_state = False, dropout=0.1))
        self.gru_layer_que1 = Bidirectional(GRU(num_units, return_sequences = True, return_state = False, dropout=0.1))
        self.gru_layer_que2 = Bidirectional(GRU(num_units, return_sequences = True, return_state = False, dropout=0.1))
        self.final_grulayer_con = Bidirectional(GRU(num_units, return_sequences = True, return_state = True, dropout=0.1))
        self.final_grulayer_que = Bidirectional(GRU(num_units, return_sequences = True, return_state = True, dropout=0.1))
        #self.embedding_mask = Embedding(self.vocab_size, 300, weights=[self.embedding_matrix], trainable=False, mask_zero=True)
        self.embedding = Embedding(self.vocab_size, 300, weights=[self.embedding_matrix], trainable=False, mask_zero=True)
        #self.con_mask = Masking(mask_value = 0.0, input_shape = (809,))
        #self.que_mask = Masking(mask_value = 0.0, input_shape = (60,))
        '''self.ch_gru_con._could_use_gpu_kernel = False
        self.ch_gru_que._could_use_gpu_kernel = False
        self.gru_layer_con1._could_use_gpu_kernel = False
        self.gru_layer_con2._could_use_gpu_kernel = False
        self.gru_layer_que1._could_use_gpu_kernel = False
        self.gru_layer_que2._could_use_gpu_kernel = False
        self.final_grulayer_con._could_use_gpu_kernel = False
        self.final_grulayer_que._could_use_gpu_kernel = False'''

        #self.query_embedding = Embedding(self.vocab_size, 300, weights=[self.embedding_matrix], trainable=False)
        
    def __call__(self, w_cont, w_query, ch_cont, ch_query):
        
        #context_cvec = Embedding(self.char_size, self.char_size, trainable=True, mask_zero = True)(ch_cont)
        #query_cvec = Embedding(self.char_size, self.char_size, trainable=True, mask_zero = True)(ch_query)

        context_wvec = self.embedding(w_cont)
        query_wvec = self.embedding(w_query)

        #contextm = self.embedding_mask(w_cont)
        #querym = self.embedding_mask(w_query)

        c_mask = tf.cast((w_cont), tf.bool)
        q_mask = tf.cast((w_query), tf.bool)

        #c_mask = tf.cast(self.con_mask(w_cont), tf.bool)
        #q_mask = tf.cast(self.que_mask(w_query), tf.bool)   
        #c_mask = self.con_mask(w_cont)
        #q_mask = self.que_mask(w_query)

        context_cvec = self.ch_gru_con(ch_cont)
        query_cvec = self.ch_gru_que(ch_query)
        
        context_embed = Concatenate(axis = 2)([context_wvec, context_cvec])
        query_embed = Concatenate(axis = 2)([query_wvec, query_cvec])

        con_en1 = self.gru_layer_con1(context_embed)
        con_en2 = self.gru_layer_con2(con_en1)
        con_en3, c_state1, c_state2 = self.final_grulayer_con(con_en2)

        que_en1 = self.gru_layer_que1(query_embed)
        que_en2 = self.gru_layer_que2(que_en1)
        que_en3, q_state1, q_state2 = self.final_grulayer_que(que_en2)
        
        return con_en3, c_state1, c_state2, que_en3, q_state1, q_state2, c_mask, q_mask
        
    
class GatedAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(GatedAttention, self).__init__()
        self.W1 = Dense(units, use_bias = False)
        self.W2 = Dense(units, use_bias = False)
        self.W3 = Dense(units, use_bias = False)
        self.ans_Dense = Dense(4*units, use_bias = False)
        self.V = Dense(1)

    def __call__(self, query, passage, hidden, mask):
        hidden_passage = tf.expand_dims(hidden, 1)
        passage_dim = tf.expand_dims(passage, 1)
        score = self.V(tf.nn.tanh(self.W1(query) + self.W2(passage_dim) + self.W3(hidden_passage)))
        mask_value = tf.expand_dims((1 - mask) * -1e30, 2)
        mask_score = mask_value + score
        attention_weights = tf.nn.softmax(mask_score, axis=1)
        context_vector = attention_weights * query
        context_vector = tf.reduce_sum(context_vector, axis=1)
        ans = tf.concat([passage, context_vector], axis = 1)
        shape = np.shape(ans)
        ans = self.ans_Dense(ans)
        g = tf.math.sigmoid(ans)

        return g * ans, attention_weights

class GatedAttentionDecoder(tf.keras.Model):
    def __init__(self, dec_units):
        super(GatedAttentionDecoder, self).__init__()
        self.dec_units = dec_units
        self.gru = Bidirectional(GRU(self.dec_units, return_sequences = True, return_state = True))
        self.attention = GatedAttention(self.dec_units)
        #self.gru._could_use_gpu_kernel = False
        
    def __call__(self, query, passage, hidden, mask): # call by passage particular time instant, consider passage as decoder input
        context_vector, weights = self.attention(query, passage, hidden, mask)
        context_vector = tf.expand_dims(context_vector, 1)
        output, state1, state2 = self.gru(context_vector)
        state = tf.concat([state1, state2], axis=-1)
        return output, state, weights


class SelfAttention(tf.keras.Model):
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.W1 = Dense(units, use_bias = False)
        self.W2 = Dense(units, use_bias = False)
        self.V = Dense(units)
        
    def __call__(self, passage1, passage2, mask):#passage 2 decoder input
        passage1 = tf.nn.relu(self.W1(passage1))
        passage2 = tf.nn.relu(self.W2(passage2))
        score = tf.matmul(passage1, tf.transpose(passage2, [0, 2, 1]))
        mask_value = tf.expand_dims((1 - mask) * -1e30, 2)
        attention_weights = tf.nn.softmax((score + mask_value), axis=1)
        context_vector = tf.matmul(attention_weights, passage1)
        
        concate = tf.concat([passage2, context_vector], axis = 2)
        
        return concate, attention_weights

class InitialStateoutput(tf.keras.Model):
    def __init__(self, units):
        super(InitialStateoutput, self).__init__()
        self.W1 = Dense(units, use_bias = False)
        self.W2 = Dense(units, use_bias = False)
        self.V = Dense(1)
        
    def __call__(self, query, query_hidden, mask):
        query_hidden = tf.expand_dims(query_hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(query) + self.W2(query_hidden)))
        mask_value = tf.expand_dims((1 - mask) * -1e30, 2)
        mask_score = mask_value + score
        attention_weights = tf.nn.softmax(mask_score, axis=1)
        context_vector = attention_weights * query
        context_vector = tf.reduce_sum(context_vector, axis=1)
    
        return context_vector, attention_weights

class OutputAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(OutputAttention, self).__init__()
        self.W1 = Dense(units, use_bias = False)
        self.W2 = Dense(units, use_bias = False)
        self.V = Dense(1)
        
    def __call__(self, passage, hidden, mask):#passage 2 decoder input
        hidden = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(passage) + self.W2(hidden)))
        mask_value = tf.expand_dims((1 - mask) * -1e30, 2)
        attention_weights = tf.nn.softmax((score + mask_value), axis=1)
        context_vector = attention_weights * passage
        context_vector = tf.reduce_sum(context_vector, axis=1)
    
        return context_vector, attention_weights

class OutputDecoder(tf.keras.Model):
    def __init__(self, dec_units):
        super(OutputDecoder, self).__init__()
        self.dec_units = dec_units
        self.gru = Bidirectional(GRU(self.dec_units, return_sequences = True, return_state = True))
        self.attention = OutputAttention(self.dec_units)
        #self.gru._could_use_gpu_kernel = False
    
    def __call__(self, passage, hidden, mask): # call by attention hidden state particular time instant, consider passage as decoder input
        context_vector, weights = self.attention(passage, hidden, mask)
        context_vector = tf.expand_dims(context_vector, 1)
        output, state1, state2 = self.gru(context_vector)
        state = tf.concat([state1, state2], axis=-1)
        return output, state, weights

class SAdecode(tf.keras.Model):
    def __init__(self, dec_units):
        super(SAdecode, self).__init__()
        self.dec_units = dec_units
        self.gru = Bidirectional(GRU(self.dec_units, return_sequences = True, return_state = False, dropout=0.05))
        #self.gru._could_use_gpu_kernel = False

    def __call__(self,input):
        output = self.gru(input)
        return output


def evalution():
    with open('val_context_char.pkl', 'rb') as handle:
        eval_c_ch = pickle.load(handle)
    with open('val_query_char.pkl', 'rb') as handle:
        eval_q_ch = pickle.load(handle)
    with open('val_context_word_token.pkl', 'rb') as handle:
        eval_c_w = pickle.load(handle)
    with open('val_query_word_token.pkl', 'rb') as handle:
        eval_q_w = pickle.load(handle)

    em, f1 = 0, 0
    eval_batch_size = 64
    itr = int(np.floor(len(eval_c_w) / batch_size))
    for i in range(itr):
        eval_start = i*eval_batch_size
        eval_end = eval_start + eval_batch_size
        tmp_f1, tmp_em = evaluate_step(eval_c_w[eval_start:eval_end], eval_q_w[eval_start:eval_end], eval_c_ch[eval_start:eval_end], eval_q_ch[eval_start:eval_end], eval_start)
        em = em + tmp_em
        f1 = f1 + tmp_f1
    start, end = end, len(eval_c_w)
    tmp_f1, tmp_em = evaluate_step(eval_c_w[eval_start:eval_end], eval_q_w[eval_start:eval_end], eval_c_ch[eval_start:eval_end], eval_q_ch[eval_start:eval_end], eval_start)
    em = em + tmp_em
    f1 = f1 + tmp_f1
    return em, f1

def evaluate_step(eval_c_w, eval_q_w, eval_c_ch, eval_q_ch, start):
    with open('val_context.pkl', 'rb') as handle:
        answer_context = pickle.load(handle)
    with open('val_answers.pkl', 'rb') as handle:
        val_answer = pickle.load(handle)

    enc_c, state_c1, state_c2, enc_q, state_q1, state_q2 = encoder_lyr(eval_c_w, eval_q_w, eval_c_ch, eval_q_ch)
    shape = np.shape(enc_c)
    #hidden = tf.math.add(state_c1, state_c2)
    #hidden = tf.math.divide(hidden,2)
    hidden = tf.concat([state_c1, state_c2], axis=-1)
    for i in range(shape[1]):
        output, hidden, _ = (G_decode(enc_q, enc_c[:, i, :], hidden))
        if i != 0:
            decode_o = tf.concat([decode_o, output], axis = 1)
        else:
            decode_o = output

    #for i in range(shape[1]):
    SA_o, _ = S_attn(decode_o, decode_o[:, i, :])
        #if i != 0:
        #    SA_o = tf.concat([SA_o, attn_vec], axis=1)
        #else:
        #    SA_o = attn_vec

    SA_output = S_decode(SA_o)
    #state_q = tf.math.add(state_q1, state_q2)
    #state_q = tf.math.divide(state_q,2)
    state_q = tf.concat([state_q1, state_q2], axis=-1)
    
    O_hidden, _ = O_initial(enc_q, state_q)
    
    for i in range(2):
        output, O_hidden, a_weights = O_decode(SA_output, O_hidden)
        #O_hidden = tf.concat([O_hidden1, O_hidden2], axis=-1)
        a_weights = tf.squeeze(a_weights, axis = 2)
        #y = tf.math.argmax(a_weights, axis = 1)
        #print(np.shape(y))
        if i == 0:
            pred_y1 = a_weights
        else:
            pred_y2 = a_weights
    
    pred_y1 = tf.argmax(pred_y1, axis = 1)
    pred_y2 = tf.argmax(pred_y2, axis = 1)


    em, f1 = 0,0
    for i in range(len(pred_y1)):
        prediction = answer_context[start+i][pred_y1:pred_y2+1]
        true_answer = val_answer[start+i]
        em = em + (exact_match(true_answer, prediction) * 100)
        f1 = f1 + (f1_score(true_answer, prediction) * 100)
    f1 = f1/len(pred_y1)
    em = em/len(pred_y1)
    return f1, em

import os
def save_wgts(i,j):
    path = 'epoch'+str(1)+'itr'+str(j)
    os.mkdir(path)
    path = path + '/'
    
    encoder_lyr.save_weights(path)
    
    G_decode.save_weights(path)
   
    S_attn.save_weights(path)
    
    S_decode.save_weights(path)
    
    O_initial.save_weights(path)
    
    O_decode.save_weights(path)
    

def loss_function(y1, y2, pred1, pred2):
    loss_sum = []
    for i in range(len(y1)):
        #print(pred1[i,y1[i]], pred2[i,y2[i]])
        x1, x2 = pred1[i,y1[i]], pred2[i,y2[i]]
        if x1 == 0:
            print('predict zero {}'.format(i))
            x1 = 0.000001
        if x2 == 0:
            print('predict zero {}'.format(i))
            x2 = 0.000001
        loss_sum.append(tf.math.log(x1) + tf.math.log(x2))
    loss = (-sum(loss_sum)) /len(y1)
    return loss


#@tf.function
def train(query_w_token,context_w_token,context_ch,query_ch, start, end):

    with tf.GradientTape() as tape:
        
        enc_c, state_c1, state_c2, enc_q, state_q1, state_q2, c_mask, q_mask = encoder_lyr(context_w_token, query_w_token, context_ch, query_ch)
        shape = np.shape(enc_c)
        #hidden = tf.math.add(state_c1, state_c2)
        #hidden = tf.math.divide(hidden,2)
        state_q = tf.concat([state_q1, state_q2], axis=-1)
        q_mask = tf.cast(enc_q._keras_mask, tf.float32)
        c_mask = tf.cast(enc_c._keras_mask, tf.float32)
        O_hidden, _ = O_initial(enc_q, state_q, q_mask)
        state_q1, state_q2, state_q = [], [], []

        hidden = tf.concat([state_c1, state_c2], axis=-1)
        for i in range(shape[1]):
            output, hidden, _ = (G_decode(enc_q, enc_c[:, i, :], hidden, q_mask))
            if i != 0:
                decode_o = tf.concat([decode_o, output], axis = 1)
            else:
                decode_o = output
        enc_c, output, hidden, state_c1, state_c2, enc_q = [], [], [], [], [], []

        SA_o, _ = S_attn(decode_o, decode_o, c_mask)
            

        decode_o, attn_vec = [], []
        SA_output = S_decode(SA_o)
        SA_o = []
        #state_q = tf.math.add(state_q1, state_q2)
        #state_q = tf.math.divide(state_q,2)
        
        for i in range(2):
            output, O_hidden, a_weights = O_decode(SA_output, O_hidden, c_mask)
            #O_hidden = tf.concat([O_hidden1, O_hidden2], axis=-1)
            a_weights = tf.squeeze(a_weights, axis = 2)
            #y = tf.math.argmax(a_weights, axis = 1)
            #print(np.shape(y))
            if i == 0:
                pred_y1 = a_weights
            else:
                pred_y2 = a_weights
        output, O_hidden = [], []
        loss = loss_function(start, end, pred_y1, pred_y2)

    variables = encoder_lyr.trainable_variables + G_decode.trainable_variables + S_attn.trainable_variables + O_initial.trainable_variables + O_decode.trainable_variables + S_decode.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return loss

def preprocess_answer(x):
    exclude = set(string.punctuation)
    answer = [word.lower() for word in x if word not in exclude]
    return answer

def exact_match(true_answer, prediction):
    return preprocess_answer(true_answer) == preprocess_answer(prediction)

def f1_score(true_answer, prediction):
    true_answer, prediction = preprocess_answer(true_answer), preprocess_answer(prediction)
    common_answer = Counter(true_answer) & Counter(prediction)
    if common_answer == 0:
        return 0
    precision = common_answer / len(prediction)
    recall = common_answer / len(true_answer)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

with open('context_ch_1.pkl', 'rb') as handle:
    print('yes')
    context_ch = pickle.load(handle)
with open('query_ch_1.pkl', 'rb') as handle:
    query_ch = pickle.load(handle)
with open('context_w_token.pkl', 'rb') as handle:
    context_w_token = pickle.load(handle)
with open('query_w_token.pkl', 'rb') as handle:
    query_w_token = pickle.load(handle)
with open('y1.pkl', 'rb') as handle:
    y1 = pickle.load(handle)
with open('y2.pkl', 'rb') as handle:
    y2  = pickle.load(handle)
with open('embedding.pkl', 'rb') as handle:
    embedding_matrix  = pickle.load(handle)
with open('word_tokenizer.pkl', 'rb') as handle:
    word_tokenizer  = pickle.load(handle)

context_w_token = context_w_token[:342*64,:]
query_w_token = query_w_token[:342*64,:]
y1 = y1[:342*64]
y2 = y2[:342*64]

vocab_size = len(word_tokenizer)
word_tokenizer = {}

encoder_lyr = encoder(embedding_matrix, 30, vocab_size)
G_decode = GatedAttentionDecoder(30)
S_attn = SelfAttention(30)
S_decode = SAdecode(30)
O_decode = OutputDecoder(30)
O_initial = InitialStateoutput(30)
optimizer = tf.keras.optimizers.Adadelta(learning_rate = 1, rho = 0.95,epsilon=1e-06)

checkpoint_dir = 'checkpoints/./training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder_lyr=encoder_lyr,
                                 S_decode=S_decode,
                                 S_attn=S_attn,
                                 G_decode=G_decode,
                                 O_decode = O_decode,
                                 O_initial=O_initial)
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
status.assert_consumed()




epochs = 20
batch_size = 48
n_samples = len(query_w_token)
context_ch, query_ch = tf.cast(context_ch, tf.float32), tf.cast(query_ch, tf.float32)
context_w_token, query_w_token = tf.cast(context_w_token, tf.float32), tf.cast(query_w_token, tf.float32)

slice_val, slice_idx = 165*64, 0
for i in range(epochs):
    loss_val = []
    itr = int(np.floor(n_samples / batch_size))
    for j in range(itr):
        start = j*batch_size 
        end = start+batch_size
        cont_start = (j*batch_size) - slice_idx  
        cont_end = cont_start + batch_size
        loss = train(query_w_token[start:end],context_w_token[start:end],context_ch[cont_start:cont_end],query_ch[cont_start:cont_end], y1[start:end], y2[start:end])
        print('iteration no {} loss {}'.format(j,tf.keras.backend.get_value(loss)))
        loss_val.append(tf.keras.backend.get_value(loss))
        if end % slice_val == 0:
            file_name = 'context_ch_' + str(int(end / slice_val) + 1) +'.pkl'
            with open(file_name, 'rb') as handle:
                context_ch = pickle.load(handle)
            context_ch = tf.cast(context_ch, tf.float32)
            file_name = 'query_ch_' + str(int(end / slice_val) + 1) +'.pkl'
            with open(file_name, 'rb') as handle:
                query_ch = pickle.load(handle)
            query_ch = tf.cast(query_ch, tf.float32)
            slice_idx = slice_val*int(end / slice_val)
        if j == int(itr/2) or j == itr - 1:
            checkpoint.save(file_prefix = checkpoint_prefix)
            #print('weights saved')
            #save_wgts(i,j)
    #em, f1 = evalution()
    with open('output.txt', 'a') as file:
        #file.write('epoch {} em {} f1 {}\n'.format(i, em, f1))
        for k in loss_val:
            file.write(str(k))
            file.write('\n')
    loss_val = []
    #if em > 71 or f1 > 79.5:
    #    save_wgts(50,0)
    #    break

save_wgts(50,1)