import json  
  
from keras.layers import Dense  
from keras.models import Model  
from tqdm import tqdm  
import pickle

from bert4keras.backend import keras, K  
from bert4keras.layers import ConditionalRandomField  
from bert4keras.models import build_transformer_model  
from bert4keras.optimizers import Adam  
from bert4keras.snippets import ViterbiDecoder, to_array  
from bert4keras.snippets import sequence_padding, DataGenerator  
from bert4keras.tokenizers import Tokenizer  
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':1})))

maxlen = 3037  
epochs = 20
batch_size = 40
bert_layers = 6  
learing_rate = 1e-5 # bert_layers越小，学习率应该要越大  
crf_lr_multiplier = 1000 # 必要时扩大CRF层的学习率  
  
# bert配置  
config_path = 'chinese_L-12_H-768_A-12/bert_config.json'  
# checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'  
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'  


labels = []  
def load_data(filename):  
    D = []  
    with open('a.pkl','rb') as f :
        
        l_medical=pickle.load(f)
    max_n=0
    for medical in l_medical:  
        d = []  
        medical_text = medical["text"]  
        if len(medical_text)>max_n:
            max_n=len(medical_text)
        medical_labels = medical["labels"]  
        laster_label = 0  
        for medical_label in medical_labels:  
            begin_label = medical_label[0]  

            # d.append([medical_text[laster_label:begin_label], "O"])  
            last_label = medical_label[1]  
            d.append([medical_text[begin_label:last_label], medical_label[2]])  
            laster_label = last_label  
            if medical_label[2] not in labels:  
                labels.append(medical_label[2])  
        D.append(d)  
    print(max_n)
    return D  
  
  
# 标注数据  
train_data = load_data('train.pkl')  
  
# 建立分词器  
tokenizer = Tokenizer(dict_path, do_lower_case=True)  
  
# 类别映射  
id2label = dict(enumerate(labels))  
label2id = {j: i for i, j in id2label.items()}  
num_labels = len(labels) * 2 + 1  
  
  
class data_generator(DataGenerator):  
    """数据生成器  
 """  
    def __iter__(self, random=False):  
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []  
        for is_end, item in self.sample(random):  
            token_ids, labels = [tokenizer._token_start_id], [0]  
            for w, l in item:  
                w_token_ids = tokenizer.encode(w)[0][1:-1]  
                if len(token_ids) + len(w_token_ids) < maxlen:  
                    token_ids += w_token_ids  
                    if l == 'O':  
                        labels += [0] * len(w_token_ids)  
                    else:  
                        B = label2id[l] * 2 + 1  
                        I = label2id[l] * 2 + 2  
                        labels += ([B] + [I] * (len(w_token_ids) - 1))  
                else:  
                    break  
            token_ids += [tokenizer._token_end_id]  
            labels += [0]  
            segment_ids = [0] * len(token_ids)  
            batch_token_ids.append(token_ids)  
            batch_segment_ids.append(segment_ids)  
            batch_labels.append(labels)  
            if len(batch_token_ids) == self.batch_size or is_end:  
                batch_token_ids = sequence_padding(batch_token_ids) 
                batch_segment_ids = sequence_padding(batch_segment_ids)  
                batch_labels = sequence_padding(batch_labels)  
                yield [batch_token_ids, batch_segment_ids], batch_labels  
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []  
  
  
model = build_transformer_model(  
    config_path,  
    None
#   checkpoint_path,  
)  

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)  
output = model.get_layer(output_layer).output  
output = Dense(num_labels)(output)  
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)  
output = CRF(output)  
  
model = Model(model.input, output)  
model.summary()  
  
model.compile(  
    loss=CRF.sparse_loss,  
  optimizer=Adam(learing_rate),  
  metrics=[CRF.sparse_accuracy]  
)  
  
  
class NamedEntityRecognizer(ViterbiDecoder):  
    """命名实体识别器  
 """  
    def recognize(self, text):  
        tokens = tokenizer.tokenize(text)  
        # while len(tokens) > 512:  
        #     tokens.pop(-2)  
        mapping = tokenizer.rematch(text, tokens)  
        token_ids = tokenizer.tokens_to_ids(tokens)  
        segment_ids = [0] * len(token_ids)  
        token_ids, segment_ids = to_array([token_ids], [segment_ids])  
        nodes = model.predict([token_ids, segment_ids])[0]  
        labels = self.decode(nodes)  
        entities, starting = [], False  
        for i, label in enumerate(labels):  
            if label > 0:  
                if label % 2 == 1:  
                    starting = True  
                    entities.append([[i], id2label[(label - 1) // 2]])  
                else:
                    if starting:  
                        entities[-1][0].append(i)  
                # else:  
                #     starting = False  
            else:  
                starting = False  
        ner_answer = []  
        for w, l in entities:  
            ner_answer.append([mapping[w[0]][0],mapping[w[-1]][-1] + 1, l])  
        return ner_answer  

NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])  
  
  
def evaluate(data):  
    """评测函数  
"""  
    X, Y, Z = 1e-10, 1e-10, 1e-10  
    for d in tqdm(data):  
        text = ''.join([i[0] for i in d])  
        R = set([tuple([text[_[0]:_[1]+1],_[2]]) for _ in NER.recognize(text)])  
        T = set([tuple(i) for i in d if i[1] != 'O'])  
        X += len(R & T)  
        Y += len(R)  
        Z += len(T)  
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z  
    return f1, precision, recall  
  
  
class Evaluator(keras.callbacks.Callback):  
    def __init__(self):  
        self.best_val_f1 = 0  
  
    def on_epoch_end(self, epoch, logs=None):  
        if epoch % 10 == 0:  
            f1, precision, recall = evaluate(train_data[0:len(train_data) // 2])  
            # 保存最优  
            if f1 >= self.best_val_f1:  
                self.best_val_f1 = f1  
                model.save_weights('./20200730.weights')  
                print(  
                    'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %  
                    (f1, precision, recall, self.best_val_f1)  
                )  
  
  
if __name__ == '__main__':  
    evaluator = Evaluator()  
    train_generator = data_generator(train_data, batch_size)  
  
    model.fit_generator(  
        train_generator.forfit(),  
  steps_per_epoch=len(train_generator),  
  epochs=epochs,  
  callbacks=[evaluator]  
    )