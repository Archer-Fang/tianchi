#! -*- coding: utf-8 -*-  
import json
from keras.layers import Dense  
from keras.models import Model  
from tqdm import tqdm  
  
from bert4keras.backend import K  
from bert4keras.layers import ConditionalRandomField  
from bert4keras.models import build_transformer_model  
from bert4keras.optimizers import Adam  
from bert4keras.snippets import ViterbiDecoder, to_array  
from bert4keras.snippets import sequence_padding, DataGenerator  
from bert4keras.tokenizers import Tokenizer  
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle

maxlen = 9999  
epochs = 10  
batch_size = 32  
bert_layers = 6
learing_rate = 1e-5 # bert_layers越小，学习率应该要越大  
crf_lr_multiplier = 1000 # 必要时扩大CRF层的学习率  
  
# bert配置  
config_path = 'chinese_L-12_H-768_A-12/bert_config.json'  
# checkpoint_path = '20200730.weights'  
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'  
  
  
def load_data(filename):  
    with open('test.pkl','rb') as f :
        l_medical=pickle.load(f)
    
    return l_medical  
labels = []  
def load_data2(filename):  
    D = []  
    with open('a.pkl','rb') as f :
        
        l_medical=pickle.load(f)
    for medical in l_medical:  
        d = []  
        medical_text = medical["text"]  
        medical_labels = medical["labels"]  
        laster_label = 0  
        for medical_label in medical_labels:  
            begin_label = medical_label[0]  

            d.append([medical_text[laster_label:begin_label], "O"])  
            last_label = medical_label[1]  
            d.append([medical_text[begin_label:last_label], medical_label[2]])  
            laster_label = last_label  
            if medical_label[2] not in labels:  
                labels.append(medical_label[2])  
        D.append(d)  
    return D  
  
test_data = load_data('test.pkl')  
  
# 标注数据  
train_data = load_data2('train.pkl')  
del train_data
# 标注数据  
  
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
    # checkpoint_path
#   None  
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
  
if __name__ == '__main__':  
    model.load_weights(r'20200730.weights')  
    # medical_dicts_drop_duplicates = open("..\medical_data\chinese_medical_drop.csv", "r",encoding="utf-8")  
    export = []  
    import json  
    L=[]
    for item in test_data:
        num=0
        text=item['text']
        R = NER.recognize(text)  
        RR=[]
        for _ in R:
            num=num+1
            id='T'+str(num)
            tmp=[id,]
            part_seq=text[_[0]:_[1]]
            tmp=tmp+_+[part_seq]
            RR.append(tmp)

        print(RR)
        L.append(RR)
        break
    for index,item in enumerate(L):
        with open(f'result/{index+1000}.ann','w') as f:
            for _ in item:
                f.write(f"{_[0]}	{_[3]} {_[1]} {_[2]}	{_[4]}\n")
    # ids = 999  
    # with open("medical_ner_auto_predict_export.txt", "w") as write_file:  
    #     # for i in tqdm(medical_dicts_drop_duplicates):  
    #     export_dict = {}  
    #     ids += 1  
    #     # i_text = json.loads(i)  
    #     export_dict["id"] = ids  
    #     export_dict["text"] = "（１）食物多样，谷类为主。（２）多吃蔬菜、水果和薯类。（３）每天吃奶类、豆类或豆制品。（４）经常吃适量的鱼、禽、蛋、瘦肉，少吃肥肉或荤油。" 
    #     R = NER.recognize(export_dict["text"]) 
    #     # text="i have apple"
    #     # R = NER.recognize(test)  

    #     export_dict["labels"] = R  
    #     export_dict = json.dumps(export_dict, ensure_ascii=False)  
    #     write_file.write(export_dict)
    
