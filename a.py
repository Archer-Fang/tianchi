import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import  json
import pickle

dataset = []
train_file = 'a.txt'
for i in range(1000):
	txt_path = f'tmp/train/{i}.txt'
	ann_path = f'tmp/train/{i}.ann'
	# open('train/')
	ann = pd.read_table(ann_path, header=None)
	txt = open(txt_path, 'r',encoding='UTF-8')
	text = txt.read()
	data = {}
	data["id"] = i
	data["text"] = str(text)
	label = []
	for j in range(ann.shape[0]):
		tmp = ann.iloc[j, 1]
		tmps = tmp.split(' ')
		label.append([int(tmps[1]), int(tmps[2]), tmps[0]])
	data["labels"] = label

	dataset.append(data)
with open('a.pkl','wb') as f :
	pickle.dump(dataset,f)



# a = []
# b = []
# c = []
# d = []
# for i in range(200):
# 	i += 1
# 	a.append(i)
# 	c.append((i/200)**3)
# 	d.append((i/200)**7)
# 	if i < 160:
# 		t = 1-(i/200)**3
# 	else:
# 		t = 1-(160/200)**3
# 	b.append(t)
# # plt.plot(a, b)
# plt.plot(a, c)
# plt.plot(a, b)
# plt.plot(a, d)
# plt.legend(['γ (τ=3)','β (τ=3)','γ (τ=7)'])
# plt.show()
# a = np.array([5000,2997,1796,1077,645,387,232,139,83,50])
# print(a/5000)
# e = []
# f = []
# for i in range(400):
# 	i += 1
# 	e.append(i)
# 	f.append((i/400)**16)
# plt.plot(e,f)
# plt.show()