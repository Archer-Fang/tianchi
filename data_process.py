import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import  json
import pickle
def get_process_dataset(type_d):
	dataset = []
	train_file = 'a.txt'
	if type_d=='train':
		n=0
		for i in range(1000):
			txt_path = f'tmp/{type_d}/{i}.txt'
			ann_path = f'tmp/{type_d}/{i}.ann'
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
				if int(tmps[2])>n:
					n=int(tmps[2])
				label.append([int(tmps[1]), int(tmps[2]), tmps[0]])
			data["labels"] = label
			dataset.append(data)
	else:
		for i in range(1000,1500):
			txt_path = f'tmp/{type_d}/{i}.txt'
			# ann_path = f'tmp/{type_d}/{i}.ann'
			# open('train/')
			# ann = pd.read_table(ann_path, header=None)
			txt = open(txt_path, 'r',encoding='UTF-8')
			text = txt.read()
			data = {}
			data["id"] = i
			data["text"] = str(text)
			label = []
			# for j in range(ann.shape[0]):
			# 	tmp = ann.iloc[j, 1]
			# 	tmps = tmp.split(' ')
			# 	label.append([int(tmps[1]), int(tmps[2]), tmps[0]])
			# data["labels"] = label
			dataset.append(data)
	with open(f'{type_d}.pkl','wb') as f :
		if type_d=='train':
			print(n)
		pickle.dump(dataset,f)


if __name__ == "__main__":
	# type_d='train'
	type_d='test'

	get_process_dataset(type_d)