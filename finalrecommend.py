import pandas as pd
from sklearn.externals import joblib
import numpy as np

def randomsong(newsong):
	listsongs=[]
	for i in range(10):
		query_index = np.random.choice(newsong.shape[0])
		listsongs.append(newsong.index[query_index])
	return listsongs	

def recommend(newsong,loaded_model,msg):
	listsongs = []
	distances, indices = loaded_model.kneighbors(newsong.loc[msg, :].values.reshape(1, -1), n_neighbors = 10)
	for i in range(10):
		listsongs.append(newsong.index[indices.flatten()[i]])
	return listsongs 