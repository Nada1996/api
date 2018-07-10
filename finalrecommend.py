import pandas as pd
from sklearn.externals import joblib
import numpy as np

happy=['Happy - Pharrell Williams','All About That Bass - Meghan Trainor','Cups -  Anna Kendrick',
'Cyndi Lauper - Girls Just Want To Have Fun','Oasis - Live Forever','Good Vibrations - The Beach Boys',
'The Twist - Chubby Checker','Bruno Mars - The Lazy Song','Joni Mitchell - Big Yellow Taxi','Ariana Grande - Break Free']
sad=['Hello - Adele','Someone Like You - Adele','Im With You - Avril Lagivne','Alive - Sia','Big Girls Cry - Sia',
'Fort Romeau - Say Something','Aly & Fila - Without You','Lana Del Rey - Is This Happiness',
'Demi Lovato - Stone Cold','My Happy Ending - Avril Lavigne']

def songbymood(text_clf,sad,happy,msg):
	#print('hey there')
	listsongs=[]
	
	predicted = text_clf.predict([msg])
	if predicted[0]=='JOY':
		listsongs.append('Smile and shine, Heres something that will make you more happy' )
		for i in range(5):
			query_index = np.random.choice(len(happy))
			listsongs.append(happy[query_index])
	if predicted[0]=='SADNESS':
		listsongs.append('Uhhh oh, I see some sad vibes, heres something you need for now' )
		for i in range(5):
			query_index = np.random.choice(len(sad))
			listsongs.append(sad[query_index])

	return listsongs
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