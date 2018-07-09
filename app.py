import sumTwoNumbers
import finalrecommend
from flask import Flask, request
from flask_restful import Resource, Api
import pandas as pd
from sklearn.externals import joblib

loaded_model = joblib.load('recommender_model.sav')
song_df = pd.read_csv("song_df.csv")
df = song_df.drop_duplicates(['song','user_id'])
newsong = df.pivot('song','user_id','listen_time')
newsong = newsong.fillna(0)

app = Flask(__name__)
api = Api(app)
 

class recommend(Resource):
    def get(self, msg):
        return {'song1': finalrecommend.recommend(newsong,loaded_model,msg)[0],'song2': finalrecommend.recommend(newsong,loaded_model,msg)[1],'song3': finalrecommend.recommend(newsong,loaded_model,msg)[2],
        'song4': finalrecommend.recommend(newsong,loaded_model,msg)[3],'song5': finalrecommend.recommend(newsong,loaded_model,msg)[4],'song6':finalrecommend.recommend(newsong,loaded_model,msg)[5],
        'song7': finalrecommend.recommend(newsong,loaded_model,msg)[6],'song8': finalrecommend.recommend(newsong,loaded_model,msg)[7],'song9': finalrecommend.recommend(newsong,loaded_model,msg)[8],
        'song10': finalrecommend.recommend(newsong,loaded_model,msg)[9]}

class randomsong(Resource):
    def get(self):
        return {'song1': finalrecommend.randomsong(newsong)[0],'song2': finalrecommend.randomsong(newsong)[1],'song3': finalrecommend.randomsong(newsong)[2],
        'song4': finalrecommend.randomsong(newsong)[3],'song5': finalrecommend.randomsong(newsong)[4],'song6':finalrecommend.randomsong(newsong)[5],
        'song7': finalrecommend.randomsong(newsong)[6],'song8': finalrecommend.randomsong(newsong)[7],'song9': finalrecommend.randomsong(newsong)[8],
        'song10': finalrecommend.randomsong(newsong)[9]}        



api.add_resource(recommend,'/recommend/<msg>')
api.add_resource(randomsong,'/randomsong/')
if __name__ == '__main__':
     app.run()