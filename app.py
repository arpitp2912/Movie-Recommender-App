from flask import Flask, render_template, request
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movie_data = pd.read_csv('movies.csv')

movie_data = movie_data.drop_duplicates()

movie_data.movie_title = movie_data.movie_title.str.lower().str.strip()
movie_data['index'] = movie_data.index

movie_data.genres = movie_data['genres'].str.split('|').fillna('').apply(lambda x: ' '.join(x))
movie_data.plot_keywords = movie_data['plot_keywords'].str.split('|').fillna('').apply(lambda x: ' '.join(x))
movie_data = movie_data.fillna('')
features = ['director_name', 'actor_2_name', 'actor_1_name','genres','actor_3_name', 'language', 'country', 'content_rating', 
           'imdb_score', 'plot_keywords']

def combine_features(row):
    return row['director_name'] +" "+row['actor_2_name']+" "+row["actor_1_name"]+" "+row["genres"]+" "+row["actor_3_name"] +" "+row["language"]+" "+row['country']+" "+row['content_rating']+" "+str(row['imdb_score'])+" "+row['plot_keywords']


movie_data["combined_features"] = movie_data.apply(combine_features,axis=1)
cv = CountVectorizer()
count_matrix = cv.fit_transform(movie_data["combined_features"])

cosine_sim = cosine_similarity(count_matrix)

def get_title_from_index(df, index):
    return df[df.index == index]["movie_title"].values[0]


def get_index_from_title(df, title):
    return df[df.movie_title == title]["index"].values[0]


def get_plot_from_index(df, index):
    return df[df.index == index]["plot"].values[0]

def recommend(movie_user_likes):
    try:
        movie_user_likes = movie_user_likes.lower()
        movie_index = get_index_from_title(movie_data, movie_user_likes)
        similar_movies =  list(enumerate(cosine_sim[movie_index]))
        sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
        i=0
        print("Top 5 similar shows like "+movie_user_likes+" are:\n")
        recommended_movies = []

        for element in sorted_similar_movies:

            recommended_movies.append([get_title_from_index(movie_data, element[0]), get_plot_from_index(movie_data, element[0])])
            i=i+1
            if i>=5:
                break
        return recommended_movies
    except:
        # return('Movie not found on Netflix. Please retry!')
        return None

app = Flask(__name__)

@app.route('/')
def root():
    return render_template('home.html')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		input_movie = request.form.get('movie')
	movies = recommend(input_movie)
	if movies is not None:
		return render_template('output.html',movies=movies, input_movie = input_movie.title())
	else:
		return render_template('error.html')


if __name__ == '__main__':
    app.run()
