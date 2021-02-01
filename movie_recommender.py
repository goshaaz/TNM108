import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Preprocessing data
movies=pd.read_csv("IMDb movies.csv")

def extract_first_six_actors(row):
	actors_list = row["actors"].split()
	if len(actors_list) > 6:
		actors = ",".join(actors_list[0:5])
	else:
		actors = row["actors"]	
	return actors

for index, row in movies.iterrows():
	if isinstance(row['year'], str):
		if movies.loc[index, 'year'] == 'TV Movie 2019':
			movies.at[index, 'year'] = 2019
		else:
			movies.at[index, 'year'] = int(movies.loc[index, 'year'])

movies_short = movies[movies["year"] >= 1990]

movies_short = movies_short[movies_short["avg_vote"] >= 7]

movies_short = movies_short[movies_short["votes"] >= 20000]

movies_short = movies_short.replace(np.nan, '', regex=True)

movies_short["actors"] = movies_short.apply(extract_first_six_actors,axis=1)	

movies_short.reset_index(inplace=True)

#Adding new feature to dataset (combination of relevant features for finding similarity between movies)
features=['director','genre','actors','production_company']

for feature in features:
	movies_short[feature] = movies_short[feature].fillna('')

def combine_features(row):
	return row['director'] + " " + row['genre'] + " " + row["actors"] + " " + row["production_company"] 

movies_short["combined_features"] = movies_short.apply(combine_features,axis=1)

#Creating count matrix using combined_features
cv = CountVectorizer()
count_matrix = cv.fit_transform(movies_short["combined_features"])

#Computing Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)
movie_user_likes = "The Avengers"

#Getting index for movie based on its title
index = movies_short[movies_short["original_title"] == movie_user_likes].index

#Creating a list of similar movies in descending order of similarity score. 
#Top 8 are extracted (index 0 is the selected movie and is therefore excluded)
similiar_movies = list(enumerate(cosine_sim[index.values[0]]))
similiar_movies_sorted = sorted(similiar_movies, key=lambda x:x[1],reverse=True)[1:9]

#Printing recommended movies and corresponding similarity score
print("\nRecommended movies based on " + movie_user_likes)
i=0
for movie in similiar_movies_sorted:
	print(movies_short.loc[movie[0], 'original_title'] + " " +str(round(movie[1],2)))
	i=i+1
	if i>7:
		break

#Adding popularity aspect to algorithm and printing the result. The earlier 8 movies are sorted by their imdb user score
sort_by_average_vote = sorted(similiar_movies_sorted, key=lambda x:movies_short["avg_vote"][x[0]],reverse=True)
print("\nRecommended movies based on " + movie_user_likes +" sorted by average rating")
i=0
for movie in sort_by_average_vote:
	movie_rating=movies_short[movies_short.index == movie[0]]["avg_vote"].values[0]
	print(movies_short.loc[movie[0], 'original_title'] + " " +str(movie_rating))
	i=i+1
	if i>7:
		break		