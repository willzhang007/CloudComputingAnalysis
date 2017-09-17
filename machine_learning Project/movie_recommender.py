__author__ = 'lichaozhang'

from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
import math
import os

sc = SparkContext()

###################################################################
#Using the complete dataset to build the final model

# Load the complete dataset file
#set optimal parameters

best_rank = 8
seed = 5L
iterations = 10
regularization_parameter = 0.2
###################################################################
#use the model that we built with ml-latest-small

small_ratings_raw_data = sc.textFile("datasets/ml-latest-small/ratings.csv")
small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]

small_ratings_data = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()

training_RDD, validation_RDD, test_RDD = small_ratings_data.randomSplit([6, 2, 2], seed=0L)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))

model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)

###################################################################

#complete_ratings_raw_data = sc.textFile("datasets/ml-latest/ratings.csv")
complete_ratings_raw_data = sc.textFile("datasets/ml-1m/ratings.csv")
#complete_ratings_raw_data = sc.textFile("datasets/ml-10M100K/ratings.csv")


complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]

# Parse
#complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line!=complete_ratings_raw_data_header)\
   #.map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line!=complete_ratings_raw_data_header)\
    .map(lambda line: line.split("::")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()

print "There are %s recommendations in the complete dataset" % (complete_ratings_data.count())

###################################################################
#train the model with the latest data


training_RDD, test_RDD = complete_ratings_data.randomSplit([7, 3], seed=0L)

complete_model = ALS.train(training_RDD, best_rank, seed=seed,
                           iterations=iterations, lambda_=regularization_parameter)

###################################################################
# test on  testing set
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

predictions = complete_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())

print 'For testing data the RMSE is %s' % (error)


###################################################################
#load the full movies.csv

#complete_movies_raw_data = sc.textFile("datasets/ml-latest/movies.csv")
complete_movies_raw_data = sc.textFile("datasets/ml-1m/movies.csv")
#complete_movies_raw_data = sc.textFile("datasets/ml-1OM100K/movies.csv")



complete_movies_raw_data_header = complete_movies_raw_data.take(1)[0]

# Parse
#complete_movies_data = complete_movies_raw_data.filter(lambda line: line!=complete_movies_raw_data_header)\
#    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()
complete_movies_data = complete_movies_raw_data.filter(lambda line: line!=complete_movies_raw_data_header)\
    .map(lambda line: line.split("::")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()

complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]),x[1]))

print "There are %s movies in the complete dataset" % (complete_movies_titles.count())



###################################################################
#count the number of ratings per movie so that we can give recommendations of movies
# with a certain minimum number of ratings

def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)

movie_ID_with_ratings_RDD = (complete_ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))

###################################################################
# rate some movies for the new user.
# We will put them in a new RDD and we will use the user ID 0,
# that is not assigned in the MovieLens dataset.

new_user_ID = 0

# The format of each line is (userID, movieID, rating)
new_user_ratings = [
     (0,260,9), # Star Wars (1977)
     (0,1,8), # Toy Story (1995)
     (0,16,7), # Casino (1995)
     (0,25,8), # Leaving Las Vegas (1995)
     (0,32,9), # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
     (0,335,4), # Flintstones, The (1994)
     (0,379,3), # Timecop (1994)
     (0,296,7), # Pulp Fiction (1994)
     (0,858,10) , # Godfather, The (1972)
     (0,50,8) # Usual Suspects, The (1995)
    ]
new_user_ratings_RDD = sc.parallelize(new_user_ratings)
print 'New user ratings: %s' % new_user_ratings_RDD.take(10)

#add them to the data that will be used to train recommender model

complete_data_with_new_ratings_RDD = complete_ratings_data.union(new_user_ratings_RDD)

#train the ALS model using all the optimal parameters

from time import time

t0 = time()

new_ratings_model = ALS.train(complete_data_with_new_ratings_RDD, best_rank, seed=seed,
                              iterations=iterations, lambda_=regularization_parameter)
tt = time() - t0

print "New model trained in %s seconds" % round(tt,3)


###################################################################
#Getting top recommendations so that we can make some recommendation to users
#get an RDD with all the movies the new user hasn't rated yet

new_user_ratings_ids = map(lambda x: x[1], new_user_ratings) # get just movie IDs
# keep just those not on the ID list
new_user_unrated_movies_RDD = (complete_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0])))

# Use the input RDD, new_user_unrated_movies_RDD, with new_ratings_model.predictAll() to predict new ratings for the movies
new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)

# Transform new_user_recommendations_RDD into pairs of the form (Movie ID, Predicted Rating)
new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
new_user_recommendations_rating_title_and_count_RDD = \
    new_user_recommendations_rating_RDD.join(complete_movies_titles).join(movie_rating_counts_RDD)
new_user_recommendations_rating_title_and_count_RDD.take(3)

#flat this down a bit in order to have (Title, Rating, Ratings Count).
new_user_recommendations_rating_title_and_count_RDD = \
    new_user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))

#get the highest rated recommendations for the new user, filtering out movies with less than 25 ratings.

top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=25).takeOrdered(25, key=lambda x: -x[1])

print ('TOP recommended movies (with more than 25 reviews):\n%s' %
        '\n'.join(map(str, top_movies)))


###################################################################
#Getting individual ratings

#get the predicted rating for a particular movie for a given user


my_movie = sc.parallelize([(0, 500)]) # Quiz Show (1994)
individual_movie_rating_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)
individual_movie_rating_RDD.take(1)


from pyspark.mllib.recommendation import MatrixFactorizationModel



# Save and load model

model_path = os.path.join('..','models', 'movie_lens_als_saved')
model.save(sc,model_path )
saved_model = MatrixFactorizationModel.load(sc, model_path)`
