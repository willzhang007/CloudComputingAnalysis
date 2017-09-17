__author__ = 'lichaozhang'





########################################################################
from pyspark import SparkContext

sc = SparkContext()



small_ratings_raw_data = sc.textFile("datasets/ml-latest-small/ratings.csv")
#small_ratings_raw_data = sc.textFile("Users/lichaozhang/Desktop/datasets/ml-latest-small/ratings.csv")
small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]

small_ratings_data = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()
small_ratings_data.take(3)

#########################################################################



small_movies_raw_data = sc.textFile("datasets/ml-latest-small/movies.csv")
small_movies_raw_data_header = small_movies_raw_data.take(1)[0]

small_movies_data = small_movies_raw_data.filter(lambda line: line!=small_movies_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1])).cache()

print small_movies_data.take(3)

#########################################################################
#In order to determine the best ALS parameters,
##  I use the small dataset.
##  first to split it into train, validation, and test datasets.



training_RDD, validation_RDD, test_RDD = small_ratings_data.randomSplit([6, 2, 2], seed=0L)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
#########################################################################
#Selecting ALS parameters using the small dataset

from pyspark.mllib.recommendation import ALS
import math

seed = 5L
iterations = 10
regularization_parameter = 0.2
ranks = [4, 8, 12, 16, 20, 50]
errors = [0, 0, 0, 0, 0, 0]
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < min_error:
        min_error = error
        best_rank = rank

print 'The best model was trained with rank %s' % best_rank

###################################################################
#print some of the prediction and some explain

print predictions.take(3)
#Basically I have the UserID, the MovieID, and the Rating, as I have in our ratings dataset.
# In this case the predictions third element, the rating for that movie and user, is the predicted by our ALS model.
#Then we join these with our validation data (the one that includes ratings) and the result looks as follows:
print rates_and_preds.take(3)
#To that, I apply a squared difference and the I use the mean() action to get the MSE and apply sqrt






###################################################################
#test the selected model using test data

model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())

print 'For testing data the RMSE is %s' % (error)


