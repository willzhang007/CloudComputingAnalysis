
#################################################
This is for CS589 Project

Editor Lichao Zhang
##################################################

To rerun my code:
1.Run dataset_small.py. This is for model building and parameter selection using “ml-lateset-small” dataset.
2.Run movie_recommonder.py. This is for final model using optimal parameters and make recommenders for specific users.The dataset this model use is ml-1m.After you run movie_recommonder.py. The model would be saved in “model” folder. 
3.To see my figures in report ,run plot_figures.py
4.To make my code reproducible,I am using ml-1m datasets.In fact,it can be used for ml-latest,but sometimes it receive “Error out of memory”.To solve this problem,see “notation 5”

##################################################################################
#some notation

1.I use the spark-1.4.0 to finish this project.Somehow spark-1.6.1 can’t be install on my computer.But I think it would be ok.

2.Because the bug of spark-1.4.0,the version of bumpy has to be 1.4-1.9.I used numpy-version-1.6.2.

3. There may be a lot of INFO logging in console.
You can just go to your spark director : ~/spark/conf/log4j.properties
Open it in any text editor and change the first line “log4j.rootCategory=INFO, console” to “log4j.rootCategory=WARN, console”.Then save and restart your shell.That should work.

4.I tried to train the latest full and other movieLens datasets.And they all worked out .Just remember when you are loading “ml-10M100K” and “ml-1m’,change the code from:

complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line!=complete_ratings_raw_data_header)\
   .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()

to :
complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line!=complete_ratings_raw_data_header)\
    .map(lambda line: line.split("::")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()

5.If you see “ERROR:out of memory”,see http://stackoverflow.com/questions/26562033/how-to-set-apache-spark-executor-memory .That happened when I run “ml-latest”data sets. 
