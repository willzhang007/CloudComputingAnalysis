__author__ = 'lichaozhang'


import os
import sys

# Path for spark source folder
os.environ['SPARK_HOME']="/Users/lichaozhang/spark-1.4.0"

# Append pyspark  to Python Path
sys.path.append("/Users/lichaozhang/spark-1.4.0/python/")
sys.path.append("/Users/lichaozhang/spark-1.4.0/python/lib/py4j-0.8.2.1-src.zip")

try:
    from pyspark import SparkContext
    from pyspark import SparkConf

    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

sc = SparkContext('local')
words = sc.parallelize(["scala","java","hadoop","spark","akka"])
print words.count()