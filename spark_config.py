from pyspark import SparkContext, SparkConf

# Set up Spark
conf = SparkConf().setAppName("PBAFFS").setMaster("local[*]")
sc = SparkContext.getOrCreate(conf = conf)