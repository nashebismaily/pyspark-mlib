from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.stat import Statistics

from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("Actopms").setMaster("yarn")
sc = SparkContext(conf=conf)

#sentence = "hello hello world"
#words = sentence.split() # Split sentence into a list of terms
#tf = HashingTF(10000) # Create vectors of size S = 10,000
#tf.transform(words)
#SparseVector(10000, {3065: 1.0, 6861: 2.0})


rdd = sc.wholeTextFiles("dracula.txt").map(lambda (name, text): text.split())
tf = HashingTF(10000)
tfVectors = tf.transform(rdd).cache()


# Compute the IDF, then the TF-IDF vectors
idf = IDF(10000)
idfModel = idf.fit(tfVectors)

tfIdfVectors = idfModel.transform(tfVectors)


Statistics.colStats(tfIdfVectors)

Statistics.corr(tfIdfVectors)
Statistics.corr(tfIdfVectors,"spearman")

Statistics.corr(tfVectors,tfIdfVectors)

Statistics.chiSqTest(tfIdfVectors)

