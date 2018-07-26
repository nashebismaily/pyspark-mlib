from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.linalg import Vectors

from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("Scaling").setMaster("yarn")
sc = SparkContext(conf=conf)

vectors = [Vectors.dense([-2.0, 5.0, 1.0]), Vectors.dense([2.0, 0.0, 1.0])]

dataset = sc.parallelize(vectors)
scaler = StandardScaler(withMean=True, withStd=True)
model = scaler.fit(dataset)
result = model.transform(dataset)

print (result.collect())
