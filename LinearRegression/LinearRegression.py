# Probablity of getting a Loan (between 0 and 1)

# Person has 3 characterstics: Credit Score, Income, Age. These are the 3 features we will use.


from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import Vectors


conf = SparkConf().setAppName("Liner Regression").setMaster("yarn")
sc = SparkContext(conf=conf)

approved = [
             LabeledPoint(0.9, Vectors.dense([750/850, 150000/200000])),
             LabeledPoint(0.7, Vectors.dense([700/850, 100000/200000])),
             LabeledPoint(0.8, Vectors.dense([750/850, 120000/200000])),
             LabeledPoint(0.8, Vectors.dense([775/850, 130000/200000])),
             LabeledPoint(0.9, Vectors.dense([715/850, 140000/200000]))
            ]

approvedExamples = sc.parallelize(approved)

denied = [
            LabeledPoint(0, Vectors.dense([400/850, 70000/200000])),
            LabeledPoint(0, Vectors.dense([450/850, 50000/200000])),
            LabeledPoint(0, Vectors.dense([315/850, 30000/200000])),
            LabeledPoint(0, Vectors.dense([420/850, 65000/200000])),
            LabeledPoint(0, Vectors.dense([300/850, 60000/200000]))
         ]

deniedExamples = sc.parallelize(denied)


trainingData = approvedExamples.union(deniedExamples)
trainingData.cache()

model = LinearRegressionWithSGD.train(trainingData, iterations=200, intercept=True)
print ("weights: %s, intercept: %s" % (model.weights, model.intercept))


approvedTest = Vectors.dense([800/850, 190000/200000])
deniedTest = Vectors.dense([200/850, 30000/200000])


print ("Prediction for approved: %g" % model.predict(approvedTest))
print ("Prediction for denied: %g" % model.predict(deniedTest))
