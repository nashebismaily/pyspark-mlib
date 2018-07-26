from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("Liner Regression").setMaster("yarn")
sc = SparkContext(conf=conf)

data = [
             LabeledPoint(0, Vectors.dense([1, 0, 0])),
             LabeledPoint(0, Vectors.dense([2, 0, 0])),
             LabeledPoint(1, Vectors.dense([0, 1, 0])),
             LabeledPoint(1, Vectors.dense([0, 2, 0])),
             LabeledPoint(2, Vectors.dense([0, 0, 1])),
             LabeledPoint(2, Vectors.dense([0, 0, 2]))
]

# $example on$
data = sc.parallelize(data)

 # Split data aproximately into training (60%) and test (40%)
training, test = data.randomSplit([0.6, 0.4], seed=0)
training.cache()

model = NaiveBayes.train(training, 1.0)

predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / test.count()

print("Test Data:")
print(test.collect())

print("Prediction and Label")
print(predictionAndLabel.collect())

print ("Predicted Accuracy:  %g" %  accuracy)

# Save and load model
# model.save(sc, "target/tmp/myNaiveBayesModel")
# sameModel = NaiveBayesModel.load(sc, "target/tmp/myNaiveBayesModel")
#  $example off$
