from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("Spam Emails").setMaster("yarn")
sc = SparkContext(conf=conf)

spam = sc.textFile("spam.txt")
normal = sc.textFile("normal.txt")

# Create a HashingTF instance to map email text to vectors of 10,000 features.
tf = HashingTF(numFeatures = 10000)

# Each email is split into words, and each word is mapped to one feature.
spamFeatures = spam.map(lambda email: tf.transform(email.split(" ")))
normalFeatures = normal.map(lambda email: tf.transform(email.split(" ")))

# Create LabeledPoint datasets for positive (spam) and negative (normal) examples.
positiveExamples = spamFeatures.map(lambda features: LabeledPoint(1, features))
negativeExamples = normalFeatures.map(lambda features: LabeledPoint(0, features))
trainingData = positiveExamples.union(negativeExamples)
trainingData.cache() # Cache since Logistic Regression is an iterative algorithm.

# Run Logistic Regression using the SGD algorithm.
model = LogisticRegressionWithSGD.train(trainingData)

# Test on a positive example (spam) and a negative one (normal). We first apply
# the same HashingTF feature transformation to get vectors, then apply the model.
posTest = tf.transform("O M G GET cheap stuff by sending money to ...".split(" "))
negTest = tf.transform("Hi Dad, I started studying Spark the other ...".split(" "))
print "Prediction for positive test example: %g" % model.predict(posTest)
print "Prediction for negative test example: %g" % model.predict(negTest)

print("spamFeatures: {}").format(spamFeatures)
print("normalFeatures: {}").format(normalFeatures)

print("positiveExamples: {}").format(positiveExamples)
print("negativeExamples: {}").format(negativeExamples)

print("trainingData: {}").format(trainingData)





#SPAM.TXT

#Dear sir, I am a Prince in a far kingdom you have not heard of.  I want to send you money via wire transfer so please ...
#Get Viagra real cheap!  Send money right away to ...
#Oh my gosh you can be really strong too with these drugs found in the rainforest. Get them cheap right now ...
#YOUR COMPUTER HAS BEEN INFECTED!  YOU MUST RESET YOUR PASSWORD.  Reply to this email with your password and SSN ...
#THIS IS NOT A SCAM!  Send money and get access to awesome stuff really cheap and never have to ...

#NORMAL.TXT
#Patrick, How are you?  Love, Dad
#Dear all, attached are the IPython notebooks for today's session.
#Paul, this is Joris from the Big Data team.  We've finally updated Spark in our cluster to 1.5.  Enjoy!
#Dear Patrick, Your books from Amazon have now shipped!

