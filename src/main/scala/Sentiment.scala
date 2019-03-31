import java.io.{BufferedWriter, FileWriter}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.SparkSession

object Sentiment {
  def main(args: Array[String]) {

    val spark = SparkSession
      .builder
      .appName("Sentence Sentiment Analysis")
      .master("local") // remove this when running in a Spark cluster
      .getOrCreate()

    println("Connected to Spark")

    // Display only ERROR logs in terminal
    spark.sparkContext.setLogLevel("ERROR")

    // Get current time
    val xt = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYMMddHHmmss"))

    // Specify data file
    val dataFile = "data/sentiment_labelled_sentences/*.txt"

    // Specify output file
    val outFile = new BufferedWriter(new FileWriter(("sentiment_" + xt + ".txt")))

    // Create DataFrame using the data file
    val df = spark.read
      .option("header", "false")
      .option("inferSchema", "true")
      .option("delimiter", "\t")
      .csv(dataFile)
      .na.drop() // remove rows with null or NaN values
      .toDF("text", "label")

    // tokenize the text to words
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val tokenizedDF = tokenizer.transform(df).select("words", "label")

    // remove the stopwords
    val stopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("cleanWords")
    val cleanDF = stopWordsRemover.transform(tokenizedDF).select("cleanWords", "label")

    // create feature set from 1000 words
    val hashingTF = new HashingTF().setInputCol("cleanWords").setOutputCol("featureSet").setNumFeatures(1000)
    val featurizedDF = hashingTF.transform(cleanDF).select("featureSet", "label")

    // rescale feature set
    val idf = new IDF().setInputCol("featureSet").setOutputCol("features")
    val idfModel = idf.fit(featurizedDF) // rescaling
    val rescaledDF = idfModel.transform(featurizedDF).select("features", "label")

    // split data into train and test sets
    val Array(trainSet, test) = rescaledDF.randomSplit(Array(0.8, 0.2))
    val Array(train, vali) = trainSet.randomSplit(Array(0.8, 0.2))

    /** Learn models and check accuracy */

    val evaluator = new MulticlassClassificationEvaluator()

    // Create list of regularization values for the model to be trained with
    val regList = List(0.001, 0.01, 0.1, 1.0, 10)

    var accMap: Map[Double, Double] = Map()

    for (c <- regList) {

      /** Choose Classifier */
      val classifier = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")
        .setMaxIter(100)
        .setRegParam(c)
        .setElasticNetParam(0.6)

      /** Train model */
      val model = classifier.fit(train)

      /** Validate model */
      val result = model.transform(vali)
      result.select("label", "prediction")

      /** Check accuracy */
      evaluator.setLabelCol("label").setMetricName("accuracy")
      val accuracy = evaluator.evaluate(result)

      accMap += (c -> accuracy)

    }

    for ((c, accuracy) <- accMap){
      print("Accuracy on train set\n")
      outFile.append("Reg: " + c + " Accuracy: " + "%6.3f".format(accuracy) + "\n")
      println("Reg: " + c + " Accuracy: " + "%6.3f".format(accuracy))
    }

    val c_best = accMap.maxBy(_._2)._1

    outFile.append("Accuracy on train set is maximum for Reg= " + c_best.toString())
    println("Accuracy on train set is maximum for Reg= " + c_best.toString())

    /** Select best classifier */
    val best_classifier = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")
      .setMaxIter(100)
      .setRegParam(c_best)
      .setElasticNetParam(0.6)

    /** Create and save the best model */
    val best_model = best_classifier.fit(train)
    best_model.save("best.model")
    outFile.append("Best model is saved\n")
    println("Best model is saved")

    /** Select best model and run the TEST program with it */
    val savedModel = LogisticRegressionModel.load("best.model")
    outFile.append("Best model loaded\n")
    println("Best model loaded")

    val testResult = savedModel.transform(test)
    testResult.select("label", "prediction")

    val testAccuracy = evaluator.evaluate(testResult)

    outFile.append("Accuracy on test set is = " + testAccuracy)
    println("Accuracy on test set is = " + testAccuracy)

    outFile.close()

    spark.stop()
    println("Disconnected from Spark")

  }

}
