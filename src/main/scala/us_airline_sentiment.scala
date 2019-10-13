import java.io.{File, PrintWriter}
import java.util.Calendar

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.{Row, SparkSession}

object USAirlineSentiment {

  def main(args: Array[String]): Unit = {

    if (args.length < 2) {
      System.err.println("Usage: USAirlineSentiment <input file path> <output directory path>")
      System.exit(1)
    }

    val Array(srcDataFile, outputDir) = args.take(2)

    val time = Calendar.getInstance().getTime.toString.replaceAll(" ", "")
    val writer = new PrintWriter(new File(outputDir + "/output_" + time + ".txt"))

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val debug = false

    val spark = SparkSession
      .builder
      .appName("USAirlineSentiment")
      .getOrCreate()

    if (debug) println("Connected to Spark")

    spark.sparkContext.setLogLevel("ERROR")

    /** Load Data */

    // Create a DataFrame from the input data file
    var dataDF = spark.read
      .format("com.databricks.spark.csv") // allows reading CSV files as Spark DataFrames
      .option("delimiter", ",")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(srcDataFile)
      .withColumnRenamed("airline_sentiment", "sentiment")
      .withColumnRenamed("tweet_id", "id")
      .select("id", "text", "sentiment")

    if (debug) println("Data read into DataFrame | Rows: " + dataDF.count().toString)

    // Remove rows from the DataFrame that have null in text column
    dataDF = dataDF.na.drop(Seq("text"))

    if (debug) println("Dropped rows with null Text | Remaining Rows: " + dataDF.count().toString)

    println("Data loaded")
    writer.write("Data loaded\n")

    /** Set stages of Pre-processing */

    //Breaking down the sentence in text column into words
    val tokenizer = new RegexTokenizer().setPattern("[a-zA-Z'](\\w+)").setGaps(false).setMinTokenLength(3).setInputCol("text").setOutputCol("words")

    // Remove stop-words from the words column
    val remover = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("cleanWords")

    // Convert words to term-frequency vectors
    val hashingTF = new HashingTF().setInputCol(remover.getOutputCol).setOutputCol("features")
      .setNumFeatures(50)

    // Convert label to numeric format
    val indexer = new StringIndexer().setInputCol("sentiment").setOutputCol("label")

    if (debug) println("Pre-processing stages specified")

    /** Create Pipeline */

    // Create pre-processing timeline with all the steps
    val pipeline = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, indexer))

    if (debug) println("Pre processing Pipeline created")

    /** Pre-process the dataset using the pipeline */

    dataDF = pipeline.fit(dataDF).transform(dataDF)

    println("Data Pre-processed")
    writer.write("Data Pre-processed\n")

    if (debug) dataDF.show(2)
    if (debug) dataDF.dtypes.foreach(println)

    /** Specify Models */

    val nb = new NaiveBayes()
      .setModelType("multinomial")
      .setFeaturesCol("features")
      .setLabelCol("label")

    val lr = new LogisticRegression()
      .setFamily("multinomial")
      .setFeaturesCol("features")
      .setLabelCol("label")

    val rfc = new RandomForestClassifier()
      .setImpurity("entropy")
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxBins(50)

    println("Models specified")
    writer.write("Data Pre-processed\n")

    /** Create Pipeline and Parameter builder for Hyper-parameter tuning of models */

    val modelPipeline = new Pipeline()

    val paramGridNb = new ParamGridBuilder()
      .baseOn(modelPipeline.stages -> Array[PipelineStage](nb))
      .addGrid(nb.smoothing, Array(15.0, 20.0, 25))
      .build()

    val paramGridLr = new ParamGridBuilder()
      .baseOn(modelPipeline.stages -> Array[PipelineStage](lr))
      .addGrid(lr.maxIter, Array(20, 25, 30, 35))
      .addGrid(lr.regParam, Array(0.0005, 0.001, 0.005, 0.01))
      .addGrid(lr.elasticNetParam, Array(0.7, 0.8, 0.9))
      .build()

    val paramGridRf = new ParamGridBuilder()
      .baseOn(modelPipeline.stages -> Array[PipelineStage](rfc))
      .addGrid(rfc.maxDepth, Range(1, 11))
      .addGrid(rfc.numTrees, Array(5, 10, 15, 20))
      .build()

    val modelParamGrid = paramGridRf ++ paramGridNb ++ paramGridLr

    if (debug) println("Pipeline and Parameter grid built for models")

    /** Set evaluator */

    val modelEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy") // "f1" (default), "weightedPrecision", "weightedRecall", "accuracy"

    /** Find best model */

    val cv = new CrossValidator()
      .setEstimator(modelPipeline)
      .setEvaluator(modelEvaluator)
      .setEstimatorParamMaps(modelParamGrid)
      .setNumFolds(5)
      .setParallelism(3) // Evaluate up to 3 parameter settings in parallel

    if (debug) println("Cross validator set for finding best model parameters")

    /** Split the data into training and test sets */

    val Array(training, test) = dataDF.randomSplit(Array(0.9, 0.1), seed = 11L)

    if (debug) println("Training and Test set formed")

    /** Training with Best Model */

    println("Running cross-validation to choose the best model...")
    writer.write("Running cross-validation to choose the best model...\n")

    val cvModel = cv.fit(training)

    println("Best model found")
    writer.write("Best model found\n")

    val bestModel = cvModel.bestEstimatorParamMap
    println(bestModel)
    writer.write(bestModel.toString() + "\n")

    println("Trained using best model\n")

    /** Make predictions */

    // Make predictions on test set. cvModel uses the best model found.
    val prediction = cvModel.transform(test)

    println("Predictions made on test set\n")

    if (debug) prediction.select("id", "text", "label", "prediction")
      .collect()
      .foreach { case Row(id: String, text: String, label: Double, prediction: Double) =>
        println(s"prediction=$prediction, label=$label <- $text")
        writer.write(s"prediction=$prediction, label=$label <- $text")
      }

    /** Evaluate Model */

    val modelAccuracy = modelEvaluator.evaluate(prediction)
    println(s"Accuracy is: $modelAccuracy")
    writer.write(s"Accuracy is: $modelAccuracy\n")

    writer.close()
    spark.stop()

    if (debug) println("Disconnected from Spark")

  }

  implicit class BestParamMapCrossValidatorModel(cvModel: CrossValidatorModel) {
    def bestEstimatorParamMap: ParamMap = {
      cvModel.getEstimatorParamMaps
        .zip(cvModel.avgMetrics)
        .maxBy(_._2)
        ._1
    }
  }

}
