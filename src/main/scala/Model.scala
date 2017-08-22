import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.mleap.SparkUtil
import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport.SparkTransformerOps
import resource._
/**
  * Created by gsaur on 8/21/17.
  */
case class Point(text: String, label: Double)

object Model {
  def main(args: Array[String]) : Unit = {
    val spark = SparkSession.builder()
      .master("local")
      .appName("Spam Classifier")
      .getOrCreate()
    val sc = spark.sparkContext
    val rdd = sc.textFile("data/smsspamcollection/SMSSpamCollection")
    val data = rdd.map(rawString => {
      val row = rawString.split("\t")
      if (row(0) == "spam") Point(row(1), 1.0)
      else {
        Point(row(1), 0.0)
      }
    }
    )

    val df = spark.createDataFrame(data)
    val splits = df.randomSplit(Array(.7, .3), seed = 10L)
    val training = splits(0).cache()
    val test = splits(1)

    // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setNumFeatures(10000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("hashed")
    val idf = new IDF().setInputCol(hashingTF.getOutputCol).setOutputCol("features")
    val lr = new LogisticRegression()
      .setMaxIter(2)
      .setRegParam(0.1)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, idf, lr))

    // Fit the pipeline to training documents.
    val model = pipeline.fit(training)

//    model.save("/Users/gsaur/workspaces/SparkTesting/Model10k")

    // Make predictions on test documents.
    val results = model.transform(test).cache()

    results
      .select("text", "probability", "prediction", "label")
      .collect()
      .foreach { case Row(text: String, prob: Vector, prediction: Double, label: Double) =>
        println(s"($text) --> prob=$prob, prediction=$prediction, label=$label")
      }

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    val accuracy = evaluator.evaluate(results)
    println("Test Error = " + (1.0 - accuracy))


    // Export model to Mleap
    val pipelineExport = SparkUtil.createPipelineModel(uid = "pipelineExport", Array(model))
    for(modelFile <- managed(BundleFile("jar:file:/Users/gsaur/workspaces/untitled/spamClassifierIM.zip"))) {
      pipelineExport.writeBundle.save(modelFile)
    }
  }
}
