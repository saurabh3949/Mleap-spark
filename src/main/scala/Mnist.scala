import org.apache.log4j.Logger
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql._
import ml.combust.bundle.BundleFile
import org.apache.spark.ml.mleap.SparkUtil
import ml.combust.mleap.spark.SparkSupport.SparkTransformerOps
import org.apache.spark.ml.bundle.SparkBundleContext
import resource._


object MnistDriver {

  val logger = Logger.getLogger(getClass().getName())

  val errorMsg =
    """ MnistSpark needs 2 parameters to start:
      |
      | 1) A train CSV file.
      | 2) A test CSV file.
      |
    """.stripMargin

  def main(args: Array[String]) {
    if (args.length < 2) {
      println(errorMsg)
      throw new RuntimeException("Not enough args")
    }

    val trainFilePath = args(0)
    val testFilePath = args(1)

    val spark = SparkSession.builder()
      .master("local")
      .appName("MLP MNIST Classifier")
      .getOrCreate()
    val sc = spark.sparkContext

    var dataset = spark.sqlContext.read.format("com.databricks.spark.csv").
      option("header", "true").
      option("inferSchema", "true").
      load(trainFilePath)

    var test = spark.sqlContext.read.format("com.databricks.spark.csv").
      option("inferSchema", "true").
      option("header", "true").
      load(testFilePath)

    val predictionCol = "label"
    val labels = Seq("0","1","2","3","4","5","6","7","8","9")
    val pixelFeatures = (0 until 784).map(x => s"x$x").toArray

    //Layers of MLP - 784 -> 512 -> 10
    val layers = Array[Int](pixelFeatures.length, 512, labels.length)

    val vector_assembler = new VectorAssembler()
      .setInputCols(pixelFeatures)
      .setOutputCol("features")

    val scaler = new StandardScaler()
      .setInputCol(vector_assembler.getOutputCol)
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)


    val featurePipeline = new Pipeline().setStages(Array(vector_assembler, scaler))

    // Transform the raw data with the feature pipeline and persist it
    val featureModel = featurePipeline.fit(dataset)
    val datasetWithFeatures = featureModel.transform(dataset)

    // Select only the data needed for training and persist it
    val datasetPcaFeaturesOnly = datasetWithFeatures.select(predictionCol, scaler.getOutputCol)
    val datasetPcaFeaturesOnlyPersisted = datasetPcaFeaturesOnly.persist()

    // For test dataset
    val testDatasetWithFeatures = featureModel.transform(test)
    val testDatasetPcaFeaturesOnly = testDatasetWithFeatures.select(predictionCol, scaler.getOutputCol)
    val testDatasetPcaFeaturesOnlyPersisted = testDatasetPcaFeaturesOnly.persist()

    val mlp = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(50).
      setFeaturesCol(scaler.getOutputCol).
      setLabelCol(predictionCol).
      setPredictionCol("prediction")

    val mlpModel = mlp.fit(datasetPcaFeaturesOnlyPersisted)

    val results = mlpModel.transform(testDatasetPcaFeaturesOnlyPersisted).select("prediction", predictionCol).cache()

    val predictionAndLabels = results.rdd.map { case Row(prediction:Double, label:Int) =>
      (prediction, label.toDouble)
    }

    val metrics = new MulticlassMetrics(predictionAndLabels)

    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    // Overall Statistics
    val accuracy = metrics.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $accuracy")


    val pipeline = SparkUtil.createPipelineModel(uid = "pipe2line", Array(featureModel, mlpModel))

    val sbc = SparkBundleContext()
    for(bf <- managed(BundleFile("jar:file:/tmp/mnist.model5.mlp.zip"))) {
      pipeline.writeBundle.save(bf)(sbc).get
    }
  }
}