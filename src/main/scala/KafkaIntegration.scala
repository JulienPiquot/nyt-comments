import NewYorkTimesComments.{sparkSession, _}
import org.apache.commons.math3.distribution.FDistribution
import org.apache.kafka.clients.consumer.ConsumerRecord
import org.apache.kafka.common.serialization.{ByteArrayDeserializer, StringDeserializer}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor}
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.streaming.dstream.{DStream, InputDStream}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.streaming.kafka010.{KafkaUtils, OffsetRange}
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
import org.apache.spark.streaming.kafka010.ConsumerStrategies.Subscribe
import org.apache.spark.sql.functions._


object KafkaIntegration {

  val sparkSession: SparkSession = SparkSession.builder.
    master("local[*]")
    .appName("NYT Comments")
    //.config("spark.driver.cores", "4")
    //.config("spark.executor.cores", "4")
    .getOrCreate()
  LogManager.getRootLogger.setLevel(Level.WARN)
  //val ssc = new StreamingContext(sparkSession.sparkContext, Seconds(10))

  import scala.collection.JavaConverters._
  import sparkSession.implicits._

  val kafkaParams: Map[String, Object] = Map[String, Object](
    "bootstrap.servers" -> "localhost:9092",
    "key.deserializer" -> classOf[StringDeserializer],
    "value.deserializer" -> classOf[ByteArrayDeserializer],
    "group.id" -> "test_group",
    //"auto.offset.reset" -> "latest",
    "enable.auto.commit" -> (false: java.lang.Boolean),
    "auto.offset.reset" -> "earliest"
  )

  def main(args: Array[String]): Unit = {
    // entrainement du modele Word2Vec
    //val w2vModel = computeW2V(commentDF)
    val w2vModel = Word2VecModel.load(sparkSession.sparkContext, "comments-w2v")

    // vectorisation du texte grace aux representations w2v
    //var vectorised = prepareData(w2vModel)
    var vectorised = sparkSession.read.parquet("prepared-data.small.parquet").repartition(24).cache()
    println("number of partitions commentDF : " + vectorised.rdd.getNumPartitions)
    println("number of comments : " + vectorised.count())
    vectorised.show(5, false)
    vectorised.select("recommandations", "normRecommandations", "logRecommandations").summary().show()
    vectorised.groupBy($"discreteRecommandations").count().show()
    vectorised.groupBy($"discreteNormRecommandations").count().show()


    // calcul de l'ANOVA - rejet de l'hypothèse nulle potentiellement due a un effet "Big Data"
    //computeAnova(vectorised)

    // calcul de la decroissance des inerties intra classes
    //computeKMeansInerties(vectorised)

    // choix d'une valeur k et visualisation des classes
    //computeAndVisualizeKMeans(w2vModel, vectorised, 10)

    // run LSA in order to compare with
    //lsa(vectorised, "commentBody", "tokens", preprocess = false)

    //runDecisionTree(vectorised)
    //checkVariableRelations(vectorised)
    //runDecisionTreeClassification(vectorised)
  }

  def checkVariableRelations(comments: DataFrame): Unit = {
    // check relation between editorsSelection and articleWordCount
    comments.createOrReplaceTempView("comments")
    sparkSession.sql("SELECT COUNT(articleWordCount), MEAN(articleWordCount), STDDEV(articleWordCount), MIN(articleWordCount), MAX(articleWordCount) FROM comments GROUP BY editorsSelection").show()

    //comments.groupBy($"editorsSelection").agg(count($"articleWordCount"), mean($"articleWordCount"), stddev($"articleWordCount"), min($"articleWordCount"), max($"articleWordCount")).show()
    //comments.groupBy($"editorsSelection").agg(count($"printPage"), mean($"printPage"), stddev($"printPage"), min($"printPage"), max($"printPage")).show()
  }

  def prepareData(word2VecModel: Word2VecModel): DataFrame = {
    val comments: RDD[Comment] = buildCommentsRdd(sparkSession.sparkContext).sample(withReplacement = false, fraction = 0.01d).repartition(24)

    // calcul des tfidfs
    var commentDF = sparkSession.createDataFrame(comments.map(c => Row.fromSeq(Seq(c.getCommentID, c.getCommentBody, c.getCommentType, c.getCreateDate, c.getDepth, c.isEditorsSelection, c.getRecommandations, c.getReplyCount, c.getSharing, c.getUserDisplayName, c.getUseLocation, c.getArticleID, c.getNewDesk, c.getArticleWordCount, c.getPrintPage, c.getTypeOfMaterial))),
      NewYorkTimesComments.commentSchema).cache()
    val preprocessTextUdf = udf((t: String) => {
      val sentiments = sentimentAnalysis(t)
      (sentiments.tokens, sentiments.sentiments)
    })
    val selectTokens = udf((nlpResult: Row) => nlpResult.getAs[Seq[String]](0))
    val selectSentiments = udf((nlpResult: Row) => {
      val sentiments = nlpResult.getAs[Vector](1)
      if (sentiments.size != 3) {
        throw new RuntimeException()
      }
      sentiments
    })
    commentDF = commentDF.withColumn("nlp", preprocessTextUdf($"commentBody")).cache()
    commentDF = commentDF.withColumn("tokens", selectTokens($"nlp")).withColumn("sentimentVector", selectSentiments($"nlp"))

    val (tfidfDF, bVocabulary) = compteAndCacheTfIdf(commentDF)
    commentDF = tfidfDF.cache()
    commentDF.limit(10).foreach(print(_))

    // vectorisation du texte grace aux representations w2v
    var vectorised = vectorizeText(commentDF, bVocabulary.value, word2VecModel).cache()
    vectorised = normalisePredictions(vectorised)
    vectorised.write.mode("overwrite").parquet("prepared-data.small.parquet")
    vectorised
  }

  def runSentimentAnalysis(df: DataFrame): DataFrame = {
    val sentAnalysis = udf((text: String) => NewYorkTimesComments.sentimentAnalysis(text))
    df.withColumn("sentimentVector", sentAnalysis($"commentBody")).cache()
  }

  def normalisePredictions(df: DataFrame): DataFrame = {
    val descretiseReco = udf((r: Int) => {
      if (r == 0) 0
      else if (r <= 1) 1
      else if (r <= 10) 2
      else if (r <= 100) 3
      else 4
    })
    val descretiseNormReco = udf((r: Double) => {
      if (r == 0) 0
      else if (r <= 0.01) 1
      else if (r <= 0.1) 2
      else if (r <= 0.5) 3
      else 4
    })
    val normReco = udf((n: Int, d: Int) => n.toDouble / d.toDouble)
    val logReco = udf((r: Int) => if(r == 0) -1 else Math.log10(r))
    val sumDf = df.groupBy($"articleID").agg(sum($"recommandations") as "sum_recommandations").filter(r => r.getAs[Long]("sum_recommandations") > 0).cache()
    val normDf = df.join(sumDf, "articleID")
      .withColumn("normRecommandations", normReco($"recommandations", $"sum_recommandations"))
      .withColumn("logRecommandations", logReco($"recommandations"))
      .withColumn("discreteRecommandations", descretiseReco($"recommandations"))
      .withColumn("discreteNormRecommandations", descretiseNormReco($"normRecommandations")).cache()
    normDf.groupBy($"discreteRecommandations").count().show()
    normDf.groupBy($"discreteNormRecommandations").count().show()
    normDf.select("recommandations", "normRecommandations", "logRecommandations").summary().show()
    printHist("src/main/gnuplot/logrecommandation_hist.txt", normDf.map(r => r.getAs[Double]("logRecommandations")).rdd.histogram(20))
    printHist("src/main/gnuplot/recommandation_hist.txt", normDf.map(r => r.getAs[Int]("recommandations")).rdd.histogram(20))
    printHist("src/main/gnuplot/normrecommandation_hist.txt", normDf.map(r => r.getAs[Double]("normRecommandations")).rdd.histogram(20))
    normDf
  }

  def runDecisionTreeClassification(comments: DataFrame): Unit = {
    //    val toMLVector = udf((v: Vector, wc: Int, printPage: Int, typeOfMaterial: String, newDesk: String) => new org.apache.spark.ml.linalg.DenseVector(v.toArray ++ Array(wc.toDouble, printPage.toDouble, typeOfMaterial.hashCode.toDouble, newDesk.hashCode.toDouble)))
    val toMLVector = udf((v: Vector, s: Vector, wc: Int, printPage: Int) => new org.apache.spark.ml.linalg.DenseVector(v.toArray ++ s.toArray ++ Array(wc.toDouble, printPage.toDouble)))
    var data: DataFrame = comments.withColumn("mlvector", toMLVector($"vector", $"sentimentVector", $"articleWordCount", $"printPage")).cache()
    data.show(10)

    val varToPredict = "editorsSelectionLabel"
    val toIntUdf = udf((b: Boolean) => if (b) 1 else 0)
    data = data.withColumn(varToPredict, toIntUdf($"editorsSelection"))
    data.groupBy(varToPredict).count().show()

    val featureIndexer = new VectorIndexer()
      .setInputCol("mlvector")
      .setOutputCol("indexedVector")
      .setMaxCategories(4)
      .fit(data)
    val labelIndexer = new StringIndexer()
      .setInputCol(varToPredict)
      .setOutputCol("indexedLabel")
      .fit(data)
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // Train a DecisionTree model.
    val dt = new GBTClassifier()
      //.setMaxBins(64)
      //.setMinInstancesPerNode(2)
      .setLabelCol(varToPredict)
      .setFeaturesCol("indexedVector")
      .setMaxIter(10)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)

    // Make predictions
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("prediction", varToPredict, "mlVector").show(10)
    predictions.select("prediction", varToPredict).printSchema()
    val predictionsAndLabels: RDD[(Double, Double)] = predictions.select("prediction", varToPredict).map(r => (r.getDouble(0), r.getInt(1).toDouble)).rdd
    val metrics = new org.apache.spark.mllib.evaluation.MulticlassMetrics(predictionsAndLabels)
    println(metrics.confusionMatrix)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    val treeModel = model.stages(2).asInstanceOf[GBTClassificationModel]
    println("Learned classification tree model:\n" + treeModel.toDebugString)
    println(treeModel.featureImportances)
  }

  def runDecisionTreeRegression(comments: DataFrame): Unit = {
//    val toMLVector = udf((v: Vector, wc: Int, printPage: Int, typeOfMaterial: String, newDesk: String) => new org.apache.spark.ml.linalg.DenseVector(v.toArray ++ Array(wc.toDouble, printPage.toDouble, typeOfMaterial.hashCode.toDouble, newDesk.hashCode.toDouble)))
    val toMLVector = udf((v: Vector, s: Vector, wc: Int) => new org.apache.spark.ml.linalg.DenseVector(v.toArray ++ s.toArray ++ Array(wc.toDouble)))

    val data: DataFrame = comments.withColumn("mlvector", toMLVector($"vector", $"sentimentVector", $"articleWordCount")).cache()
    data.show(10)

    val varToPredict = "logRecommandations"

    val featureIndexer = new VectorIndexer()
      .setInputCol("mlvector")
      .setOutputCol("indexedVector")
      .setMaxCategories(60)
      .fit(data)
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    val dt = new DecisionTreeRegressor()
      .setMaxBins(64)
      .setMinInstancesPerNode(2)
      .setLabelCol(varToPredict)
      .setFeaturesCol("indexedVector")
    val pipeline = new Pipeline().setStages(Array(featureIndexer, dt))
    val model = pipeline.fit(trainingData)
    val predictions = model.transform(testData)
    predictions.select("prediction", varToPredict, "indexedVector").show(10)
    val evaluator = new RegressionEvaluator()
      .setLabelCol(varToPredict)
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)
    val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
    println("Learned regression tree model:\n" + treeModel.toDebugString)
  }

  def vectorizeText(comments: DataFrame, voc: Array[String], w2v: Word2VecModel): DataFrame = {
    val embeddings = sparkSession.sparkContext.broadcast(w2v.getVectors)
    val vectorised = addVector(embeddings.value, voc, comments).cache()
    println("number of vectorized paritions : " + vectorised.rdd.getNumPartitions)
    println("### embeddings examples ###")
    vectorised.head(10).foreach(row => {
      println(row.getAs[String]("commentBody"))
      w2v.findSynonyms(row.getAs[Vector]("vector"), 10).foreach(println(_))
    })
    vectorised
  }

  def computeW2V(comments: DataFrame): Word2VecModel = {
    val corpus = buildW2VCorpus(comments).cache()
    println("number of paragraphs : " + corpus.count())
    println("number of partitions corpus : " + corpus.getNumPartitions)
    val w2vModel = buildW2VModel(corpus)
    w2vModel.save(sparkSession.sparkContext, "comments-w2v")
    w2vModel
  }

  def computeKMeansInerties(comments: DataFrame): Unit = {
    println("number of vectorized paritions : " + comments.rdd.getNumPartitions)
    println("### embeddings ###")
    for (i <- 1 to 20) {
      println("### kmeans k = " + i + " ###")
      kmeans(comments, i, null)
    }
  }

  def computeAndVisualizeKMeans(w2v: Word2VecModel, comments: DataFrame, k: Int): Unit = {
    kmeans(comments, k, w2v)
  }

  def computeAnova(comments: RDD[Comment]): Unit = {
    val stats = Anova.getAnovaStats(sparkSession, comments.map(comment => CatTuple(comment.getArticleID, comment.getRecommandations)).toDS())
    println(stats)
    val fdist: FDistribution = new FDistribution(null, stats.dfb, stats.dfw)
    println("p value is :" + (1.0 - fdist.cumulativeProbability(stats.F_value)))
  }

  def buildCommentsRdd(sparkContext: SparkContext): RDD[Comment] = {
    val offsetRanges = Array(
      // topic, partition, inclusive starting offset, exclusive ending offset
      // max num is 544085
      OffsetRange("comments", 0, 0, 544085),
      OffsetRange("comments", 1, 0, 544085),
      OffsetRange("comments", 2, 0, 544085),
      OffsetRange("comments", 3, 0, 544085)
    )

    KafkaUtils.createRDD[String, Array[Byte]](sparkContext, kafkaParams.asJava, offsetRanges, PreferConsistent)
      .map(record => Comment.deserialize(record.value))
  }

  def streamArticles(ssc: StreamingContext): Unit = {
    val topics = Array("test")
    val stream: DStream[(String, String)] = KafkaUtils.createDirectStream[String, String](
      ssc,
      PreferConsistent,
      Subscribe[String, String](topics, kafkaParams)
    ).map(record => (record.key, record.value))

    println("Start streaming ...")
    stream.foreachRDD(rdd => {
      println("rdd size " + rdd.count())
      rdd.foreach(kv => println(kv._2))
    })
    ssc.start()
    ssc.awaitTermination()
    //ssc.stop(stopSparkContext = true, stopGracefully = true)
  }
}
