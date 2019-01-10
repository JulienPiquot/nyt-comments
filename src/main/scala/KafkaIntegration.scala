import NewYorkTimesComments.{sparkSession, _}
import org.apache.commons.math3.distribution.FDistribution
import org.apache.kafka.clients.consumer.ConsumerRecord
import org.apache.kafka.common.serialization.{ByteArrayDeserializer, StringDeserializer}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorIndexer}
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
    val comments: RDD[Comment] = buildCommentsRdd(sparkSession.sparkContext).sample(withReplacement = false, fraction = 0.25d).repartition(24)

    // calcul de l'ANOVA - rejet de l'hypothèse nulle potentiellement due a un effet "Big Data"
    //computeAnova(comments)

    // calcul des tfidfs
    val (commentDF, bVocabulary) = compteAndCacheTfIdf(sparkSession.createDataFrame(comments.map(c => Row.fromSeq(Seq(c.getCommentID, c.getCommentBody, c.getCommentType, c.getCreateDate, c.getDepth, c.isEditorsSelection, c.getRecommandations, c.getReplyCount, c.getSharing, c.getUserDisplayName, c.getUseLocation, c.getArticleID, c.getNewDesk, c.getArticleWordCount, c.getPrintPage, c.getTypeOfMaterial))),
      NewYorkTimesComments.commentSchema), "commentBody")
    commentDF.limit(10).foreach(print(_))
    println("number of partitions commentDF : " + commentDF.rdd.getNumPartitions)
    println("number of comments : " + commentDF.count())

    // entrainement du modele Word2Vec
    //val w2vModel = computeW2V(commentDF)
    val w2vModel = Word2VecModel.load(sparkSession.sparkContext, "comments-w2v")

    // vectorisation du texte grace aux representations w2v
    val vectorised = vectorizeText(commentDF, bVocabulary.value, w2vModel)

    // calcul de la decroissance des inerties intra classes
    //computeKMeansInerties(vectorised)

    // choix d'une valeur k et visualisation des classes
    //computeAndVisualizeKMeans(w2vModel, vectorised, 10)

    // run LSA in order to compare with
    //lsa(commentDF, "commentBody", "tokens", preprocess = false)

    // run the decision tree regression
    runDecisionTree(vectorised)
  }

  def runDecisionTree(comments: DataFrame) = {
    val toMLVector = udf((v: Vector, wc: Int, printPage: Int, typeOfMaterial: String, newDesk: String) => new org.apache.spark.ml.linalg.DenseVector(v.toArray ++ Array(wc.toDouble, printPage.toDouble, typeOfMaterial.hashCode.toDouble, newDesk.hashCode.toDouble)))
    val data: DataFrame = comments.withColumn("mlvector", toMLVector($"vector", $"articleWordCount", $"printPage", $"typeOfMaterial", $"newDesk")).cache()

    val featureIndexer = new VectorIndexer()
      .setInputCol("mlvector")
      .setOutputCol("indexedVector")
      .setMaxCategories(60)
      .fit(data)
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    val dt = new DecisionTreeRegressor()
      .setMaxDepth(10)
      .setMaxBins(64)
      .setMinInstancesPerNode(2)
      .setLabelCol("recommandations")
      .setFeaturesCol("indexedVector")
    val pipeline = new Pipeline().setStages(Array(featureIndexer, dt))
    val model = pipeline.fit(trainingData)
    val predictions = model.transform(testData)
    predictions.select("prediction", "recommandations", "indexedVector").show(5)
    val evaluator = new RegressionEvaluator()
      .setLabelCol("recommandations")
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
