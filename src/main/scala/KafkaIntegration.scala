import NewYorkTimesComments.{sparkSession, _}
import org.apache.commons.math3.distribution.FDistribution
import org.apache.kafka.clients.consumer.ConsumerRecord
import org.apache.kafka.common.serialization.{ByteArrayDeserializer, StringDeserializer}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.streaming.dstream.{DStream, InputDStream}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.streaming.kafka010.{KafkaUtils, OffsetRange}
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
import org.apache.spark.streaming.kafka010.ConsumerStrategies.Subscribe

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
    val comments: RDD[Comment] = buildCommentsRdd(sparkSession.sparkContext).sample(withReplacement = false, fraction = 0.25).repartition(24)
    //comments.take(10).foreach(println(_))
    println("number of partitions : " + comments.getNumPartitions)
    println("number of comments : " + comments.count())

    // calcul de l'ANOVA - rejet de l'hypothèse nulle potentiellement due a un effet "Big Data"
    //computeAnova(comments)
    val commentDF = sparkSession.createDataFrame(comments.map(c => Row.fromSeq(Seq(c.getCommentID, c.getCommentBody, c.getCommentType, c.getCreateDate, c.getDepth, c.isEditorsSelection, c.getRecommandations, c.getReplyCount, c.getSharing, c.getUserDisplayName, c.getUseLocation, c.getArticleID, c.getNewDesk, c.getArticleWordCount, c.getPrintPage, c.getTypeOfMaterial))),
      NewYorkTimesComments.commentSchema).cache()
    val (commentDF2, bVocabulary) = compteAndCacheTfIdf(commentDF, "commentBody")
    commentDF2.limit(10).foreach(print(_))
    println("number of partitions commentDF2 : " + commentDF2.rdd.getNumPartitions)
    val corpus = buildW2VCorpus(commentDF2).cache()
    println("number of paragraphs : " + corpus.count())
    println("number of partitions corpus : " + corpus.getNumPartitions)
    val w2vModel = buildW2VModel(corpus)
//    w2vModel.save(sparkSession.sparkContext, "comments-w2v")
    val bW2VModel = sparkSession.sparkContext.broadcast(w2vModel.getVectors)
    val vectorised = addVector(bW2VModel.value, bVocabulary.value, commentDF2).cache()
    println("number of vectorized paritions : " + vectorised.rdd.getNumPartitions)
    println("### embeddings ###")
    vectorised.head(10).foreach(row => {
      println(row.getAs[String]("commentBody"))
      w2vModel.findSynonyms(row.getAs[Vector]("vector"), 10).foreach(println(_))
    })
    for (i <- 1 to 20) {
      println("### kmeans k = " + i + " ###")
      kmeans(vectorised, i, null)
    }
//    println("### lsa ###")
//    lsa(commentDF, "commentID", "commentBody")
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
