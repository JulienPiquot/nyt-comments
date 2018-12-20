import org.apache.kafka.clients.consumer.ConsumerRecord
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.dstream.{DStream, InputDStream}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.streaming.kafka010.{KafkaUtils, OffsetRange}
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
import org.apache.spark.streaming.kafka010.ConsumerStrategies.Subscribe

object KafkaIntegration {

  val sparkSession: SparkSession = SparkSession.builder.
    master("local[4]")
    .appName("NYT Comments")
    //.config("spark.driver.cores", "2")
    .getOrCreate()
  LogManager.getRootLogger.setLevel(Level.WARN)
  val ssc = new StreamingContext(sparkSession.sparkContext, Seconds(10))

  import sparkSession.implicits._
  import scala.collection.JavaConverters._

  val kafkaParams: Map[String, Object] = Map[String, Object](
    "bootstrap.servers" -> "localhost:9092",
    "key.deserializer" -> classOf[StringDeserializer],
    "value.deserializer" -> classOf[StringDeserializer],
    "group.id" -> "test_group",
    //"auto.offset.reset" -> "latest",
    "enable.auto.commit" -> (false: java.lang.Boolean),
    "auto.offset.reset" -> "earliest"
  )

  def main(args: Array[String]): Unit = {
    //buildArticlesRdd(sparkSession.sparkContext)
    streamArticles(ssc)

  }

  def buildArticlesRdd(sparkContext: SparkContext): Unit = {
    val offsetRanges = Array(
      // topic, partition, inclusive starting offset, exclusive ending offset
      OffsetRange("test", 0, 0, 100),
      OffsetRange("test", 1, 0, 100),
      OffsetRange("test", 2, 0, 100)
    )

    val rdd = KafkaUtils.createRDD[String, String](sparkContext, kafkaParams.asJava, offsetRanges, PreferConsistent)
      .map(record => (record.key, record.value))
    rdd.foreach(println(_))
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
