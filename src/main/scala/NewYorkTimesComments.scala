import java.sql.Timestamp
import java.util.Properties

import scala.collection.JavaConverters._
import edu.stanford.nlp.pipeline.{CoreDocument, StanfordCoreNLP}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.sql._
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}


import scala.io.Source

case class Article(articleID: String,
                   articleWordCount: Int,
                   byline: String,
                   documentType: String,
                   headline: String,
                   keywords: String,
                   newDesk: String,
                   printPage: Int,
                   pubDate: Timestamp,
                   sectionName: String,
                   snippet: String,
                   source: String,
                   typeOfMaterial: String,
                   webUrl: String)

object NewYorkTimesComments {

  val article2017: Seq[String] = Seq("ArticlesJan2017.csv", "ArticlesFeb2017.csv", "ArticlesMarch2017.csv", "ArticlesApril2017.csv", "ArticlesMay2017.csv")
  val article2018: Seq[String] = Seq("ArticlesJan2018.csv", "ArticlesFeb2018.csv", "ArticlesMarch2018.csv", "ArticlesApril2018.csv")
  val articleAll: Seq[String] = article2017 ++ article2018

  val stopwords: Set[String] = Source.fromInputStream(getClass.getResourceAsStream("stopwords.txt")).getLines().toSet

  val sparkSession: SparkSession = SparkSession.builder.
    master("local")
    .appName("NYT Comments")
    .getOrCreate()
  LogManager.getRootLogger.setLevel(Level.WARN)

  import sparkSession.implicits._

  def main(args: Array[String]): Unit = {
    val articles: DataFrame = loadArticlesAsDF()
    articles.createOrReplaceTempView("articles")
    val headlineTokens = articles.map(row => preprocessText(row.getAs("headline")))

    // compute TF
    val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("value").setOutputCol("tf").fit(headlineTokens)
    val headlineTF = cvModel.transform(headlineTokens)

    // check TF
    println(headlineTF.first().getAs("value"))
    println(headlineTF.first().getAs("tf"))

    // compute IDF
    val idf = new IDF().setInputCol("tf").setOutputCol("idf")
    val idfModel = idf.fit(headlineTF)
    val rescaledData = idfModel.transform(headlineTF)

    rescaledData.show(10)
    print(rescaledData.schema)
    //rescaledData.select("label", "features").show()

  }

  def loadArticlesAsDF(): DataFrame = {
    val schema = Encoders.product[Article].schema
    sparkSession.read
      .option("header", "true")
      .option("charset", "UTF8")
      .option("delimiter", ",")
      .option("inferSchema", value = true)
      //.csv("data/test.csv")
      .csv("data/ArticlesJan2017.csv")
    //.csv(articleAll.map(f => "data/" + f): _*)
  }

  def loadArticlesAsDS(): Dataset[Article] = {
    val schema = Encoders.product[Article].schema
    sparkSession.read
      .option("header", "true")
      .option("charset", "UTF8")
      .option("delimiter", ",")
      .schema(schema)
      .csv("data/ArticlesJan2017.csv")
      //.csv(articleAll.map(f => "data/" + f): _*)
      .as[Article]
  }

  def preprocessText(text: String): Seq[String] = {
    val properties = new Properties()
    properties.setProperty("annotators", "tokenize, ssplit, pos, lemma")
    val pipeline: StanfordCoreNLP = new StanfordCoreNLP(properties)
    val document: CoreDocument = new CoreDocument(text)
    pipeline.annotate(document)
    document.tokens().asScala.map(token => token.lemma())
      .filter(token => token.length > 2)
      .filter(token => !stopwords.contains(token))
      .map(token => token.toLowerCase)
  }
}
