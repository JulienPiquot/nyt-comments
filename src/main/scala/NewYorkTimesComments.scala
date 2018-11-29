import java.sql.Timestamp
import java.util.Properties

import au.com.bytecode.opencsv.CSVReader
import breeze.io.CSVReader

import scala.collection.JavaConverters._
import edu.stanford.nlp.pipeline.{CoreDocument, StanfordCoreNLP}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._

import scala.io.Source

object NewYorkTimesComments {

  val customSchema = StructType(Array(
    //StructField("abstract", StringType, true),
    StructField("articleID", StringType, true),
    StructField("articleWordCount", IntegerType, true),
    StructField("byline", StringType, true),
    StructField("documentType", StringType, true),
    StructField("headline", StringType, true),
    StructField("keywords", StringType, true),
    StructField("multimedia", IntegerType, true),
    StructField("newDesk", StringType, true),
    StructField("printPage", IntegerType, true),
    StructField("pubDate", TimestampType, true),
    StructField("sectionName", StringType, true),
    StructField("snippet", StringType, true),
    StructField("source", StringType, true),
    StructField("typeOfMaterial", StringType, true),
    StructField("webURL", StringType, true)
  ))

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
    articles.printSchema()
    println(articles.count())
    println(articles.show(50))


    lsa(articles, "headline", "snippet")

  }

  def lsa(df: DataFrame, titleCol: String, textCol: String) = {
    val tokens: RDD[(String, Seq[String])] = df.map(row => (row.getAs[String](titleCol), preprocessText(row.getAs(textCol)))).rdd
    val numTerms = 1000
    val (termDocMatrix, termIds, docIds, idfs) =
      LSAUtils.termDocumentMatrix(tokens, numTerms, sparkSession.sparkContext)
    val mat = new RowMatrix(termDocMatrix)
    val k = 200
    val svd = mat.computeSVD(k, computeU = true)

    val topConceptTerms = RunLSA.topTermsInTopConcepts(svd, 10, 10, termIds)
    val topConceptDocs = RunLSA.topDocsInTopConcepts(svd, 10, 10, docIds)
    for ((terms, docs) <- topConceptTerms.zip(topConceptDocs)) {
      println("Concept terms: " + terms.map(_._1).mkString("|"))
      println("Concept docs: " + docs.map(_._1).mkString("|"))
      println()
    }
  }

  def basicStats(articles: DataFrame): Unit = {
    val authorsCount = articles.flatMap(row => parseAuthor(row.getAs("byline")))
      .groupBy("value")
      .count()
      .orderBy(desc("count"))
    val authorNumber = authorsCount.count()
    println(s"number of authors : $authorNumber")
    authorsCount.show()

    val keywordsCount = articles.flatMap(row => parseKeywords(row.getAs("keywords")))
      .filter(author => !author.isEmpty)
      .groupBy("value")
      .count()
      .orderBy(desc("count"))
    val keywordNumber = keywordsCount.count()
    println(s"number of keyword : $keywordNumber")
    keywordsCount.show()
  }

  def parseKeywords(text: String): Seq[String] = {
    text.replaceAll("(^\\[)|(\\]$)", "").split("', '").map(_.replaceAll("'", "").trim)
  }

  def parseAuthor(text: String): Seq[String] = {
    text.replaceAll("^By ", "").split(",|( and )").map(_.trim)
  }

  def tfidf(df: DataFrame): DataFrame = {
    val tokens = df.map(row => preprocessText(row.getAs("snippet")))

    // compute TF
    val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("value").setOutputCol("tf").fit(tokens)
    val headlineTF = cvModel.transform(tokens)

    // compute IDF
    val idf = new IDF().setInputCol("tf").setOutputCol("idf")
    val idfModel = idf.fit(headlineTF)
    idfModel.transform(headlineTF)
  }

  def loadArticlesAsDF(): DataFrame = {
    val csv: CSVReader = new CSVReader(null, ',', "\"")

    //val schema = Encoders.product[Article].schema
    sparkSession.read
      .option("header", "true")
      .option("charset", "UTF8")
      .option("delimiter", ",")
      //.schema(customSchema)
      .option("inferSchema", value = true)
      //.csv("data/test.csv")
      .csv("data/ArticlesApril2017.csv")
      //.csv(articleAll.map(f => "data/" + f): _*)
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
