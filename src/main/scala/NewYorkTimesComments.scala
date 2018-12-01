import java.io.FileWriter
import java.sql.Timestamp
import java.util.Properties

import au.com.bytecode.opencsv.CSVParser
import breeze.linalg.DenseVector

import scala.collection.JavaConverters._
import edu.stanford.nlp.pipeline.{CoreDocument, StanfordCoreNLP}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._

import scala.collection.mutable
import scala.io.Source

object NewYorkTimesComments {

  case class QualitativeVar (
    name: String,
    modalities: Seq[String]
  )

  val customSchema = StructType(Array(
    StructField("articleID", StringType, nullable = true),
    StructField("articleWordCount", IntegerType, nullable = true),
    StructField("byline", StringType, nullable = true),
    StructField("documentType", StringType, nullable = true),
    StructField("headline", StringType, nullable = true),
    StructField("keywords", StringType, nullable = true),
    StructField("newDesk", StringType, nullable = true),
    StructField("printPage", IntegerType, nullable = true),
    StructField("pubDate", TimestampType, nullable = true),
    StructField("sectionName", StringType, nullable = true),
    StructField("snippet", StringType, nullable = true),
    StructField("source", StringType, nullable = true),
    StructField("typeOfMaterial", StringType, nullable = true),
    StructField("webURL", StringType, nullable = true)
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
    //articles.groupBy("snippet").count().orderBy(desc("count")).show()

    val tdc = disjunctiveForm(articles, Seq("documentType", "newDesk", "source", "typeOfMaterial"))
    //tdc._1.take(10).foreach(println(_))
    val sumVector = tdc._1.reduce((v1, v2) => addVectors(v1, v2))
    println(sumVector.toDense)
    //val modalities = articles.select("sectionName").distinct().orderBy(asc("sectionName")).map(row => row.getString(0)).collect()
    //disjunctiveForm(articles, "sectionName", modalities).show(10)
    basicStats(articles)
    //lsa(articles, "headline", "snippet")

  }

  def addVectors(v1: Vector, v2: Vector): Vector = {
    Vectors.dense((new DenseVector(v1.toDense.values) + new DenseVector(v2.toDense.values)).data).toSparse
  }

  def rowToVector(row: Row, vars: Seq[QualitativeVar]): Vector = {
    val v: Seq[Double] = for (variable <- vars; modality <- variable.modalities) yield {
      if (row.getAs(variable.name).equals(modality)) {
        1.0
      } else {
        0.0
      }
    }
    Vectors.dense(v.toArray).toSparse
  }

  def disjunctiveForm(df: DataFrame, colNames: Seq[String]): (RDD[Vector], Seq[QualitativeVar]) = {
    val vars = for (colName <- colNames) yield {
      val modalities = df.select(colName).distinct().orderBy(asc(colName)).map(row => row.getString(0)).collect()
      QualitativeVar(colName, modalities)
    }
    (df.rdd.map(row => rowToVector(row, vars)), vars)
  }

  def lsa(df: DataFrame, titleCol: String, textCol: String) = {
    val tokens: RDD[(String, Seq[String])] = df.map(row => (row.getAs[String](titleCol), preprocessText(row.getAs(textCol)))).rdd
    val numTerms = 1000
    val (termDocMatrix, termIds, docIds, idfs) =
      LSAUtils.termDocumentMatrix(tokens, numTerms, sparkSession.sparkContext)
    val mat = new RowMatrix(termDocMatrix)
    val k = 20
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

    articles.describe("articleWordCount", "printPage").show()

    printHist("count_word_hist.txt", articles.select("articleWordCount").map(v => v.getInt(0)).rdd.histogram(20))
    printHist("print_page_hist.txt", articles.select("printPage").map(v => v.getInt(0)).rdd.histogram(50))

    articles.groupBy("documentType").count().orderBy(desc("count")).show(100)
    articles.groupBy("newDesk").count().orderBy(desc("count")).show(100)
    articles.groupBy("sectionName").count().orderBy(desc("count")).show(100)
    articles.groupBy("source").count().orderBy(desc("count")).show(100)
    articles.groupBy("typeOfMaterial").count().orderBy(desc("count")).show(100)

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

  def parseCsvRow(headers: Seq[String], row: String): Row = {
    val parser: CSVParser = new CSVParser(',', '"')
    val article = parser.parseLine(row)
    val articleFields: mutable.TreeMap[String, Any] = mutable.TreeMap()
    article.zip(headers).foreach(articleField => articleFields(articleField._2) = convert(articleField._2, articleField._1))
    articleFields.remove("abstract")
    articleFields.remove("multimedia")
    if (articleFields.size != 14) {
      throw new Exception("an article should have 14 fields")
    }
    Row.fromSeq(articleFields.values.toSeq)
  }

  def convert(fieldName: String, fieldValue: String): Any = {
    fieldName match {
      case "articleWordCount" => Integer.parseInt(fieldValue)
      case "multimedia" => Integer.parseInt(fieldValue)
      case "printPage" => Integer.parseInt(fieldValue)
      case "pubDate" => Timestamp.valueOf(fieldValue)
      case _ => fieldValue
    }
  }

  def loadArticlesAsDF(): DataFrame = {
    var allRDD = sparkSession.sparkContext.emptyRDD[Row]
    for (path <- articleAll.map(p => "data/" + p)) {
      val csvFile: RDD[String] = sparkSession.sparkContext.textFile(path)
      val headers = csvFile.first().split(",").map(h => h.trim)
      val articleRDD = csvFile.mapPartitionsWithIndex({
        (idx, iter) => if (idx == 0) iter.drop(1) else iter
      }).map(row => parseCsvRow(headers, row))
      allRDD = allRDD.union(articleRDD)
    }
    sparkSession.createDataFrame(allRDD, customSchema)
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

  def printHist(filename: String, hist: (Array[Double], Array[Long])) = {
    val writer = new FileWriter(filename)
    hist._1.zip(hist._2).foreach(col => writer.append(col._1.toString).append(' ').append(col._2.toString).append('\n'))
    writer.close()

  }
}
