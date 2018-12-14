import java.io.FileWriter
import java.sql.Timestamp
import java.util.Properties

import au.com.bytecode.opencsv.CSVParser
import breeze.linalg.DenseVector

import scala.collection.JavaConverters._
import edu.stanford.nlp.pipeline.{CoreDocument, StanfordCoreNLP}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.linalg.{Matrix, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.ml.feature.{CountVectorizer, IDF}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.feature.{HashingTF, Word2Vec, Word2VecModel}
import org.apache.spark.mllib.linalg.{Vector => SparkVector}

import scala.collection.mutable
import scala.io.Source

object NewYorkTimesComments {

  def toBreeze(v:SparkVector) = BV(v.toArray)
  def fromBreeze(bv:BV[Double]): SparkVector = Vectors.dense(bv.toArray)
  def add(v1:SparkVector, v2:SparkVector): SparkVector = fromBreeze(toBreeze(v1) + toBreeze(v2))
  def scalarMultiply(a:Double, v:SparkVector): SparkVector = fromBreeze(a * toBreeze(v))

  case class QualitativeVar (name: String, modalities: Seq[String]) {
    def getModalities: Seq[String] = {
      for (modality <- modalities) yield name.toUpperCase + "." + modality
    }
  }

  val w2vSize = 200

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
    master("local[4]")
    .appName("NYT Comments")
    //.config("spark.driver.cores", "2")
    .getOrCreate()
  LogManager.getRootLogger.setLevel(Level.WARN)

  import sparkSession.implicits._

//  def preprocessTextUdf() = {
//    udf(
//      (snippet: String) => preprocessText(snippet)
//    )
//  }

  def main(args: Array[String]): Unit = {


    loadArticlesAsDF().take(10).foreach(println(_))
    //basicStats(articles)

    val (articledf, bVocabulary) = compteTfIdf(loadArticlesAsDF())


    val w2vModel = Word2VecModel.load(sparkSession.sparkContext, "w2vmodel-enwik9")
    //val w2vModel = buildW2VModel(buildW2VCorpus(articledf))
    //w2vModel.save(sparkSession.sparkContext, "w2vmodel-enwik9")
    val bW2VModel = sparkSession.sparkContext.broadcast(w2vModel)

    articledf.take(10).foreach(row => {
      println()
      val vocabulary = bVocabulary.value
      val snippet = row.getAs[String]("snippet")
      println(snippet)
      val tfidf = row.getAs[SparseVector]("snippet_tfidf")
      val tfidfTokens = tfidf.indices.zip(tfidf.values).map(zipped => (vocabulary.apply(zipped._1), zipped._2))
      computeTextW2V(bW2VModel.value, tfidfTokens).foreach(v => bW2VModel.value.findSynonyms(v, 10).foreach(println(_)))
    })
  }

  def computeTextVector(model: Word2VecModel, vocabulary: Array[String], tfidf: SparseVector): Option[Vector] = {
    val tfidfTokens = tfidf.indices.zip(tfidf.values).map(zipped => (vocabulary.apply(zipped._1), zipped._2))
    computeTextW2V(model, tfidfTokens)
  }

  def addTextVector(model: Word2VecModel, vocabulary: Array[String], df: DataFrame): DataFrame = {
    val compteTextVectorUdf = udf(compteTextVectorUdf)
    df.withColumn("vector", null)
  }

  def compteTfIdf(df: DataFrame): (DataFrame, Broadcast[Array[String]]) = {
    val preprocessTextUdf = udf(preprocessText(_))
    var articles = df.withColumn("snippet_tokens", preprocessTextUdf($"snippet"))

    val cvModel = new CountVectorizer().setInputCol("snippet_tokens").setOutputCol("snippet_count").fit(articles)
    articles = cvModel.transform(articles)
    val bVocabulary = sparkSession.sparkContext.broadcast(cvModel.vocabulary)
    val idfModel = new IDF().setInputCol("snippet_count").setOutputCol("snippet_tfidf").fit(articles)
    (idfModel.transform(articles), sparkSession.sparkContext.broadcast(cvModel.vocabulary))
  }

  def buildW2VCorpus(articles: DataFrame): RDD[Seq[String]] = {
    val xmlPages: RDD[String] = LSAUtils.readFile("data/enwik9", sparkSession.sparkContext).sample(withReplacement = false, fraction = 0.1)
    val wikipediaParagraph: RDD[(String, String)] = xmlPages.filter(_ != null).flatMap(LSAUtils.wikiXmlToPlainText).flatMap(r => {
      r._2.split("\n").map(s => (r._1, s.trim))
    }).filter(r => !r._2.equals(""))

    val nytLemmas = articles.rdd.map(row => row.getAs[Seq[String]]("snippet_tokens"))
    val wikiLemmas: RDD[Seq[String]] = wikipediaParagraph.map(row => preprocessText(row._2))
    val corpus = nytLemmas.union(wikiLemmas).filter(row => row.length > 5)
    println("number of paragraphs : " + corpus.count())
    corpus
  }

  def buildW2VModel(corpus: RDD[Seq[String]]): Word2VecModel = {
    val word2vec = new Word2Vec()
    word2vec.setVectorSize(w2vSize)
    word2vec.fit(corpus)
  }

  def computeTextW2V(model: Word2VecModel, tfidfTokens: Seq[(String, Double)]): Option[Vector] = {
    val vectSize = w2vSize
    var vSum = Vectors.zeros(vectSize)
    var vNb: Double = 0.0
    tfidfTokens.foreach(x => {
      val word: String = x._1
      val tfidf: Double = x._2
      model.getVectors.get(word).foreach(v => {
        vSum = add(scalarMultiply(tfidf, Vectors.dense(v.map(_.toDouble))), vSum)
        vNb += tfidf
      })
    })
    if (vNb != 0) {
      vSum = scalarMultiply(1.0 / vNb, vSum)
    }
    Option(vSum).filter(Vectors.norm(_, 1.0) > 0.0)
  }

  def analyseArticleTyplogy(articles: DataFrame): Unit = {
    val colNames = Seq("documentType", "newDesk", "source", "typeOfMaterial")
    val tdc = disjunctiveForm(articles, colNames)
    val matIndividus: BlockMatrix = toBlockMatrix(new RowMatrix(tdc._1))
    val matVariables: BlockMatrix = matIndividus.transpose
    val burtTable = matVariables.multiply(matIndividus)
    printBurtTable(burtTable.toLocalMatrix(), tdc._2, "data_results/burt_table.txt")
  }

  def toBlockMatrix(rm: RowMatrix): BlockMatrix = {
    new IndexedRowMatrix(rm.rows.zipWithIndex().map({case (row, idx) => IndexedRow(idx, row)})).toBlockMatrix()
  }

  def printBurtTable(bt: Matrix, mod: Seq[QualitativeVar], filename: String): Unit = {
    val w = new FileWriter(filename)
    val modalities = mod.flatMap(m => m.getModalities)
    w.append(modalities.mkString(",")).append('\n')
    bt.rowIter.zip(modalities.iterator).foreach({
      case (vector, modality) => w.append(vector.toArray.mkString(",")).append('\n')
    })
    w.close()
  }

  def normalizeTdc(tdc: RDD[Vector], size: Long): RDD[Vector] = {
    val sumVector = tdc.reduce((v1, v2) => addVectors(v1, v2))
    val tdcNorm = tdc.map(v => norm(v, sumVector, size))
    tdcNorm
  }

  def norm(v1: Vector, sumV: Vector, numArticles: Long): Vector = {
    val p = new DenseVector(sumV.toDense.values) / numArticles.toDouble
    val x = (new DenseVector(v1.toDense.values) /:/ p) - 1.0
    Vectors.dense(x.data)
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

    println("number of articles : " + articles.count())
    println(articles.show(20))
    articles.describe("articleWordCount", "printPage").show()

    printHist("src/main/gnuplot/count_word_hist.txt", articles.select("articleWordCount").map(v => v.getInt(0)).rdd.histogram(20))
    printHist("src/main/gnuplot/print_page_hist.txt", articles.select("printPage").map(v => v.getInt(0)).rdd.histogram(20))

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
    document.tokens().asScala.map(token => token.originalText())
      .filter(token => token.length > 2)
      .filter(token => !stopwords.contains(token))
      .map(token => token.toLowerCase)
  }



  def printHist(filename: String, hist: (Array[Double], Array[Long])): Unit = {
    val writer = new FileWriter(filename)
    hist._1.zip(hist._2).foreach(col => writer.append(col._1.toString).append(' ').append(col._2.toString).append('\n'))
    writer.close()

  }
}
