import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, RelationalGroupedDataset, SparkSession}
import org.apache.spark.sql.functions.lit

final case class CatTuple(cat: String, value: Double)

final case class ANOVAStats(dfb: Long, dfw: Double, F_value: Double, etaSq: Double, omegaSq: Double)

object Anova {
  def getAnovaStats(spark: SparkSession, categoryData: org.apache.spark.sql.Dataset[CatTuple]): ANOVAStats = {
    categoryData.createOrReplaceTempView("df")
    val newdf = spark.sql(
      """select A.cat, A.value, cast((A.value * A.value) as double) as valueSq, ((A.value - B.avg) * (A.value - B.avg)) as diffSq
        | from df A
        | join (
        |     select cat, avg(value) as avg from df group by cat
        | ) B
        | where A.cat = B.cat""".stripMargin)
    val grouped = newdf.groupBy("cat")
    val sums = grouped.sum("value")
    val counts = grouped.count
    val numCats = counts.count
    val sumsq = grouped.sum("valueSq")
    val avgs = grouped.avg("value")

    val totN: Double = counts.agg(org.apache.spark.sql.functions.sum("count")).first.get(0) match {
      case d: Double => d
      case l: Long => l.toDouble
    }
    val totSum: Double = sums.agg(org.apache.spark.sql.functions.sum("sum(value)")).first.get(0) match {
      case d: Double => d
      case l: Long => l.toDouble
    }
    val totSumSq: Double = sumsq.agg(org.apache.spark.sql.functions.sum("sum(valueSq)")).first.get(0) match {
      case d: Double => d
      case l: Long => l.toDouble
    }

    val totMean: Double = totSum / totN

    val dft: Double = totN - 1
    val dfb: Double = numCats - 1
    val dfw: Double = totN - numCats

    val joined: DataFrame = counts.as("a")
      .join(sums.as("b"), "cat")
      .join(sumsq.as("c"), "cat")
      .join(avgs.as("d"), "cat")
      .select("a.cat", "count", "sum(value)", "sum(valueSq)", "avg(value)")
    val finaldf: DataFrame = joined.withColumn("totMean", lit(totMean))

    val ssb_tmp: RDD[(String, Double)] = finaldf.rdd.map(x => (x(0).asInstanceOf[String], (((x(4) match {
      case d: Double => d
      case l: Long => l.toDouble
    }) - (x(5) match {
      case d: Double => d
      case l: Long => l.toDouble
    })) * ((x(4) match {
      case d: Double => d
      case l: Long => l.toDouble
    }) - (x(5) match {
      case d: Double => d
      case l: Long => l.toDouble
    }))) * (x(1) match {
      case d: Double => d
      case l: Long => l.toDouble
    })))
    import spark.implicits._
    val ssb = ssb_tmp.toDF.agg(org.apache.spark.sql.functions.sum("_2")).first.get(0) match {
      case d: Double => d
      case l: Long => l.toDouble
    }

    val ssw_tmp = grouped.sum("diffSq")
    val ssw = ssw_tmp.agg(org.apache.spark.sql.functions.sum("sum(diffSq)")).first.get(0) match {
      case d: Double => d
      case l: Long => l.toDouble
    }

    val sst = ssb + ssw

    val msb = ssb / dfb
    val msw = ssw / dfw
    val F = msb / msw

    val etaSq = ssb / sst
    val omegaSq = (ssb - ((numCats - 1) * msw)) / (sst + msw)

    ANOVAStats(dfb.toLong, dfw, F, etaSq, omegaSq)
  }
}
