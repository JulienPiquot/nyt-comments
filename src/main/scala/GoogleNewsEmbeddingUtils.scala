import java.io.{BufferedInputStream, DataInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import org.apache.spark.mllib.feature.Word2VecModel

object GoogleNewsEmbeddingUtils {

  def loadBin(file: String): Word2VecModel = {
    def readUntil(inputStream: DataInputStream, term: Char, maxLength: Int = 1024 * 8): String = {
      var char: Char = inputStream.readByte().toChar
      val str = new StringBuilder
      while (!char.equals(term)) {
        str.append(char)
        assert(str.size < maxLength)
        char = inputStream.readByte().toChar
      }
      str.toString
    }

    val inputStream: DataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))
    try {
      val header = readUntil(inputStream, '\n')
      val (records, dimensions) = header.split(" ") match {
        case Array(rec, dim) => (rec.toInt, dim.toInt)
      }
      println("number of records : " + records)
      println("number of dimensions : " + dimensions)
      new Word2VecModel((0 until records).toArray.map(recordIndex => {
        print("read " + recordIndex + " records\r")
        readUntil(inputStream, ' ') -> (0 until dimensions).map(dimensionIndex => {
          java.lang.Float.intBitsToFloat(java.lang.Integer.reverseBytes(inputStream.readInt()))
        }).toArray
      }).toMap)
    } finally {
      inputStream.close()
    }
  }
}
