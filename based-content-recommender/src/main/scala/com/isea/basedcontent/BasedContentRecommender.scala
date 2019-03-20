package com.isea.basedcontent

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix


/**
  * MongoDB的连接配置
  *
  * @param uri MongoDB的连接
  * @param db  MongoDB要操作数据库
  */
case class MongoConfig(val uri: String, val db: String)

case class Movie(val mid: Int, val name: String, val descri: String, val timelong: String, val issue: String,
                 val shoot: String, val language: String, val genres: String, val actors: String, val directors: String)

//推荐
case class Recommendation(rid: Int, r: Double)

// 用户的推荐
case class UserRecs(uid: Int, recs: Seq[Recommendation])

//电影的相似度
case class MovieRecs(mid: Int, recs: Seq[Recommendation])


object BasedContentRecommender {

  val MONGODB_MOVIE_COLLECTION = "Movie"
  val MOVIE_RECS = "ContentBasedMovieRecs"

  def consinSim(movie1: DoubleMatrix, movie2: DoubleMatrix): Double = {
    movie1.dot(movie2) / (movie1.norm2() * movie2.norm2())
  }

  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://hadoop101:27017/recommender",
      "mongo.db" -> "reommender"
    )

    //创建一个SparkConf配置
    val sparkConf = new SparkConf().setAppName("ContentBasedRecommender").setMaster(config("spark.cores")).set("spark.executor.memory", "6G").set("spark.driver.memory", "2G")

    //基于SparkConf创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    //创建一个MongoDBConfig
    val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))

    import spark.implicits._


    //电影数据集 RDD[Int]
    val movieRDD = spark
      .read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_MOVIE_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Movie]
      .rdd
      .map(x => (x.mid, x.name, x.genres.map(c => if (c == '|') ' ' else c)))


    val movieSeq = movieRDD.collect() // 数据格式变为：mid,movieName,genres with ' ' not |

    val tagsData = spark.createDataFrame(movieSeq).toDF("mid", "name", "genres") // 转为结构化数据

    // 实例化一个分词器，默认按空格分
    val tokenizer = new Tokenizer().setInputCol("genres").setOutputCol("words")

    // 用分词器做转换，生成列“words”，返回一个dataframe，增加一列words
    val wordsData = tokenizer.transform(tagsData) // 此时，由原来的三列变为四列，是一个list，第四列的类型之间按照 , 分隔

    // HashingTF是一个工具，可以把一个词语序列，转换成词频(初始特征)；words是输入，输出是原始特征，特征的个数是20
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

    // 用 HashingTF 做处理，返回dataframe，此时最后一列是一个稀疏矩阵的表示
    val featurizedData = hashingTF.transform(wordsData)
    // 此时的数据格式变为：五列，最后一列是rowFeatures(20,[哪些位置不为0],[不为0的位置的元素是什么（tf词频）])


    // IDF 也是一个工具，用于计算文档的IDF
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

    // 将词频数据传入，得到idf模型（统计文档）
    val idfModel = idf.fit(featurizedData)

    // 模型对原始数据做处理，计算出idf后，用tf-idf得到新的特征矩阵
    val rescaledData = idfModel.transform(featurizedData)
    // 新增一列的名字叫做features，(20,[哪些位置不为0],[不为0的位置的元素是什么（tf-idf值）])

    val movieFeatures = rescaledData.map { // 将mid变成整型， 把features变成稀疏矩阵，转为数组
      case row => (row.getAs[Int]("mid"), row.getAs[SparseVector]("features").toArray)
    }
      .rdd  // 转为rdd
      .map(x => {
        (x._1, new DoubleMatrix(x._2))  // 将数组转为向量
      })

    /** 得到的矩阵如下：
      *        f1  f2  f3  f4
      * mid1   -   -   -   -
      * mid2   -   -   -   -
      * mid3   -   -   -   -
      * 由tf-idf的稀疏矩阵得到最后的电影相似度矩阵
      */

    // 计算电影相似度矩阵
    val movieRecs = movieFeatures.cartesian(movieFeatures)
      .filter{case (a,b) => a._1 != b._1}
      .map{case (a,b) =>
        val simScore = this.consinSim(a._2,b._2)
        (a._1,(b._1,simScore))
      }.filter(_._2._2 > 0.6)
      .groupByKey()
      .map{case (mid,items) =>
        MovieRecs(mid,items.toList.map(x => Recommendation(x._1,x._2)))
      }.toDF()

    // movieRecs.show(5)

    movieRecs
      .write
      .option("uri", mongoConfig.uri)
      .option("collection",MOVIE_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    //    rescaledData.select("features", "mid").take(3).foreach(println)

    //关闭Spark
    spark.close()
  }
}
