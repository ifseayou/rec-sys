package com.isea.offline

import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix

/**
  * Movie数据集，数据集字段通过分割
  *
  * 151^                          电影的ID
  * Rob Roy (1995)^               电影的名称
  * In the highlands ....^        电影的描述
  * 139 minutes^                  电影的时长
  * August 26, 1997^              电影的发行日期
  * 1995^                         电影的拍摄日期
  * English ^                     电影的语言
  * Action|Drama|Romance|War ^    电影的类型
  * Liam Neeson|Jessica Lange...  电影的演员
  * Michael Caton-Jones           电影的导演
  *
  **/

case class Movie(val mid: Int, val name: String, val descri: String, val timelong: String, val issue: String,
                 val shoot: String, val language: String, val genres: String, val actors: String, val directors: String)

/**
  * Rating数据集，用户对于电影的评分数据集，用，分割
  *
  * 1,           用户的ID
  * 31,          电影的ID
  * 2.5,         用户对于电影的评分
  * 1260759144   用户对于电影评分的时间
  */
case class MovieRating(val uid: Int, val mid: Int, val score: Double, val timestamp: Int)

/**
  * MongoDB的连接配置
  *
  * @param uri MongoDB的连接
  * @param db  MongoDB要操作数据库
  */
case class MongoConfig(val uri: String, val db: String)

//推荐 mid : rating
case class Recommendation(rid: Int, r: Double)

// 用户的推荐
case class UserRecs(uid: Int, recs: Seq[Recommendation])

//电影的相似度
case class MovieRecs(mid: Int, recs: Seq[Recommendation])

object OfflineRecommender {

  // 保存原始评分和电影数据
  val MONGODB_RATING_COLLECTION = "Rating"
  val MONGODB_MOVIE_COLLECTION = "Movie"

  val USER_MAX_RECOMMENDATION = 20

  // 隐特征矩阵
  val USER_RECS = "UserRecs"
  val MOVIE_RECS = "MovieRecs"

  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://hadoop101:27017/recommender",
      "mongo.db" -> "reommender"
    )

    //创建一个SparkConf配置
    val sparkConf = new SparkConf().setAppName("OfflineRecommender").setMaster(config("spark.cores")).set("spark.executor.memory", "6G").set("spark.driver.memory", "2G")

    //基于SparkConf创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    //创建一个MongoDBConfig
    val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))

    import spark.implicits._

    //读取mongoDB中的评分数据
    val ratingRDD = spark
      .read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[MovieRating]
      .rdd
      .map(rating => (rating.uid, rating.mid, rating.score)).cache()

    //用户的数据集 RDD[Int]
    val userRDD = ratingRDD.map(_._1).distinct()

    //电影数据集 RDD[Int]
    val movieRDD = spark
      .read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_MOVIE_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Movie]
      .rdd
      .map(_.mid).cache()

    // 某个用户对某个电影打了多少分 => 构成一个训练集
    val trainData = ratingRDD.map(x => Rating(x._1, x._2, x._3))
    val (rank, iterations, lambda) = (70, 5, 0.1)
    //将训练集输入之后，训练ALS模型
    val model = ALS.train(trainData, rank, iterations, lambda)

    // ----------计算用户推荐矩阵------------

    //需要构造一个usersProducts  RDD[(Int,Int)]，这个movieRDD中没有评分
    val userMovies = userRDD.cartesian(movieRDD)

    val preRatings = model.predict(userMovies)

    val userRecs = preRatings  // 得到每一个用户对某一个电影的评分矩阵
      .filter(_.rating > 0)
      .map(rating => (rating.user, (rating.product, rating.rating))) // 某一个用户对mid的评分
      .groupByKey() // 将相同uid的聚合在一起
      .map {
        case (uid, recs) => // recs 是一个[(mid,rating),(),()...]
          UserRecs(uid, recs.toList.sortWith(_._2 > _._2).take(USER_MAX_RECOMMENDATION) // 按照打分进行排序,并只取x个
            .map(x => Recommendation(x._1, x._2)))
      }.toDF()


    userRecs.write
      .option("uri", mongoConfig.uri)
      .option("collection", USER_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()


    // -----------计算电影相似度矩阵-----------

    val movieFeatures = model.productFeatures.map { case (mid, freatures) => // productFeatures会形成关于product 和其隐特征向量
      (mid, new DoubleMatrix(freatures))
    }

    val movieRecs = movieFeatures.cartesian(movieFeatures) // ((Int,DoubleMatrix),(Int,DoubleMatrix))
      .filter { case (a, b) => a._1 != b._1 } // 过滤掉自己和自己相乘的
      .map { case (a, b) =>
        val simScore = this.consinSim(a._2, b._2) // 计算余弦相似度，实际是两个向量相乘
        (a._1, (b._1, simScore)) // 形成(mid,(mid,similarDegree))
      }.filter(_._2._2 > 0.6) // 将similarDegree的过滤出来
      .groupByKey()  // 按照mid聚合，形成(mid,[(mid,similarDegree),(mid,similarDegree),,,]
      .map { case (mid, items) =>
        MovieRecs(mid, items.toList.map(x => Recommendation(x._1, x._2)))
      }.toDF()


    movieRecs
      .write
      .option("uri", mongoConfig.uri)
      .option("collection", MOVIE_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    //关闭Spark
    spark.close()
  }

  //计算两个电影之间的余弦相似度
  def consinSim(movie1: DoubleMatrix, movie2: DoubleMatrix): Double = {
    movie1.dot(movie2) / (movie1.norm2() * movie2.norm2())
  }
}
