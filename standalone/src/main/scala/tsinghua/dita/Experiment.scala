package tsinghua.dita

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import tsinghua.dita.algorithms.{TrajectoryRangeAlgorithms, TrajectorySimilarityWithKNNAlgorithms, TrajectorySimilarityWithThresholdAlgorithms}
import tsinghua.dita.common.DITAConfigConstants
import tsinghua.dita.common.shape.{Point, Rectangle}
import tsinghua.dita.common.trajectory.{Trajectory, TrajectorySimilarity}
import tsinghua.dita.rdd.TrieRDD

import scala.collection.mutable.ArrayBuffer

object Experiment {

  private def getTrajectory(line: (String, Long)): Trajectory = {
    val points = line._1.split(";").map(_.split(","))
      .map(x => Point(x.map(_.toDouble)))
    Trajectory(points, line._2)
  }

  case class TrajectoryRecord(id: Long, traj: Array[Array[Double]])

  private def getQueryTrajectories(line: (String, Long)): TrajectoryRecord = {
    //println(line._2)
    val points = line._1.split(";").map(_.split(","))
      .map(x => x.map(_.toDouble))
    TrajectoryRecord(line._2, points)
  }

  private def getSimFunc(simF: String): TrajectorySimilarity = {
    simF match {
      case "DTW" => TrajectorySimilarity.DTWDistance
      case "F" => TrajectorySimilarity.FrechetDistance
      case "EDR" => TrajectorySimilarity.EDRDistance
      case "LCSS" => TrajectorySimilarity.LCSSDistance
      case _ =>  null
    }
  }

  var sc: SparkContext = null
  var eachQueryLoopTimes = 1 // -step
  var showResult = false

  def main(args: Array[String]): Unit = {
    var filePath = "file:///Users/tianwei/Projects/data/dita_porto_small.txt" // -p
    var threshold = 0.05 // -t
    var sRange = "31.35387,-117.18512,31.44107,-117.15221" //-s
    var kValue = 50 //-k
    var distanceFunction = "DTW" //-dis
    var queryId = 83 // -qid
    var queryStrin = "" // -qstr
    var queryCenter = "31.35387,-117.18512" // -cen
    var radius = 0.01 // -r
    var master ="local[*]" //-m
    var mrs = "8g"
    var lower = 6
    var upper = 1000000
    var queryPath = "file:///Users/tianwei/Projects/data/dita_porto_small.txt"
    var thresholds = Array(0.01, 0.02, 0.05, 0.1, 0.2, 0.4)
    var ks = Array(1, 2, 5, 10, 20, 50)
    var queryInfo = Map[String, Int]()
//    queryInfo += ("sim"->0)
//    queryInfo += ("simJoin"->0)
//    queryInfo += ("knnSim"->0)
    queryInfo += ("knnSimJoin"->0)

    if (args.length > 0) {
      if (args.length % 2 == 0) {
        var i = 0
        while (i < args.length) {
          args(i) match {
            case "-p" => filePath = args(i+1)
            case "-qp" => queryPath = args(i+1)
            case "-m" => master = args(i+1)
            case "-k" => ks = args(i+1).split(",").map(_.toInt)
            case "-df" => distanceFunction = args(i+1)
            case "-t" => thresholds = args(i+1).split(",").map(_.toDouble)
            case "-s" => sRange = args(i)
            case "-qid" => queryId = args(i+1).toInt
            case "-qstr" => queryStrin = args(i+1)
            case "-r" => radius = args(i+1).toDouble
            case "-cen" => queryCenter = args(i+1)
            case "-step" => eachQueryLoopTimes = args(i+1).toInt
            case "-l" => lower = args(i+1).toInt
            case "-u" => upper = args(i+1).toInt
            case "-q" => val qs = args(i+1).split(",").toList
              queryInfo = Map[String, Int]()
              qs.foreach(q=> {
                queryInfo += (q->0)
              })
            case "-show" => showResult = args(i+1).toBoolean
            case "-knnIterGT" => DITAConfigConstants.KNN_MAX_GLOBAL_ITERATION = args(i+1).toInt
            case "-knnIterLT" => DITAConfigConstants.KNN_MAX_LOCAL_ITERATION = args(i+1).toInt
            case "-pivotC" => DITAConfigConstants.LOCAL_INDEXED_PIVOT_COUNT = args(i+1).toInt
            case "-knnLT" => DITAConfigConstants.kNN_LOCAL_THRESHOLD = args(i+1).toDouble
            case "-sampleRate" => DITAConfigConstants.BALANCING_SAMPLE_RATE = args(i+1).toDouble
            case "-kNNsampleRate" => DITAConfigConstants.KNN_MAX_SAMPLING_RATE = args(i+1).toDouble
            case "-mrs" => mrs = args(i+1)
            case e => {println(s"wrong parameter $e ..."); return}
          }
          i+=2
        }
      } else {
        println("wrong parameter length ...")
        return
      }
    }

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)



    //    val spark = SparkSession
    //      .builder()
    //      //.master(master)
    //      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    //      .getOrCreate()

    val conf = new SparkConf().setAppName("DITA")
      .set("spark.locality.wait", "0")
      .set("spark.driver.maxResultSize", mrs)
    if (!master.equals("dla")) {
      conf.setMaster(master)
    }
    sc = new SparkContext(conf)

    println(filePath, queryPath, distanceFunction)

    val trajs = //spark.sparkContext
    //.textFile("hdfs://master:9000"+filePath)
      sc.textFile(filePath)
        .zipWithIndex().map(getTrajectory)
        .filter(_.points.length >= lower)
        .filter(_.points.length <= upper)
    println(s"Trajectory count: ${trajs.count()}")

    val queries = //spark.sparkContext
    //.textFile("hdfs://master:9000"+queryPath)
      sc.textFile(queryPath)
        .zipWithIndex().map(getQueryTrajectories).collect()
    println(s"Query Trajectory count: ${queries.length}")
    //trajs.

    val rdd1 = new TrieRDD(trajs)
    val rdd2 = if (queryInfo.contains("simJoin") || queryInfo.contains("knnSimJoin")) new TrieRDD(trajs) else null

    // threshold-based search
    if (queryInfo.contains("sim")) {
      println(s"<DITA> Sim Trajectory Query")
      thresholds.foreach(t => {
        val times = Array.ofDim[Double](queries.length)
        for (i <- queries.indices) {
          val item = queries(i)
          times(i) = simProcessing(item, rdd1, distanceFunction, t)
        }
        println(s"====>Avg Time, $t, ${times.sum / queries.length.toDouble}")


//        println(s"<DITA> Sim Trajectory Query With Threshold $t")
//        var allSum = 0L
//        qqq.foreach(q=>{
//          println(s"<DITA> trajectory ${q.id}")
//          var sumOne = 0L
//          for (i <- 0 until executionTime) {
//            //trajs.filter(t => t.= queryId).take(1).head.traj//trajs.take(1).head
//            val qt = Trajectory(q.traj.map(Point(_)))
//            val thresholdSearch = TrajectorySimilarityWithThresholdAlgorithms.DistributedSearch
//            val start0 = System.currentTimeMillis()
//            val thresholdSearchAnswer = thresholdSearch.search(sc, qt, rdd1, getSimFunc(distanceFunction), t)
//            //thresholdSearchAnswer.
//            println(s"<DITA> Threshold search answer count: ${thresholdSearchAnswer.count()}")
//            sumOne += (System.currentTimeMillis() - start0)
//            //println(s"<DITA> One Loop Time: ${System.currentTimeMillis() - start0} ms")
//          }
//          println(s"Avg Time of ${q.id}: ${sumOne/executionTime.toDouble}")
//          allSum += sumOne
//        })
//        println(s"Avg Time (All) of t=$t: ${allSum/(executionTime*qqq.length).toDouble} ms")
      })
    }

    // knn search
    if (queryInfo.contains("knnSim")) {
      println(s"<DITA> KNN Trajectory Query")
      ks.foreach(k => {
        val times = Array.ofDim[Double](queries.length)
        for (i <- queries.indices) {
          val item = queries(i)
          times(i) = kNNProcessing(item, rdd1, distanceFunction, k)
        }
        println(s"====>Avg Time, $k, ${times.sum / queries.length.toDouble}")


//        println(s"<DITA> Sim Trajectory Query With K $k")
//        var allSum = 0L
//        qqq.foreach(q=>{
//          println(s"<DITA> trajectory ${q.id}")
//          var sumOne = 0L
//          for (i <- 0 until executionTime) {
//            val qt = Trajectory(q.traj.map(Point(_)))
//            val knnSearch = TrajectorySimilarityWithKNNAlgorithms.DistributedSearch
//            val start0 = System.currentTimeMillis()
//            val knnSearchAnswer = knnSearch.search(sc, qt, rdd1, getSimFunc(distanceFunction), k)
//            println(s"KNN search answer count: ${knnSearchAnswer.count()}")
//            sumOne += (System.currentTimeMillis() - start0)
//          }
//          println(s"Avg Time of ${q.id}: ${sumOne/executionTime.toDouble}")
//          allSum += sumOne
//        })
//        println(s"Avg Time (all) of k=$k: ${allSum/(executionTime*ks.length).toDouble} ms")
      })
    }

    if (queryInfo.contains("simJoin")) {
      println(s"<DITA> Threshold-based Trajectory Similarity Join")
      thresholds.foreach(t => {
        var times = 0.0
        times = tJoinProcessing(rdd1, rdd2, distanceFunction, t)
        println(s"====>Avg Time, $t, $times")


//        var sumOne = 0L
//        for (i <-0 until executionTime) {
//          val thresholdJoin = TrajectorySimilarityWithThresholdAlgorithms.FineGrainedDistributedJoin
//          val start0 = System.currentTimeMillis()
//          val thresholdJoinAnswer = thresholdJoin.join(sc, rdd1, rdd2,  getSimFunc(distanceFunction), t)
//          println(s"Threshold join answer count: ${thresholdJoinAnswer.count()}")
//          sumOne += (System.currentTimeMillis()-start0)
//        }
//        println(s"Avg Time of $t :${sumOne/executionTime.toDouble}")
      })
    }

    if (queryInfo.contains("knnSimJoin")) {
      ks.foreach(k => {
        var times = 0.0
        times = kJoinProcessing(rdd1, rdd2, distanceFunction, k)
        println(s"====>Avg Time, $k, $times")


//        var sumOne = 0L
//        for (i <-0 until executionTime) {
//          val knnJoin = TrajectorySimilarityWithKNNAlgorithms.DistributedJoin
//          val start0 = System.currentTimeMillis()
//          val knnJoinAnswer = knnJoin.join(sc, rdd1, rdd2, getSimFunc(distanceFunction), k)
//          println(s"KNN join answer count: ${knnJoinAnswer.count()}")
//          sumOne += (System.currentTimeMillis()-start0)
//        }
//        println(s"Avg Time of $k :${sumOne/executionTime.toDouble}")
      })
    }

  }

  def simProcessing(q: TrajectoryRecord, rdd1: TrieRDD, simFunc: String, t: Double): Double = {
    val qt = Trajectory(q.traj.map(Point(_)), q.id)
    val thresholdSearch = TrajectorySimilarityWithThresholdAlgorithms.DistributedSearch
    var res: Array[(Trajectory, Double)] = null
    val times = Array.ofDim[Long](eachQueryLoopTimes)
    for (i <- 0 until eachQueryLoopTimes) {
      val t0 = System.currentTimeMillis()
      res = thresholdSearch.search(sc, qt, rdd1, getSimFunc(simFunc), t).collect()
      val t1 = System.currentTimeMillis()
      times(i) = t1 - t0
    }

    var str = s"${q.id},${qt.points.length},${res.length},${times.sum / eachQueryLoopTimes.toDouble},${times.min},${times.max},${times.sum}, "
    println(str + times.mkString(","))
    times.sum / eachQueryLoopTimes.toDouble
  }

  def kNNProcessing(q: TrajectoryRecord, rdd1: TrieRDD, simFunc: String, k: Int): Double = {
    val qt = Trajectory(q.traj.map(Point(_)), q.id)
    val knnSearch = TrajectorySimilarityWithKNNAlgorithms.DistributedSearch
    var res: Array[(Trajectory, Double)] = null
    val times = Array.ofDim[Long](eachQueryLoopTimes)
    for (i <- 0 until eachQueryLoopTimes) {
      val t0 = System.currentTimeMillis()
      res = knnSearch.search(sc, qt, rdd1, getSimFunc(simFunc), k).collect()
      val t1 = System.currentTimeMillis()
      times(i) = t1 - t0
    }

    if (showResult) {
      res.foreach(item => {
        println(qt.tid, item._1, item._2)
      })
    }

    var str = s"${q.id},${qt.points.length},${res.length},${times.sum / eachQueryLoopTimes.toDouble},${times.min},${times.max},${times.sum}, "
    println(str + times.mkString(","))
    times.sum / eachQueryLoopTimes.toDouble
  }

  def tJoinProcessing(rdd1: TrieRDD, rdd2: TrieRDD, simFunc: String, t: Double): Double = {
    val thresholdJoin = TrajectorySimilarityWithThresholdAlgorithms.FineGrainedDistributedJoin
    //var res: Array[(Trajectory, Trajectory, Double)] = null
    var res = 0L
    val times = Array.ofDim[Long](eachQueryLoopTimes)
    for (i <- 0 until eachQueryLoopTimes) {
      val t0 = System.currentTimeMillis()
      res = thresholdJoin.join(sc, rdd1, rdd2,  getSimFunc(simFunc), t).count()
      val t1 = System.currentTimeMillis()
      times(i) = t1 - t0
    }

    var str = s"${res},${times.sum / eachQueryLoopTimes.toDouble},${times.min},${times.max},${times.sum}, "
    println(str + times.mkString(","))
    times.sum / eachQueryLoopTimes.toDouble
  }

  def kJoinProcessing(rdd1: TrieRDD, rdd2: TrieRDD, simFunc: String, k: Int): Double = {
    val knnJoin = TrajectorySimilarityWithKNNAlgorithms.DistributedJoin
    //var res: Array[(Trajectory, Trajectory, Double)] = null
    var res = 0L
    val times = Array.ofDim[Long](eachQueryLoopTimes)
    for (i <- 0 until eachQueryLoopTimes) {
      val t0 = System.currentTimeMillis()
      res = knnJoin.join(sc, rdd1, rdd2, getSimFunc(simFunc), k).count()
      val t1 = System.currentTimeMillis()
      times(i) = t1 - t0
    }

    var str = s"${res},${times.sum / eachQueryLoopTimes.toDouble},${times.min},${times.max},${times.sum}, "
    println(str + times.mkString(","))
    times.sum / eachQueryLoopTimes.toDouble
  }


//  def main(args: Array[String]): Unit = {
//    var filePath = "/ais_25/trajectory.txt" // -p
//    var threshold = 0.05 // -t
//    var sRange = "31.35387,-117.18512,31.44107,-117.15221" //-s
//    var kValue = 50 //-k
//    var distanceFunction = "DTW" //-dis
//    var queryId = 83 // -qid
//    var queryStrin = "" // -qstr
//    var queryCenter = "31.35387,-117.18512" // -cen
//    var radius = 0.01 // -r
//    var master ="local[*]" //-m
//    var executionTime = 1 // -step
//    var lower = 6
//    var upper = 1000000
//    var queryInfo = Map[String, Int]()
//    var queryPath = ""
//    queryInfo += ("sim"->0)
//    queryInfo += ("simJoin"->0)
//    queryInfo += ("knnSim"->0)
//    var mrs = "4g"
//    var showResult = 0
//    if (args.length > 0) {
//      if (args.length % 2 == 0) {
//        var i = 0
//        while (i < args.length) {
//          args(i) match {
//            case "-p" => filePath = args(i+1)
//            case "-qp" => queryPath = args(i+1)
//            case "-m" => master = args(i+1)
//            case "-k" => kValue = args(i+1).toInt
//            case "-df" => distanceFunction = args(i+1)
//            case "-t" => threshold = args(i+1).toDouble
//            case "-s" => sRange = args(i)
//            case "-qid" => queryId = args(i+1).toInt
//            case "-qstr" => queryStrin = args(i+1)
//            case "-r" => radius = args(i+1).toDouble
//            case "-cen" => queryCenter = args(i+1)
//            case "-step" => executionTime = args(i+1).toInt
//            case "-l" => lower = args(i+1).toInt
//            case "-u" => upper = args(i+1).toInt
//            case "-q" => val qs = args(i+1).split(",").toList
//              queryInfo = Map[String, Int]()
//              qs.foreach(q=> {
//                queryInfo += (q->0)
//              })
//            case "-show" => showResult = args(i+1).toInt
//            case "-knnIterGT" => DITAConfigConstants.KNN_MAX_GLOBAL_ITERATION = args(i+1).toInt
//            case "-knnIterLT" => DITAConfigConstants.KNN_MAX_LOCAL_ITERATION = args(i+1).toInt
//            case "-pivotC" => DITAConfigConstants.LOCAL_INDEXED_PIVOT_COUNT = args(i+1).toInt
//            case "-knnLT" => DITAConfigConstants.kNN_LOCAL_THRESHOLD = args(i+1).toDouble
//            case "-mrs" => mrs = args(i+1)
//            case e => {println(s"wrong parameter $e ..."); return}
//          }
//          i+=2
//        }
//      } else {
//        println("wrong parameter length ...")
//        return
//      }
//    }
//
////    Logger.getLogger("org").setLevel(Level.WARN)
////    Logger.getLogger("akka").setLevel(Level.WARN)
//
//    val thresholds = Array(0.05,0.1,0.2,0.3,0.4,0.5)
//    val ks = Array(1,10,30,50,70,100)
//
//    val spark = SparkSession
//      .builder()
//      .master(master)
//      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//      .getOrCreate()
//
////    val conf = new SparkConf().setAppName("DITA")
////      .set("spark.locality.wait", "0")
////      .set("spark.driver.maxResultSize", mrs)
////    if (!master.equals("dla")) {
////      conf.setMaster(master)
////    }
////    val sc = new SparkContext(conf)
//
//    val trajs = spark.sparkContext
//      .textFile("hdfs://master:9000"+filePath)
//      //sc.textFile(filePath)
//      .zipWithIndex().map(getTrajectory)
//      .filter(_.points.length >= lower)
//      .filter(_.points.length <= upper)
//    println(s"Trajectory count: ${trajs.count()}")
//
//    val qTs = spark.sparkContext
//      .textFile("hdfs://master:9000"+queryPath)
//      //sc.textFile(queryPath)
//      .zipWithIndex().map(getQueryTrajectories)
//      .filter(_.traj.length >= lower)
//      .filter(_.traj.length <= upper)
//    println(s"Query Trajectory count: ${trajs.count()}")
//    //trajs.
//
//    val rdd1 = new TrieRDD(trajs)
//    val rdd2 = if (queryInfo.contains("simJoin") || queryInfo.contains("knnSimJoin")) new TrieRDD(trajs) else null
//
//    val qqq = qTs.collect()
//    // threshold-based search
//    if (queryInfo.contains("sim")) {
//      println(s"<DITA> Sim Trajectory Query")
//      val start = System.currentTimeMillis()
//      //thresholds.foreach(t => {
//        println(s"<DITA> Sim Trajectory Query With Threshold $threshold")
//        qqq.foreach(q=>{
//          println(s"<DITA> trajectory ${q.id}")
//          for (i <- 0 until executionTime) {
//            //trajs.filter(t => t.= queryId).take(1).head.traj//trajs.take(1).head
//
//            val qt = Trajectory(q.traj.map(Point(_)))
//            val thresholdSearch = TrajectorySimilarityWithThresholdAlgorithms.DistributedSearch
//            val start0 = System.currentTimeMillis()
//            val thresholdSearchAnswer = thresholdSearch.search(spark.sparkContext, qt, rdd1, getSimFunc(distanceFunction), threshold)
//            //thresholdSearchAnswer.
//            println(s"<DITA> Threshold search answer count: ${thresholdSearchAnswer.count()}")
//            println(s"<DITA> One Loop Time: ${System.currentTimeMillis() - start0} ms")
//          }
//        })
//        println(s"Avg Time: ${(System.currentTimeMillis()-start)/executionTime.toDouble} ms")
//      //})
//
//    }
//
//    // knn search
//    if (queryInfo.contains("knnSim")) {
//      println(s"<DITA> KNN Trajectory Query")
//      val start = System.currentTimeMillis()
//      //ks.foreach(k => {
//        println(s"<DITA> Sim Trajectory Query With Threshold $kValue")
//        qqq.foreach(q=>{
//          println(s"<DITA> trajectory ${q.id}")
//          for (i <- 0 until executionTime) {
//            val qt = Trajectory(q.traj.map(Point(_)))
//            val knnSearch = TrajectorySimilarityWithKNNAlgorithms.DistributedSearch
//            val start0 = System.currentTimeMillis()
//            val knnSearchAnswer = knnSearch.search(spark.sparkContext, qt, rdd1, getSimFunc(distanceFunction), kValue)
//            println(s"KNN search answer count: ${knnSearchAnswer.count()}")
//            println(s"<DITA> One Loop Time: ${System.currentTimeMillis() - start0} ms")
//          }
//        })
//      //})
//      println(s"Avg Time: ${(System.currentTimeMillis()-start)/executionTime.toDouble} ms")
//    }
////    val knnSearch = TrajectorySimilarityWithKNNAlgorithms.DistributedSearch
////    val knnSearchAnswer = knnSearch.search(spark.sparkContext, queryTrajectory, rdd1, TrajectorySimilarity.DTWDistance, 100)
////    println(s"KNN search answer count: ${knnSearchAnswer.count()}")
//
//    // threshold-based join
////    val thresholdJoin = TrajectorySimilarityWithThresholdAlgorithms.FineGrainedDistributedJoin
////    val thresholdJoinAnswer = thresholdJoin.join(spark.sparkContext, rdd1, rdd2, TrajectorySimilarity.DTWDistance, 0.005)
////    println(s"Threshold join answer count: ${thresholdJoinAnswer.count()}")
////
////    // knn join
////    val knnJoin = TrajectorySimilarityWithKNNAlgorithms.DistributedJoin
////    val knnJoinAnswer = knnJoin.join(spark.sparkContext, rdd1, rdd2, TrajectorySimilarity.DTWDistance, 100)
////    println(s"KNN join answer count: ${knnJoinAnswer.count()}")
//
////    // mbr range search
////    val search = TrajectoryRangeAlgorithms.DistributedSearch
////    val mbr = Rectangle(Point(Array(39.8, 116.2)), Point(Array(40.0, 116.4)))
////    val mbrAnswer = search.search(spark.sparkContext, mbr, rdd1, 0.0)
////    println(s"MBR range search count: ${mbrAnswer.count()}")
////
////    // circle range search
////    val center = Point(Array(39.9, 116.3))
////    val radius = 0.1
////    val circleAnswer = search.search(spark.sparkContext, center, rdd1, radius)
////    println(s"Circle range search count: ${circleAnswer.count()}")
//  }
}
