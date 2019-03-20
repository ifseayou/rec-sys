package com.isea.loader

import scala.collection.mutable

object Test {

  def main(args: Array[String]): Unit = {
    val str = "aaaaabbbbcccdddmm"

    val map = new mutable.HashMap[Char,Int]()

    for (i <- str){
      if (map.contains(i)){
        map(i)=map(i)+1
      }else{
        map+=(i->1)
      }
    }
    print(map)
    print(map.toSeq.sortWith(_._2>_._2).toList(2))
  }
}
