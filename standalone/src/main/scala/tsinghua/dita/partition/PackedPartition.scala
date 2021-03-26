/*
 *  Copyright 2017 by DITA Project
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package tsinghua.dita.partition

import tsinghua.dita.common.DITAConfigConstants
import tsinghua.dita.index.LocalIndex

import scala.util.Random

case class PackedPartition(id: Int, data: Array[_ <: Any], indexes: Array[LocalIndex]) {

  def getSample(sampleRate: Double): List[_ <: Any] = {
    val sampleSize = math.max((data.length * sampleRate).toInt,
      DITAConfigConstants.BALANCING_MIN_SAMPLE_SIZE)
    //println(s"sampleSize: $sampleSize, length: ${data.length}, isEqual: ${sampleSize == data.length}")
    Random.shuffle(data.toList).take(sampleSize)
  }
}
