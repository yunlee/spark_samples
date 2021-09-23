import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

val df = spark.read.format("csv").option("inferSchema", "true").option("header", "true").load("/adhoc/analysis/yl/tosimuPub/2020-06-16-11/")

val df2 = df.select("verified", "sid", "os", "browser", "dc", "adsize")



val features_indexed = Array("sid", "os", "browser", "dc")

val inds = features_indexed.map { colName =>
   new StringIndexer()
    .setInputCol(colName)
    .setOutputCol(colName + "I")
    .fit(df2)          
}

val pipeline2 = new Pipeline().setStages(inds).fit(df2)
val df3 = pipeline2.transform(df2).select("verified","sidI","osI", "browserI", "dcI", "adsize")

val features_indexed2 = Array("sidI","osI", "browserI", "dcI","adsize")

val assembler = new VectorAssembler().setInputCols(features_indexed2).setOutputCol("features")

val trainingData = assembler.transform(df3).select("verified", "features")

val labelIndexer = new StringIndexer().setInputCol("verified").setOutputCol("indexedLabel").fit(trainingData)
// Automatically identify categorical features, and index them.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(100000).fit(trainingData) // features with > 4 distinct values are treated as continuous.

val categoricalFeatures: Set[Int] = featureIndexer.categoryMaps.keys.toSet

println(s"Chose ${categoricalFeatures.size} " + s"categorical features: ${categoricalFeatures.mkString(", ")}")
// Split the data into training and test sets (30% held out for testing).
//val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a DecisionTree model.
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Chain indexers and tree in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]

println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
println(treeModel.featureImportances)

val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)
