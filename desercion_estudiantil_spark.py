from pathlib import Path

import time
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


HDFS_INPUT_PATH = "hdfs:///user/root/input/tbl_desercion_estudiantil_primer_anio.csv"
LOCAL_INPUT_PATH = "tbl_desercion_estudiantil_primer_anio.csv"


def create_spark_session() -> SparkSession:
	return SparkSession.builder.appName("DesercionEstudiantil").getOrCreate()


def resolve_input_path(spark: SparkSession) -> str:
	try:
		hadoop_fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
		hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(HDFS_INPUT_PATH)
		if hadoop_fs.exists(hdfs_path):
			return HDFS_INPUT_PATH
	except Exception:
		pass

	if Path(LOCAL_INPUT_PATH).exists():
		return LOCAL_INPUT_PATH

	return HDFS_INPUT_PATH


def load_dataset(spark: SparkSession, input_path: str):
	return (
		spark.read.option("header", True)
		.option("sep", ";")
		.option("inferSchema", True)
		.option("nullValue", "NULL")
		.csv(input_path)
	)


def prepare_features(df):
	categorical_cols = ["Carrera", "CicloIngreso"]

	df = df.fillna(0).withColumn("Deserto", F.col("Deserto").cast("int"))

	indexers = [
		StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
		for col in categorical_cols
	]

	indexer_pipeline = Pipeline(stages=indexers)
	indexer_model = indexer_pipeline.fit(df)
	df_model = indexer_model.transform(df)

	indexed_cols = [f"{col}_idx" for col in categorical_cols]
	base_feature_cols = [
		col
		for col in df_model.columns
		if col not in ["Deserto", *categorical_cols, *indexed_cols]
	]
	feature_cols = base_feature_cols + indexed_cols

	assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
	data = assembler.transform(df_model).select(F.col("Deserto").alias("label"), "features")

	return data


def build_metrics(model_name, predictions, train_time, pred_time):
	accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
	precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
	recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
	f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
	auc_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

	return {
		"Modelo": model_name,
		"Accuracy": round(accuracy_evaluator.evaluate(predictions), 4),
		"Precision": round(precision_evaluator.evaluate(predictions), 4),
		"Recall": round(recall_evaluator.evaluate(predictions), 4),
		"F1": round(f1_evaluator.evaluate(predictions), 4),
		"ROC_AUC": round(auc_evaluator.evaluate(predictions), 4),
		"Tiempo_entrenamiento_s": round(train_time, 4),
		"Tiempo_prediccion_s": round(pred_time, 4),
	}


def main():
	spark = create_spark_session()
	input_path = resolve_input_path(spark)

	print(f"Leyendo dataset desde: {input_path}")
	df = load_dataset(spark, input_path)

	output_path = "/kaggle/working/data/desercion_parquet"
	df.write.mode("overwrite").parquet(output_path)
	df = spark.read.parquet(output_path)

	print("Esquema del dataset")
	df.printSchema()

	print("Valores nulos por columna")
	missing_counts = df.select([F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns])
	missing_counts.show(truncate=False)

	print("Distribución de la variable objetivo")
	df.groupBy("Deserto").count().orderBy("Deserto").show()

	print("Preparando features")
	data = prepare_features(df)
	data.show(5, truncate=False)

	print("Dividiendo en train/test")
	train_df, test_df = data.randomSplit([0.7, 0.3], seed=42)

	train_df.cache()
	test_df.cache()

	print("Cantidad de registros de entrenamiento:", train_df.count())
	print("Cantidad de registros de prueba:", test_df.count())

	# Entrenamiento de modelos
	lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)
	rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, seed=42, maxBins=256)

	start_lr = time.perf_counter()
	lr_model = lr.fit(train_df)
	lr_train_time = time.perf_counter() - start_lr

	start_rf = time.perf_counter()
	rf_model = rf.fit(train_df)
	rf_train_time = time.perf_counter() - start_rf

	start_lr_pred = time.perf_counter()
	lr_predictions = lr_model.transform(test_df)
	lr_predictions.count()
	lr_pred_time = time.perf_counter() - start_lr_pred

	start_rf_pred = time.perf_counter()
	rf_predictions = rf_model.transform(test_df)
	rf_predictions.count()
	rf_pred_time = time.perf_counter() - start_rf_pred

	comparative_results = [
		build_metrics("Logistic Regression", lr_predictions, lr_train_time, lr_pred_time),
		build_metrics("Random Forest", rf_predictions, rf_train_time, rf_pred_time),
	]

	comparative_df = spark.createDataFrame(comparative_results)
	comparative_df.show(truncate=False)

	best_by_f1 = max(comparative_results, key=lambda row: (row["F1"], row["ROC_AUC"]))
	fastest_train = min(comparative_results, key=lambda row: row["Tiempo_entrenamiento_s"] )
	fastest_pred = min(comparative_results, key=lambda row: row["Tiempo_prediccion_s"] )

	print("Metricas usadas para comparar el modelo:")
	print("- Accuracy")
	print("- Precision ponderada")
	print("- Recall ponderado")
	print("- F1")
	print("- ROC AUC")
	print("- Tiempo de entrenamiento")
	print("- Tiempo de prediccion")
	print(f"Mejor opcion segun F1 y ROC AUC: {best_by_f1['Modelo']}")
	print(f"Modelo mas rapido de entrenar: {fastest_train['Modelo']}")
	print(f"Modelo mas rapido de predecir: {fastest_pred['Modelo']}")

	spark.stop()


if __name__ == "__main__":
	main()
