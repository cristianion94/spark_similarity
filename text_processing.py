from pyspark.ml.feature import CountVectorizer, IDF, StopWordsRemover, Tokenizer, Normalizer, HashingTF
from pyspark.ml.feature import MinHashLSH
import pyspark.sql.functions as F
from pyspark.ml.clustering import KMeans, PowerIterationClustering
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
from pyspark.mllib.evaluation import MulticlassMetrics
import string
from pyspark.sql.functions import isnan, when, count, col

from pyspark.sql import SparkSession


def get_spark_session():
	return SparkSession.builder.config("spark.driver.memory", "15g").appName("DocumentSimilarity").getOrCreate()


def create_dataframe(spark, path):
	translation_table = str.maketrans(string.punctuation + string.ascii_uppercase,
	                                  " " * len(string.punctuation) + string.ascii_lowercase)
	da = spark.sparkContext.wholeTextFiles(path)
	da = da.map(lambda id_text: (id_text[0], id_text[1].translate(translation_table)))
	df = spark.createDataFrame(da, ["id", "text"])
	return df


from pyspark.sql.functions import regexp_replace, trim, col, lower


def removePunctuation(column):
	return lower(trim(regexp_replace(column, '\\p{Punct}', ''))).alias('text')


def create_dataframe_from_csv(spark, path):
	translation_table = str.maketrans(string.punctuation + string.ascii_uppercase,
	                                  " " * len(string.punctuation) + string.ascii_lowercase)
	df = spark.read.option("header", "true").csv(path)
	df.printSchema()
	df = df.select(col("_c0").alias("id"), removePunctuation(col("text")))
	df = df.na.drop()
	return df


def show_rdd(rdd):
	output = rdd.collect()
	for (key, value) in output:
		print(key, value)


def show_dataframe(df):
	df.show(truncate=True)


def dataframe_transform(tokenizer, df, debug=False):
	wordsData = tokenizer.transform(df.dropna())
	if debug:
		show_dataframe(wordsData)
	return wordsData


def tf_idf(words_df, spark_count_vectorizer, spark_idf, debug=False):
	# cv_model = spark_count_vectorizer.fit(words_df)
	term_freqeuncy = spark_count_vectorizer.transform(words_df)

	idfModel = spark_idf.fit(term_freqeuncy)

	tfidf = idfModel.transform(term_freqeuncy)
	if debug:
		show_dataframe(term_freqeuncy)
		show_dataframe(tfidf.select("id", "features"))
	return tfidf


def spark_lsh(spark_lsh_param, tf_idf_df, debug=False):
	lsh_model = spark_lsh_param.fit(tf_idf_df)
	if debug:
		show_dataframe(lsh_model.transform(tf_idf_df))
	return lsh_model


def approximate_similarity_join(lsh_model, tf_idf_df, max_distance=0.9):
	# tf_idf_df.show()

	df_similarities = lsh_model.approxSimilarityJoin(tf_idf_df, tf_idf_df, max_distance, distCol="JaccardDistance")
	df_sim = df_similarities.select(col("datasetA.id").alias("src"), col("datasetB.id").alias("dst"),
	                                col("JaccardDistance"))
	# df_sim.write.save('/output', format='csv', mode='append', header='true')
	return df_sim


# df_similarities.select(col("datasetA.id").alias("idA"), col("datasetB.id").alias("idB"), col("JaccardDistance")).show()

def power_iteration(df_sim):
	df_columns = df_sim.select(col("src"), col("dst"), col("JaccardDistance").alias("similarity"))
	graph = df_columns.withColumn("similarity", 1 - df_columns.similarity).withColumn("src", df_columns.src.cast(
		"Int")).withColumn("dst", df_columns.dst.cast("Int"))
	pic = PowerIterationClustering(k=5, maxIter=20, initMode="random", weightCol="similarity")
	pic_df = pic.assignClusters(graph)

	return pic_df


def join_columns(df_sim, pic, cluster_table_df):
	kMeans_sim_joind = df_sim.join(cluster_table_df, df_sim.src == cluster_table_df.id, "inner").select(col("src"),
	                                                                                                    col("dst"), col(
			"JaccardDistance"), col("prediction").alias("predictionKMSrc"))
	print("First Join")
	kMeans_sim_joind = kMeans_sim_joind.join(cluster_table_df, kMeans_sim_joind.dst == cluster_table_df.id,
	                                         "inner").select(col("src"), col("dst"), col("JaccardDistance"),
	                                                         col("predictionKMSrc"),
	                                                         col("prediction").alias("predictionKMDst"))
	print("Second Join")

	kMeans_sim_joind = kMeans_sim_joind.withColumn("KMeans",
	                                               kMeans_sim_joind.predictionKMSrc == kMeans_sim_joind.predictionKMDst)

	kMeans_sim_joind = kMeans_sim_joind.join(pic, kMeans_sim_joind.src == pic.id, "inner").select(col("src"),
	                                                                                              col("dst"), col(
			"JaccardDistance"), col("KMeans"), col("cluster").alias("predictionPICSrc"))

	print("Third Join")
	kMeans_sim_joind = kMeans_sim_joind.join(pic, kMeans_sim_joind.dst == pic.id, "inner").select(col("src"),
	                                                                                              col("dst"), col(
			"JaccardDistance"), col("KMeans"), col("predictionPICSrc"), col("cluster").alias("predictionPICDst"))
	print("Fourth Join")

	kMeans_sim_joind = kMeans_sim_joind.withColumn("PIC",
	                                               kMeans_sim_joind.predictionPICSrc == kMeans_sim_joind.predictionPICDst)

	kMeans_sim_joind = kMeans_sim_joind.select(col("src"), col("dst"), col("JaccardDistance"), col("KMeans"),
	                                           col("PIC"))
	return kMeans_sim_joind


def confusion_matrix_KMeans(kMeans_sim_joind):
	prediction_int = kMeans_sim_joind.withColumn("KMeans", kMeans_sim_joind.KMeans.cast("Float"))

	prediction_int = prediction_int.withColumn("JaccardDistance",
	                                           when(prediction_int.JaccardDistance < 0.5, 1.0).otherwise(0.0))

	preds_and_label = prediction_int.select(["JaccardDistance", "KMeans"])

	metrics = MulticlassMetrics(preds_and_label.rdd.map(tuple))
	print(metrics.confusionMatrix().toArray())


def confusion_matrix_PIC(kMeans_sim_joind):
	prediction_int = kMeans_sim_joind.withColumn("PIC", kMeans_sim_joind.PIC.cast("Float"))

	prediction_int = prediction_int.withColumn("JaccardDistance",
	                                           when(prediction_int.JaccardDistance < 0.6, 1.0).otherwise(0.0))

	preds_and_label = prediction_int.select(["JaccardDistance", "PIC"])

	metrics = MulticlassMetrics(preds_and_label.rdd.map(tuple))
	print(metrics.confusionMatrix().toArray())


def main():
	spark = get_spark_session()
	# df = create_dataframe(spark, "documents/*")
	df = create_dataframe_from_csv(spark, "news.csv")

	tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
	remover = StopWordsRemover(stopWords=StopWordsRemover.loadDefaultStopWords("english"), inputCol="tokens",
	                           outputCol="words")
	cv = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=2048)
	idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)

	mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=50)

	words_df = dataframe_transform(tokenizer, df)
	# show_dataframe(words_df)
	words_df = dataframe_transform(remover, words_df)

	tf_idf_df = tf_idf(words_df, cv, idf)

	normalizer = Normalizer(inputCol="features", outputCol="normFeatures")
	l2NormData = normalizer.transform(tf_idf_df)

	kmeans = KMeans().setK(1000).setMaxIter(1000)
	km_model = kmeans.fit(l2NormData)

	clustersTable = km_model.transform(l2NormData)

	cluster_table_df = clustersTable.select(col("id"), col("prediction"))
	# cluster_table_df.write.save('/output_cluster', format='csv', mode='append', header='true')

	lsh_model = spark_lsh(mh, tf_idf_df)

	df_sim = approximate_similarity_join(lsh_model, tf_idf_df, max_distance=0.8)

	pic = power_iteration(df_sim)

	final_df = join_columns(df_sim, pic, cluster_table_df)

	# final_df.repartition(3000).write.save('/final_result_final', format='csv', mode='append', header='true')
	print("Confusion Matrix KMeans")
	confusion_matrix_KMeans(final_df)

	print("Confusion Matrix PIC")
	confusion_matrix_PIC(final_df)


if __name__ == "__main__":
	main()
