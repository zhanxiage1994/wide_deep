from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import tempfile
import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
def csv_head():
	part1 = ["C"+str(i) for i in range(1,6)]
	part2 = ["C"+str(i) for i in range(6, 6+280)]
	continuous_columns = ["I"+str(i) for i in range(1,16)]
	total_sales = ["total_sales","total_sales_flag"]
	categorical_columns = part1 #+ part2
	feature_columns = part1 + continuous_columns + part2
	CSV_COLUMNS = feature_columns + total_sales
	return categorical_columns, continuous_columns, total_sales[1], feature_columns, CSV_COLUMNS

categorical_columns, continuous_columns, label_column, feature_columns, CSV_COLUMNS = csv_head()

def input_fn(data_file, num_epochs = 1, shuffle = False, is_training = True):
	if is_training:
		df_data = pd.read_csv( tf.gfile.Open(data_file), names=CSV_COLUMNS, skipinitialspace=True,
				engine="python", index_col = False)
		df_data = df_data.dropna(how="any", axis=0)
		df_data[categorical_columns] = df_data[categorical_columns].values.astype(str)

		labels = df_data["total_sales_flag"]#.apply(lambda x: ">50K" in x).astype(int)
		return tf.estimator.inputs.pandas_input_fn(x=df_data, y=labels, num_epochs=num_epochs,
				shuffle=shuffle, num_threads=1)
	else:
		df_data = pd.read_csv( tf.gfile.Open(data_file), names=feature_columns, skipinitialspace=True,
				engine="python", index_col = False)
		df_data = df_data.dropna(how="any", axis=0)
		df_data[categorical_columns] = df_data[categorical_columns].values.astype(str)


		return tf.estimator.inputs.pandas_input_fn(x=df_data, shuffle=shuffle)

def process_feature():
	wide_columns = []
	for name in categorical_columns:
		wide_columns.append(tf.feature_column.categorical_column_with_hash_bucket(
			name, hash_bucket_size=1000))
	deep_columns = []
	for name in continuous_columns:
		deep_columns.append(tf.feature_column.numeric_column(name))
	for col in wide_columns:
		deep_columns.append(tf.feature_column.embedding_column(
			col, dimension=8))
	price_buckets = []
	for i in range(7):
		price_buckets.append( tf.feature_column.bucketized_column(deep_columns[i],
			boundaries=[10, 20, 30, 40, 60, 80, 100, 200, 300, 400,	600, 800,
				1000, 2000, 3000, 4000, 6000, 8000,	10000]))
	wide_columns.extend(price_buckets)
	cross = [["C3", "C4"], ["C4", "C5"], ["C3","C4","C5"]]
	crossed_columns = []
	for name in cross:
		crossed_columns.append(tf.feature_column.crossed_column(name,hash_bucket_size=50))
	return wide_columns, deep_columns, crossed_columns

def build_estimator(model_dir, model_type):
	wide_columns, deep_columns, crossed_columns = process_feature() 
	"""Build an estimator."""
	if model_type == "wide":
		m = tf.estimator.LinearClassifier(
				model_dir=model_dir, feature_columns=wide_columns +
				crossed_columns)
	elif model_type == "deep":
		m = tf.estimator.DNNClassifier(
				model_dir=model_dir,
				feature_columns=deep_columns,
				hidden_units=[100, 50])
	else:
		m = tf.estimator.DNNLinearCombinedClassifier(
				model_dir=model_dir,
				linear_feature_columns=crossed_columns,
				dnn_feature_columns=deep_columns,
				dnn_hidden_units=[100, 50])
	return m

def build_estimator_1(model_dir):
	wide_columns, deep_columns, crossed_columns = process_feature() 
	estimator = tf.estimator.DNNLinearCombinedClassifier(
	#estimator = tf.estimator.DNNLinearCombinedRegressor(
			model_dir = model_dir, 
			linear_feature_columns = wide_columns+ crossed_columns ,
			dnn_feature_columns = deep_columns,
			dnn_hidden_units=[100,50,30,], )#dnn_optimizer = 'Adam')
	return estimator

def save_predict_data(predict, model_step):
	file_name = './predict_results/predict_class' + str(model_step) + '.txt'
	with open(file_name, 'w') as f:
		for item in predict:
			#print(item)
			#f.write(str(item['predictions'][0]) + '\n')
			f.write(str(item['classes'][0]+'\n'))

def main(argv=None):
	train_file_name = '.csv'
	test_file_name = '.csv'
	predict_name = '.csv'
	model_dir = 'model'
	train_step = 2000
	estimator = build_estimator(model_dir,'deep')
	estimator.train(input_fn = input_fn(train_file_name,num_epochs = None,
		shuffle = True,), steps = train_step)
	result = estimator.evaluate(input_fn = input_fn(test_file_name,))
	for key in sorted(result):
		print("%s: %s"%(key, result[key]))
	predict = estimator.predict(input_fn=input_fn(predict_name, is_training =
			False))
	save_predict_data(predict, train_step)
if __name__ == '__main__':
	tf.app.run()









