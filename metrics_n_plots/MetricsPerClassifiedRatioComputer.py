#!/usr/bin/python3

import numpy as np,copy
import sys,getopt,os
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,recall_score
from scipy import interpolate
from scipy.stats.mstats import gmean
from imblearn.metrics import geometric_mean_score
import pandas as pd
import copy

pd.options.mode.chained_assignment = None  # default='warn'

class Measure(object):
	def __init__(self):
		self.mean = None
		self.std = None

class Metrics(object):
	def __init__(self):
		self.accuracy = None
		self.f_measure = None
		self.g_mean_multiclass = None
		self.g_mean_macro = None
		self.classified_ratio = None

def main(argv):

	def compute_interpolation(x,y):
		'''
		The function gets two vectors, first with x points and second with y response (value if known and None on the other hand).
		Function interpolate not known values of y vector using known one.
		'''
		known_indexes = [i for i,e in enumerate(y) if e!=None]
		to_interpolate_indexes = [i for i,e in enumerate(y) if e==None]

		# Filtering duplicate x values
		known_indexes_nodup = []
		seen = set()
		for i in known_indexes:
			if x[i] not in seen:
				known_indexes_nodup.append(i)
			seen.add(x[i])

		f = interpolate.interp1d([x[i] for i in known_indexes_nodup],[y[i] for i in known_indexes_nodup],kind='linear')

		for i in to_interpolate_indexes:
			y[i] = float(f(x[i]))

		to_normalize = np.array(y) > 1

		for i in range(0,len(y)):
			if to_normalize[i]:
				y[i] = 1.0

	data_folder = './data_rf/metrics_per_fold/'

	files = os.listdir(data_folder)

	metrics_per_config = {}

	for file in files:
		if 'gamma' in file and 'cr' not in file:
			file_splitted = file.split('_')
			config = '_'.join(file_splitted[:6])
			gamma = float(file_splitted[7])

			if config not in metrics_per_config:
				metrics_per_config[config] = {}

			fin = open(data_folder+file,'r')
			lines = fin.readlines()
			fin.close()

			fold = 0

			for line in lines:
				line_splitted = line.split()

				if fold not in metrics_per_config[config]:
					metrics_per_config[config][fold] = []

				temp_metrics = Metrics()

				temp_metrics.accuracy = float(line_splitted[0])
				temp_metrics.f_measure = float(line_splitted[1])
				temp_metrics.g_mean_multiclass = float(line_splitted[2])
				temp_metrics.g_mean_macro = float(line_splitted[3])
				temp_metrics.classified_ratio = float(line_splitted[4])

				metrics_per_config[config][fold].append(temp_metrics)

				fold += 1

#####################################################################################

	# Sort metrics to save data
	for config in metrics_per_config:
		for fold in metrics_per_config[config]:
			indexes = np.argsort([metrics.classified_ratio for metrics in metrics_per_config[config][fold]])
			metrics_per_config[config][fold] = [metrics_per_config[config][fold][i] for i in indexes]

	data_folder = './data/metrics_per_cr/'

	for config in metrics_per_config:
		for fold in metrics_per_config[config]:
			file = open(data_folder+config+'_fold_'+str(fold)+'_metrics_per_cr_pre_int.dat','w+')
			for metrics in metrics_per_config[config][fold]:
				file.write(
					str(metrics.classified_ratio)+' '+
					str(metrics.accuracy)+' '+
					'0.0 '+
					str(metrics.f_measure)+' '+
					'0.0 '+
					str(metrics.g_mean_multiclass)+' '+
					'0.0 '+
					str(metrics.g_mean_macro)+' '+
					'0.0\n')
			file.close()

#####################################################################################

	# Aggregate classified_ratio per config[fold]
	cr_aggregated_per_config = {}
	for config in metrics_per_config:
		if config not in cr_aggregated_per_config:
			cr_aggregated_per_config[config] = []
		cnt = 0
		for fold in metrics_per_config[config]:
			for metrics in metrics_per_config[config][fold]:
				cr_aggregated_per_config[config].append(metrics.classified_ratio)
				cnt += 1

	# Sort and remove duplicate from aggregated classified_ratio config[fold]
	for config in cr_aggregated_per_config:
		cr_aggregated_per_config[config] = list(set(cr_aggregated_per_config[config]))
		indexes = np.argsort(cr_aggregated_per_config[config])
		cr_aggregated_per_config[config] = [cr_aggregated_per_config[config][i] for i in indexes]

	# Compute classified_ratio's max values per config of min values per fold
	maxmin_cr_per_config = {}
	for config in metrics_per_config:
		to_maximize = []
		for fold in metrics_per_config[config]:
			to_maximize.append(min([metrics.classified_ratio for metrics in metrics_per_config[config][fold]]))
		maxmin_cr_per_config[config] = max(to_maximize)

	# sys.exit()

	# Apply threshold to aggregated classified_ratio and metrics
	for config in maxmin_cr_per_config:
		cr_aggregated_per_config[config] = [cr_aggregated for cr_aggregated in cr_aggregated_per_config[config] if cr_aggregated >= maxmin_cr_per_config[config]]

	temp = {}
	# Add to metrics_per_config the aggregated per fold classified_ratio
	for config in metrics_per_config:
		if config not in temp:
			temp[config] = {}
		for fold in metrics_per_config[config]:
			temp[config][fold] = list(metrics_per_config[config][fold])
			for cr_aggregated in cr_aggregated_per_config[config]:
				if cr_aggregated not in [metrics.classified_ratio for metrics in temp[config][fold]]:
					temp_metrics = Metrics()
					temp_metrics.classified_ratio = cr_aggregated
					temp[config][fold].append(temp_metrics)
			indexes = np.argsort([metrics.classified_ratio for metrics in temp[config][fold]])
			temp[config][fold] = [temp[config][fold][i] for i in indexes]
	metrics_per_config = temp

	# Compute interpolation per fold of metrics to obtain all values per aggregated classified_ratio, then sort metrics per classified_ratio
	for config in metrics_per_config:
		for fold in metrics_per_config[config]:

			accuracy = [metrics.accuracy for metrics in metrics_per_config[config][fold]]
			f_measure = [metrics.f_measure for metrics in metrics_per_config[config][fold]]
			g_mean_multiclass = [metrics.g_mean_multiclass for metrics in metrics_per_config[config][fold]]
			g_mean_macro = [metrics.g_mean_macro for metrics in metrics_per_config[config][fold]]
			classified_ratio = [metrics.classified_ratio for metrics in metrics_per_config[config][fold]]

			compute_interpolation(classified_ratio,accuracy)
			compute_interpolation(classified_ratio,f_measure)
			compute_interpolation(classified_ratio,g_mean_multiclass)
			compute_interpolation(classified_ratio,g_mean_macro)

			i = 0
			for accuracy_value,f_measure_value,g_mean_multiclass_value,g_mean_macro_value,classified_ratio_value in zip(accuracy,f_measure,g_mean_multiclass,g_mean_macro,classified_ratio):
				metrics_per_config[config][fold][i].accuracy = accuracy_value
				metrics_per_config[config][fold][i].f_measure = f_measure_value
				metrics_per_config[config][fold][i].g_mean_multiclass = g_mean_multiclass_value
				metrics_per_config[config][fold][i].g_mean_macro = g_mean_macro_value
				metrics_per_config[config][fold][i].classified_ratio = classified_ratio_value
				i += 1

			indexes = np.argsort([metrics.classified_ratio for metrics in metrics_per_config[config][fold]])
			metrics_per_config[config][fold] = [metrics_per_config[config][fold][i] for i in indexes]
			
	# Compute means and stds over folds per aggregated classified_ratio
	metrics_averaged_per_config = {}
	for config in metrics_per_config:

		if config not in metrics_averaged_per_config:
			metrics_averaged_per_config[config] = []

		metrics_to_average = {}

		for fold in metrics_per_config[config]:
			for metrics in metrics_per_config[config][fold]:
				if metrics.classified_ratio not in metrics_to_average:
					metrics_to_average[metrics.classified_ratio] = []
				metrics_to_average[metrics.classified_ratio].append(metrics)

		for cr in metrics_to_average:
			if len([metrics.accuracy for metrics in metrics_to_average[cr]])==10:
				metrics_averaged = Metrics()
				metrics_averaged.accuracy = Measure()
				metrics_averaged.f_measure = Measure()
				metrics_averaged.g_mean_multiclass = Measure()
				metrics_averaged.g_mean_macro = Measure()

				metrics_averaged.classified_ratio = np.mean([metrics.classified_ratio for metrics in metrics_to_average[cr]])

				metrics_averaged.accuracy.mean = np.mean([metrics.accuracy for metrics in metrics_to_average[cr]])
				metrics_averaged.f_measure.mean = np.mean([metrics.f_measure for metrics in metrics_to_average[cr]])
				metrics_averaged.g_mean_multiclass.mean = np.mean([metrics.g_mean_multiclass for metrics in metrics_to_average[cr]])
				metrics_averaged.g_mean_macro.mean = np.mean([metrics.g_mean_macro for metrics in metrics_to_average[cr]])

				metrics_averaged.accuracy.std = np.std([metrics.accuracy for metrics in metrics_to_average[cr]])
				metrics_averaged.f_measure.std = np.std([metrics.f_measure for metrics in metrics_to_average[cr]])
				metrics_averaged.g_mean_multiclass.std = np.std([metrics.g_mean_multiclass for metrics in metrics_to_average[cr]])
				metrics_averaged.g_mean_macro.std = np.std([metrics.g_mean_macro for metrics in metrics_to_average[cr]])

				metrics_averaged_per_config[config].append(metrics_averaged)

		indexes = np.argsort([metrics.classified_ratio for metrics in metrics_averaged_per_config[config]])
		metrics_averaged_per_config[config] = [metrics_averaged_per_config[config][i] for i in indexes]

	data_folder = './data_rf/metrics_per_cr/'

	if not os.path.exists('./data_rf'):
		os.makedirs('./data_rf')
		os.makedirs(data_folder)
	elif not os.path.exists(data_folder):
		os.makedirs(data_folder)

	for config in metrics_averaged_per_config:
		file = open(data_folder+config+'_metrics_per_cr.dat','w+')
		for metrics_averaged in metrics_averaged_per_config[config]:
			file.write(
				str(metrics_averaged.classified_ratio)+' '+
				str(metrics_averaged.accuracy.mean)+' '+
				str(metrics_averaged.accuracy.std)+' '+
				str(metrics_averaged.f_measure.mean)+' '+
				str(metrics_averaged.f_measure.std)+' '+
				str(metrics_averaged.g_mean_multiclass.mean)+' '+
				str(metrics_averaged.g_mean_multiclass.std)+' '+
				str(metrics_averaged.g_mean_macro.mean)+' '+
				str(metrics_averaged.g_mean_macro.std)+'\n')

		file.close()

#####################################################################################

	for config in metrics_per_config:
		for fold in metrics_per_config[config]:
			file = open(data_folder+config+'_fold_'+str(fold)+'_metrics_per_cr.dat','w+')
			for metrics in metrics_per_config[config][fold]:
				file.write(
					str(metrics.classified_ratio)+' '+
					str(metrics.accuracy)+' '+
					'0.0 '+
					str(metrics.f_measure)+' '+
					'0.0 '+
					str(metrics.g_mean_multiclass)+' '+
					'0.0 '+
					str(metrics.g_mean_macro)+' '+
					'0.0\n')
			file.close()

#####################################################################################

if __name__ == "__main__":
	os.environ["DISPLAY"] = ":0" # used to show xming display
	main(sys.argv[1:])