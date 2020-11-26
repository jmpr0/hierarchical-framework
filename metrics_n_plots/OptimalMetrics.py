#!/usr/bin/python3

import numpy as np,sys,os,getopt

def main(argv):

	classifier_name = 'rf'

	try:
		opts, args = getopt.getopt(argv,"hc:")
	except getopt.GetoptError:
		print('MetricsComputer.py -i <input_file>')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print('MetricsComputer.py -i <input_file>')
			sys.exit()
		if opt in ("-c"):
			classifier_name = arg

	files = os.listdir('./data_'+classifier_name+'/metrics/join_fp/')
	data_fold = './data_'+classifier_name+'/metrics/join_fp/'

	fout = open(data_fold+'optimal.dat','w+')
	fout.close()

	for file in sorted(files, reverse=True):
		if 'gamma' not in file and 'optimal' not in file:
			lines = open(data_fold+file,'r').readlines()
			lines_splitted = [line.split() for line in lines]
			xs = [int(line_splitted[0]) for line_splitted in lines_splitted]
			accuracies = [float(line_splitted[1]) for line_splitted in lines_splitted]
			f_measures = [float(line_splitted[3]) for line_splitted in lines_splitted]
			g_mean_multiclass = [float(line_splitted[5]) for line_splitted in lines_splitted]
			g_mean_macro = [float(line_splitted[7]) for line_splitted in lines_splitted]
			accuracies_std = [float(line_splitted[2]) for line_splitted in lines_splitted]
			f_measures_std = [float(line_splitted[4]) for line_splitted in lines_splitted]
			g_mean_multiclass_std = [float(line_splitted[6]) for line_splitted in lines_splitted]
			g_mean_macro_std = [float(line_splitted[8]) for line_splitted in lines_splitted]
			fout = open(data_fold+'optimal.dat','a')

			if 'inferred' not in file:
				try:
					max_fmeasure_index = f_measures.index(max(f_measures))
				except:
					print('error',file)

			fout.write(
				file+'\t\t'+
				str(accuracies[max_fmeasure_index])+'\t'+
				str(f_measures[max_fmeasure_index])+'\t'+
				str(g_mean_multiclass[max_fmeasure_index])+'\t'+
				str(g_mean_macro[max_fmeasure_index])+'\t'+
				str(accuracies_std[max_fmeasure_index])+'\t'+
				str(f_measures_std[max_fmeasure_index])+'\t'+
				str(g_mean_multiclass_std[max_fmeasure_index])+'\t'+
				str(g_mean_macro_std[max_fmeasure_index])+'\t'+
				# str(xs[max_fmeasure_index])+'\t'+
				# str(xs[max_fmeasure_index])+'\t'+
				# str(xs[max_fmeasure_index])+'\t'+
				str(xs[max_fmeasure_index])+'\n')

	# files = os.listdir('./data_'+classifier_name+'/metrics/join_gamma/')
	# data_fold = './data_'+classifier_name+'/metrics/join_gamma/'

	# fout = open(data_fold+'optimal.dat','w+')
	# fout.close()

	# for file in files:
	# 	lines = open(data_fold+file,'r').readlines()
	# 	lines_splitted = [line.split() for line in lines]
	# 	xs = [float(line_splitted[0]) for line_splitted in lines_splitted]
	# 	f_measures = [float(line_splitted[3]) for line_splitted in lines_splitted]
	# 	fout = open(data_fold+'optimal.dat','a')
	# 	fout.write(file+' '+str(xs[f_measures.index(max(f_measures))])+'\n')

if __name__ == "__main__":
	os.environ["DISPLAY"] = ":0" # used to show xming display
	main(sys.argv[1:])