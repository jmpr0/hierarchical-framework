#!/usr/bin/python

import numpy as np,sys,os

def main(argv):

	files = os.listdir('./')
	files_splitted = list()

	lines = {}
	gamma_lines = {}

	files = [file for file in files if 'hierarchical' not in file]

	for file in files:
		files_splitted.append(file.split('_'))

	for file,file_splitted in zip(files,files_splitted):
		if file.endswith('.dat'):
			root = '_'.join(np.r_[file_splitted[:4],file_splitted[6:]])

			x = file_splitted[5]

			fin = open(file,'r')

			if root not in lines:
				lines[root] = list()
			lines[root].append(x+' '+fin.readline())
			
			fin.close()

			if 'gamma' in file:
				root = '_'.join(np.r_[file_splitted[:6],file_splitted[8:]])

				# print file,file_splitted
				gamma = file_splitted[7]

				fin = open(file,'r')

				if root not in gamma_lines:
					gamma_lines[root] = list()
				gamma_lines[root].append(gamma+' '+fin.readline())

				fin.close()


	data_folder = './join_fp/'

	if not os.path.exists(data_folder):
		os.makedirs(data_folder)

	for key,value in lines.iteritems():
		fout = open(data_folder+key,'w+')
		fout.close()
		fout = open(data_folder+key,'a')

		to_sort = [int(v.split()[0]) for v in value]
		sorting = np.argsort(to_sort)
		value = [value[i] for i in sorting]

		fout.writelines(value)

		fout.close()

	data_folder = './join_gamma/'

	if not os.path.exists(data_folder):
		os.makedirs(data_folder)

	for key,value in gamma_lines.iteritems():
		fout = open(data_folder+key,'w+')
		fout.close()
		fout = open(data_folder+key,'a')

		to_sort = [float(v.split()[0]) for v in value]
		sorting = np.argsort(to_sort)
		value = [value[i] for i in sorting]

		fout.writelines(value)

		fout.close()

if __name__ == "__main__":
	os.environ["DISPLAY"] = ":0" # used to show xming display
	main(sys.argv[1:])