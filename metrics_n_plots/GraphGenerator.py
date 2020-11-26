#!/usr/bin/python3

import numpy as np,copy
import sys,getopt,os
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,recall_score
from scipy.stats.mstats import gmean
from imblearn.metrics import geometric_mean_score
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import write_dot,graphviz_layout
import math

pd.options.mode.chained_assignment = None  # default='warn'

os.environ["DISPLAY"] = ":0"  # used to show xming display

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

	np.random.seed(0)

	input_file = ''

	general = False
	classifier_name = 'wrf'

	config_file = ''
	config = ''

	try:
		opts, args = getopt.getopt(argv,"hi:gc:o:","[input_file=]")
	except getopt.GetoptError:
		print('GraphGenarator.py -i <input_file>')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print('GraphGenarator.py -i <input_file>')
			sys.exit()
		if opt in ("-i", "--input_file"):
			input_file = arg
		if opt in ("-g"):
			general = True
		if opt in ("-c", "--clf"):
			classifier_name = arg
		if opt in ("-o", "--configuration"):
			config_file = arg

	if config_file:
		config_name = config_file.split('.')[0]

	if config_file:
		if not config_file.endswith('.json'):
			print('config file must have .json extention')
			sys.exit()

		import json

		with open(config_file) as f:
			config = json.load(f)

	dict_w_n = {
		'wrf': 'RF',
		'wbn': 'BN_TAN',
		'wj48': 'C4.5',
		'wnb': 'NB_SD'
	}

	print('\n'+input_file+' elaboration...\n')
	#load .dat file
	G = nx.read_gpickle(input_file)
	file_name = os.path.basename(input_file)
	file_name_splitted = file_name.split('_')

	packets_number = 0
	features_number = 0

	if file_name_splitted[2]=='p':
		packets_number = file_name_splitted[3]
	else:
		features_number = file_name_splitted[3]

	folder_discriminator = classifier_name

	# if config_file:
	# 	folder_discriminator = config_name

	dict_tags_num = {
		'ROOT':'(1)',
		'Tor':'(2)',
		'I2P':'(3)',
		'TorApp':'(4)',
		'TorPT':'(5)',
		'I2PApp0BW':'(6)',
		'I2PApp80BW':'(7)',
		'I2PApp':'(8)'
	}

	data_folder = './data_'+folder_discriminator+'/metrics/'

	new_labels = {}

	color_map = {}

	for node,data in G.nodes.items():

		if config:
			classifier_name = dict_w_n[config[data['tag'] + '_' + str(data['level'] + 1)]['c']]
			if 'f' in config[data['tag'] + '_' + str(data['level'] + 1)]:
				features_number = config[data['tag'] + '_' + str(data['level'] + 1)]['f']
			else:
				features_number = config[data['tag'] + '_' + str(data['level'] + 1)]['p']

		file = open(data_folder+file_name_splitted[0]+'_'+file_name_splitted[1]+'_level_'+str(data['level']+1)+'_'+file_name_splitted[2]+'_'+file_name_splitted[3]+'_tag_'+data['tag']+'_all_metrics.dat','r')

		line = file.readline().split()
		metrics = Metrics()
		metrics.accuracy = Measure()
		metrics.f_measure = Measure()
		metrics.g_mean_multiclass = Measure()
		metrics.g_mean_macro = Measure()
		metrics.accuracy.mean = float(line[0])
		metrics.accuracy.std = float(line[1])
		metrics.f_measure.mean = float(line[2])
		metrics.f_measure.std = float(line[3])
		metrics.g_mean_multiclass.mean = float(line[4])
		metrics.g_mean_multiclass.std = float(line[5])
		metrics.g_mean_macro.mean = float(line[6])
		metrics.g_mean_macro.std = float(line[7])

		g_mean_corrected = metrics.g_mean_macro.mean
		if g_mean_corrected==0:
			g_mean_corrected=metrics.f_measure.mean
		metrics_mean = (metrics.accuracy.mean+metrics.f_measure.mean+g_mean_corrected)/3

		node_name = data['tag']

		if not ((('Tor' in node_name[-4:]  or 'JonDonym' in node_name) and data['level'] == 2) or
			('JonDonym' in node_name and data['level'] == 1)):

			node_name = dict_tags_num[node_name] + '\n' + node_name + '\n' + classifier_name + '[' + str(features_number) + ']'

		else:

			node_name = '\n' + node_name + '\n'

		if not general:

			################################################
			# if metrics.accuracy.mean < 1 and metrics.f_measure.mean < 1 and metrics.g_mean_macro.mean > 0:
			# 	node_name += '\nA: %.2f±%.2f%%\nF: %.2f±%.2f%%\nG: %.2f±%.2f%%' % (metrics.accuracy.mean*100,metrics.accuracy.std*100,metrics.f_measure.mean*100,metrics.f_measure.std*100,metrics.g_mean_macro.mean*100,metrics.g_mean_macro.std*100)
			# else:
			# 	for i in range(0,int((20-len(node_name))/2)):
			# 		node_name = ' '+node_name+' '
			################################################

			if node_name in new_labels.values():
				node_name = ' '+node_name+' '

			if not np.isnan(metrics_mean):

				coefficient = math.pow(3.0 / 8.0 * math.atan(8.0 * metrics_mean - 4.0) + .5, 2.0)

				if coefficient > 1.0:
					coefficient = 1.0

				rgb = []

				reds = True
				blues = not reds

				if reds:

					rgb.append(round(204+(1-coefficient)*51))
					rgb.append(round((1-coefficient)*255))
					rgb.append(round((1-coefficient)*255))

				if blues:

					rgb.append(round(37+(1-coefficient)*218))
					rgb.append(round(81+(1-coefficient)*174))
					rgb.append(round(135+(1-coefficient)*120))

				color_map[node_name] = '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])

			if (('Tor' in node_name[-4:]  or 'JonDonym' in node_name) and data['level'] == 2) or ('JonDonym' in node_name and data['level'] == 1):

				color_map[node_name] = '#000000'

		else:

			if not ((('Tor' in node_name[-4:]  or 'JonDonym' in node_name) and data['level'] == 2) or
				('JonDonym' in node_name and data['level'] == 1)):

				node_name = classifier_name + '\n' + node_name + '\n\n' + str(features_number)

			else:

				node_name = '\n' + node_name + '\n\n '

			node_name = node_name.split('\n')[2] # necessario per nominare tutti i nodi col solo nome della classe

			if node_name in new_labels.values():
				node_name = ' '+node_name+' '

			color_map[node_name] = '#cc0000'
			
		new_labels[node] = node_name

	image_folder = './image_'+folder_discriminator+'/graph/'

	if not os.path.exists('./image_'+folder_discriminator):
		os.makedirs('./image_'+folder_discriminator)
		os.makedirs(image_folder)
	elif not os.path.exists(image_folder):
		os.makedirs(image_folder)

	print(new_labels)

	G = nx.relabel_nodes(G,new_labels)

	plt.figure(figsize=(11,5.5))
	# plt.figure(figsize=(15,7))
	pos = graphviz_layout(G,prog='dot')

	# Sort graph
	pos_per_level = {}

	dict_tags_abc = []
	dict_tags_abc.append({
		'ROOT':'a'
	})
	dict_tags_abc.append({
		'Tor':'a',
		'JonDonym':'c',
		'I2P':'b'
	})
	dict_tags_abc.append({
		'TorPT':'c',
		'TorApp':'b',
		'I2PApp80BW':'e',
		'Tor':'a',
		'I2PApp0BW':'d',
		'JonDonym':'g',
		'I2PApp':'f'
	})

	for k in pos:
		if pos[k][1] not in pos_per_level:
			pos_per_level[pos[k][1]] = []

	for h in pos_per_level:
		for k in pos:
			if pos[k][1] == h:
				pos_per_level[h].append([k, pos[k][0]])

	sorted_nodes_per_level = {}
	sorted_positions_per_level = {}

	hs = sorted(list(pos_per_level.keys()), reverse=True)

	for index, h in enumerate(hs):
		if not general:
			# for node in pos_per_level[h]:
			# 	print(node[0])
			# 	print(node[0].split('\n'))
			# 	print(node[0].split('\n')[1])
			# 	print(node[0].split('\n')[1].replace(' ',''))
			indexes = np.argsort([ dict_tags_abc[index][node[0].split('\n')[1].replace(' ', '')] for node in pos_per_level[h]])
			# print(indexes)
		else:
			# for node in pos_per_level[h]:
				# print('node[0] ',node[0])
				# print('node[0].split() ',node[0].split('\n'))
				# print('node[0].split()[2] ',node[0].split('\n')[2])
				# print('node[0].split()[2].replace(' ','')',node[0].split()[2].replace(' ',''))
			indexes = np.argsort([ dict_tags_abc[index][node[0].split('\n')[0].replace(' ', '')] for node in pos_per_level[h]])
			# print(indexes)
		sorted_nodes_per_level[h] = [ pos_per_level[h][i][0] for i in indexes ]
		sorted_positions_per_level[h] = sorted([ node[1] for node in pos_per_level[h] ])

	max_h = max(max([ sorted_positions_per_level[h] for h in sorted_positions_per_level ]))
	min_h = min(min([ sorted_positions_per_level[h] for h in sorted_positions_per_level ]))

	for h in sorted_positions_per_level:
		h_n = len(sorted_positions_per_level[h])
		for index, position in enumerate(sorted_positions_per_level[h]):
			sorted_positions_per_level[h][index] = (max_h - min_h) / h_n * index + (max_h - min_h) / (h_n * 2)

	for h in sorted_nodes_per_level:
		for index, node in enumerate(sorted_nodes_per_level[h]):
			pos[node] = (sorted_positions_per_level[h][index], h)

	if not general:
		font_size = 15
		node_size = 10500
		# node_size = 20000
	else:
		font_size = 15
		node_size = 10500

	nx.draw(G,pos,with_labels=True,arrows=True,font_size=font_size,font_name='Helvetica',font_family='monospace',font_weight='bold')
	for node, _ in G.nodes.items():
		nx.draw_networkx_nodes(G,pos,with_labels=False,nodelist=[node],node_color=color_map[node],node_size=node_size,node_shape='o')
	for node, _ in G.nodes.items():
		nx.draw_networkx_nodes(G,pos,with_labels=False,nodelist=[node],node_color='#ffffff',node_size=node_size*.85,node_shape='o')
	plt.tight_layout()
	if not general:
		if config:
			plt.savefig(image_folder+'/graph_'+config_name+'.eps',format='eps',dpi=1000)
		elif packets_number!=0:
			plt.savefig(image_folder+'/graph_early_'+str(packets_number)+'.eps',format='eps',dpi=1000)
		else:
			plt.savefig(image_folder+'/graph_flow_'+str(features_number)+'.eps',format='eps',dpi=1000)
	else:
		plt.savefig(image_folder+'/general_graph.eps',format='eps',dpi=1000)

if __name__ == "__main__":
	os.environ["DISPLAY"] = ":0" # used to show xming display
	main(sys.argv[1:])
