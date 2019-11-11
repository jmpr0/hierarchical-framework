#!/usr/bin/python3

import getopt
import json
import os
import sys

import numpy as np

from core.hierarchical_classifier import HierarchicalClassifier
from configs import *

os.environ["DISPLAY"] = ":0"  # used to show xming display

# Dataset infos
input_file = ''
levels_number = 0
input_is_image = False
memoryless = False
anomaly_classes = ''
hidden_classes = ''
benign_class = ''

# Goal infos
k = 10
max_level_target = 0
level_target = -1
features_number = 0
packets_number = 0
starting_fold = 0
ending_fold = k

# Model infos
classifier_name = ''
detector_name = ''
epochs_number = 10
weight_features = False
anomaly = False
deep = False
early = False
unsupervised = False
optimize = False
nominal_features_index = None
n_clusters = 1

# Execution infos
workers_number = 0
executors_number = 1
parallelize = False
buckets_number = 1

# Output infos
config_file = ''
config = ''
arbitrary_discr = ''

classifier_class = ''
classifier_opts = ''
detector_class = ''
detector_opts = ''

try:
    opts, args = getopt.getopt(
        sys.argv[1:], "hi:n:t:f:p:c:o:w:x:a:e:d:s:C:H:b:N:OT:WPB:UF:G:EM",
        "[input_file=,levels_number=,max_level_target=,features_number=,packets_number=,\
            classifier_name=,configuration=,workers_number=,executors_number=,anomaly_class=,\
            epochs_number=,detector_name=,arbitrary_discriminator=,n_clusters=,hidden_classes=,\
            benign_class=,nominal_features_index=,level_target=,buckets_number=]"
    )
except getopt.GetoptError:
    print(sys.argv[0], '-i <input_file> -n <levels_number> \
        (-f <features_number>|-p <packets_number>) -c <classifier_name> (-o <configuration_file>) \
        (-w <workers_number> -x <executors_number>) (-a <anomaly_class> -e epochs_number -d detector_name) (-s arbitrary_discr) -O')
    print(sys.argv[0], '-h (or --help) for a more careful help')
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-h", "--help"):
        print(sys.argv[0], '-i <input_file> -n <levels_number> \
        (-f <features_number>|-p <packets_number>) -c <classifier_name> (-o <configuration_file>) \
        (-w <workers_number> -x <executors_number>) (-a <anomaly_class> -e epochs_number -d detector_name)')
        print(
            'Options:\n\t-i: dataset file, must be in arff or in csv format\n\t-n: number of levels (number of labels\' columns)')
        print(
            '\t-f or -p: former refers features number, latter refers packets number\n\t-c: classifier name choose from following list:')
        for sc in supported_classifiers:
            print('\t\t-c ' + sc + '\t--->\t' + supported_classifiers[sc].split('_')[1] + '\t\timplemented in ' +
                  supported_classifiers[sc].split('_')[0])
        print('\t-o: configuration file in JSON format')
        print('\t-w: number of workers when used with spark.ml classifiers')
        print('\t-x: number of executors per worker when used with spark.ml classifiers')
        print('\t-a: hierarchical classifier starts with a one-class classification at each level if set and classification is binary (i.e. Anomaly Detection)\n\
            \tmust provide label of baseline class')
        print('\t-e: number of epochs in caso of DL models')
        print('\t-s: arbitrary discriminator, a string to put at the start of saved files')
        print('\t-O: optimization of models')
        for sd in supported_detectors:
            print('\t\t-c ' + sd + '\t--->\t' + supported_detectors[sd].split('_')[1] + '\t\timplemented in ' +
                  supported_detectors[sd].split('_')[0])
        sys.exit()
    if opt in ("-i", "--input_file"):
        input_file = arg
    if opt in ("-n", "--levels_number"):
        levels_number = int(arg)
    if opt in ("-t", "--max_level_target"):
        max_level_target = int(arg)
    if opt in ("-f", "--nfeat"):
        features_number = int(arg)
    if opt in ("-p", "--npacket"):
        packets_number = int(arg)
    if opt in ("-c", "--clf"):
        classifier_name = arg
    if opt in ("-o", "--configuration"):
        config_file = arg
    if opt in ("-w", "--workers_number"):
        workers_number = int(arg)
    if opt in ("-x", "--executors_number"):
        executors_number = int(arg)
    if opt in ("-a", "--anomaly_class"):
        anomaly_classes = arg
    if opt in ("-e", "--epochs_number"):
        epochs_number = int(arg)
    if opt in ("-d", "--detector_name"):
        detector_name = arg
    if opt in ("-s", "--arbitrary_discr"):
        arbitrary_discr = arg
    if opt in ("-C", "--n_clusters"):
        n_clusters = int(arg)
    if opt in ("-H", "--hidden_classes"):
        hidden_classes = arg
    if opt in ("-b", "--benign_class"):
        benign_class = arg
    if opt in ("-N", "--nominal_features_index"):
        nominal_features_index = [int(i) for i in arg.split(',')]
    if opt in ("-O", "--optimize"):
        optimize = True
    if opt in ("-T", "--level_target"):
        level_target = int(arg)
    if opt in ("-W", "--weight_features"):
        weight_features = True
    if opt in ("-P", "--parallelize"):
        parallelize = True
    if opt in ("-B", "--buckets_number"):
        buckets_number = int(arg)
    if opt in ("-U", "--unsupervised"):
        unsupervised = True
    if opt in ("-F", "--starting_fold"):
        starting_fold = int(arg)
    if opt in ("-G", "--ending_fold"):
        ending_fold = int(arg)
    if opt in ("-E", "--early"):
        early = True
    if opt in ("-M", "--memoryless"):
        memoryless = True

if anomaly_classes != '' or (benign_class != '' and hidden_classes == ''):
    anomaly = True

# import_str variable contains initial character of name of used classifiers
# it is used to import specific module
import_str = ''

if config_file:
    if not config_file.endswith('.json'):
        print('config file must have .json extention')
        sys.exit()

    with open(config_file) as f:
        config = json.load(f)

    print(config)

    for node in config:
        print(config[node])
        if config[node]['c'] not in supported_classifiers:
            print('Classifier not supported in configuration file\nList of available classifiers:\n')
            for sc in np.sort(supported_classifiers.keys()):
                print('-c ' + sc + '\t--->\t' + supported_classifiers[sc].split('_')[1] + '\t\timplemented in ' +
                      supported_classifiers[sc].split('_')[0])
            print('Configuration inserted:\n', config)
            sys.exit()
        if config[node]['c'][0] not in import_str:
            import_str += config[node][0]
            if config[node][1] == 'k':
                import_str += config[node][1]

else:

    if packets_number != 0 and features_number != 0 or packets_number == features_number:
        print('-f and -p option should not be used together')
        sys.exit()

    if len(classifier_name.split('_')) > 1:
        classifier_class = classifier_name.split('_')[0]
        classifier_opts = classifier_name.split('_')[1:]
    else:
        classifier_class = classifier_name
        classifier_opts = []

    if len(detector_name.split('_')) > 1:
        detector_class = detector_name.split('_')[0]
        detector_opts = detector_name.split('_')[1:]
    else:
        detector_class = detector_name
        detector_opts = []

    anomaly_classes = [v for v in anomaly_classes.split(',') if v != '']
    hidden_classes = [v for v in hidden_classes.split(',') if v != '']

    if classifier_class != '' and classifier_class not in supported_classifiers:
        print('Classifier not supported\nList of available classifiers:\n')
        for sc in supported_classifiers:
            print('-c ' + sc + '\t--->\t' + supported_classifiers[sc].split('_')[1] + '\t\timplemented in ' +
                  supported_classifiers[sc].split('_')[0])
        sys.exit()

    if len(anomaly_classes) > 0 and detector_class not in supported_detectors:
        print('Detector not supported\nList of available detectors:\n')
        for sd in supported_detectors:
            print('\t\t-c ' + sd + '\t--->\t' + supported_detectors[sd].split('_')[1] + '\t\timplemented in ' +
                  supported_detectors[sd].split('_')[0])
        sys.exit()

    if classifier_name != '':
        import_str = classifier_name[0]
        if classifier_name[1] == 'k':
            import_str = classifier_name[:2]
    if detector_name != '':
        import_str = detector_name[0]

if levels_number == 0:
    print('Number of level must be positive and non zero')
    sys.exit()

if max_level_target == 0 or max_level_target > levels_number:
    max_level_target = levels_number

if level_target > levels_number:
    level_target = levels_number
    print('Warning: level target was greater than levels number, it will be treated as equal.')

if not input_file.endswith('.arff') and not input_file.endswith('.csv') and not input_file.endswith('.pickle'):
    print('input files must be or .arff or .csv or .pickle')
    sys.exit()

if 'k' in import_str:
    # For reproducibility
    from tensorflow import set_random_seed, logging

    set_random_seed(0)
    logging.set_verbosity(logging.ERROR)
    deep = True

if 'd' in import_str:
    from core.wrappers.spark_wrapper import SingletonSparkSession

    raw_conf = [
        ('spark.app.name', '_'.join(sys.argv)),
        ('spark.driver.cores', 3),
        ('spark.driver.memory', '7g'),
        ('spark.executor.memory', '2900m'),
        ('spark.master', 'spark://192.168.200.45:7077'),
        ('spark.executor.cores', str(executors_number)),
        ('spark.cores.max', str(executors_number * workers_number)),
        ('spark.rpc.message.maxSize', '512')
    ]
    SingletonSparkSession(raw_conf)

# if 'k' in import_str and 'd' in import_str:
#     import_distkeras()

# Momentarily SuperLearner works with weka models
if 'w' in import_str:
    import core.wrappers.weka_wrapper

    core.wrappers.weka_wrapper.jvm.start()

if classifier_class == 'kc2dae':
    input_is_image = True

# For reproducibility
np.random.seed(0)

hierarchical_classifier = HierarchicalClassifier(
    input_file=input_file,
    levels_number=levels_number,
    max_level_target=max_level_target,
    level_target=level_target,
    features_number=features_number,
    packets_number=packets_number,
    classifier_class=classifier_class,
    classifier_opts=classifier_opts,
    detector_class=detector_class,
    detector_opts=detector_opts,
    workers_number=workers_number,
    anomaly_classes=anomaly_classes,
    epochs_number=epochs_number,
    arbitrary_discr=arbitrary_discr,
    n_clusters=n_clusters,
    anomaly=anomaly,
    deep=deep,
    hidden_classes=hidden_classes,
    benign_class=benign_class,
    nominal_features_index=nominal_features_index,
    optimize=optimize,
    weight_features=weight_features,
    parallelize=parallelize,
    buckets_number=buckets_number,
    unsupervised=unsupervised,
)

if config:
    config_name = config_file.split('.')[0]
    hierarchical_classifier.set_config(config_name, config)

hierarchical_classifier.init_output_files()
if input_is_image:
    hierarchical_classifier.load_image()
elif early:
    hierarchical_classifier.load_early_dataset()
else:
    hierarchical_classifier.load_dataset(memoryless)
hierarchical_classifier.kfold_validation(k=k, starting_fold=starting_fold, ending_fold=ending_fold)

if 'w' in import_str:
    import core.wrappers.weka_wrapper

    core.wrappers.weka_wrapper.jvm.stop()
