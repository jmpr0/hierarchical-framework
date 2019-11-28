HierFrame
======================

<pre>
   _   _  ____  ____  ____    ____  ____    __    __  __  ____  
  ( )_( )(_  _)( ___)(  _ \  ( ___)(  _ \  /__\  (  \/  )( ___)  
   ) _ (  _)(_  )__)  )   /   )__)  )   / /(__)\  )    (  )__)  
  (_) (_)(____)(____)(_)\_)  (__)  (_)\_)(__)(__)(_/\/\_)(____)  

</pre>

## Description

The Hierarchical Framework (HierFrame) for Traffic Classification borns as publicly available implementation of the proposal paper ["A Dive into the Dark Web: Hierarchical Traffic Classification of Anonymity Tools"](https://ieeexplore.ieee.org/document/8663403).

HierFrame is an active project where new features are constantly introduced, as the enabling to BigData frameworks or the support to a wide range of Machine Learning and Deep Learning models.

HierFrame is used to conduct experimental analysis on the following under review paper:

Bovenzi, Giampaolo, Giuseppe Aceto, Domenico Ciuonzo, Valerio Persico, and Antonio Pescapé. "_Double Parallelism in Traffic Classification: Big Data-enabled Hierarchical (BDeH) Framework._" under revision at IEEE Transactions on Network Science and Engineering (2019).

If you use this code, please cite the following paper:

Montieri, Antonio, Domenico Ciuonzo, Giampaolo Bovenzi, Valerio Persico, and Antonio Pescapé. "_A Dive into the Dark Web: Hierarchical Traffic Classification of Anonymity Tools._" IEEE Transactions on Network Science and Engineering (2019).

## Examples

The folder examples contains a subset of Anon17 dataset and a configuration file for the hierarchical framework.

Ex1) Same configuration for each node - `python3 HierarchicalClassifierX.py -i Anon17_red.arff -n 3 -c wrf -f 74`

Ex2) Custom configuration for each node - `python3 HierarchicalClassifierX.py -i Anon17_red.arff -n 3 -o example_config.json`

Input description

-i Dataset file name (can be ARFF, CSV, or PICKLE (custom))  
-n Hierarchy depth (number of layers of labels)

-c Classifier model  
-f Number of features (if less then the total number of features, script applies per-node feature selection)

-o Configuration JSON file
