hierarchical-framework
======================

<span style="color:red">
   _   _  ____  ____  ____    ____  ____    __    __  __  ____  
  ( )_( )(_  _)( ___)(  _ \  ( ___)(  _ \  /__\  (  \/  )( ___)  
   ) _ (  _)(_  )__)  )   /   )__)  )   / /(__)\  )    (  )__)  
  (_) (_)(____)(____)(_)\_)  (__)  (_)\_)(__)(__)(_/\/\_)(____)  

</span>

## Description

Repository for the release of the code used in the paper ["A Dive into the Dark Web: Hierarchical Traffic Classification of Anonymity Tools"](https://ieeexplore.ieee.org/document/8663403).

If you use this code, please cite the following paper:

Montieri, Antonio, Domenico Ciuonzo, Giampaolo Bovenzi, Valerio Persico, and Antonio Pescapé. "_A Dive into the Dark Web: Hierarchical Traffic Classification of Anonymity Tools._" IEEE Transactions on Network Science and Engineering (2019).

## Examples

The folder examples contains a subset of Anon17 dataset and a configuration file for the hierarchical framework.

Ex1) Same configuration for each node - `python3 HierarchicalClassifierX.py -i Anon17_red.arff -n 3 -c wrf -f 74`

Ex2) Custom configuration for each node - `python3 HierarchicalClassifierX.py -i Anon17_red.arff -n 3 -o example_config.json`