reset;

# set key bottom right maxrows 2;
# set key samplen 5 spacing 10 width 25
set nokey
set mxtics 5
set mytics 5
set grid ytics;
set border 15 linewidth 2

set terminal pdf enhanced color size 10in,6in linewidth 1 dl 0.89;\
set size 1.2, 0.97

set xtics 5,5,75 nomirror font ", 40" offset -.5, -1;
set ytics font ", 40" offset 0, 0;
set xlabel font ", 45" offset 0, -3;
set ylabel font ", 45" offset -8, 0;
# set key font ", 40";

set macros
dummy="NaN title ' ' lt -3"

#set style data_wnb linespoints
set style fill transparent solid 0.15 noborder
set bmargin 8
set rmargin 32
set lmargin 19
set tmargin 1

#Scala rosso
set style line 30 lc rgb '#cd5855' lt 1 lw 2 pt 7 pi 1 ps 1

#Scala verde
set style line 38 lc rgb '#a6c761' lt 1 lw 2 pt 3 pi 1 ps 1

#Scala blu
set style line 34 lc rgb '#255187' lt 1 lw 2 pt 13 pi 1 ps 1

#Scala arancione
set style line 40 lc rgb '#cd8b55' lt 1 lw 2 pt 5 pi 1 ps 1

system "mkdir image_wnb"
system "mkdir image_wnb/plot_per_x"

##########################################################################################################
# Alle posizioni pari abbiamo medie, alle dispari le corrispondenti std_dev.                             
#
# 2) accuracy
# 4) f-measure
# 6) gmean multiclass
# 8) gmean macro
##########################################################################################################

stats 'data_wnb/metrics/join_fp/multi_flow_level_1_wnb_tag_ROOT_all_metrics.dat' using 1:4 name "L1ROOT"

stats 'data_wnb/metrics/join_fp/multi_flow_level_2_wnb_tag_I2P_all_metrics.dat' using 1:4 name "L2I2P"
stats 'data_wnb/metrics/join_fp/multi_flow_level_2_wnb_tag_JonDonym_all_metrics.dat' using 1:4 name "L2JD"
stats 'data_wnb/metrics/join_fp/multi_flow_level_2_wnb_tag_Tor_all_metrics.dat' using 1:4 name "L2TOR"

stats 'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_I2PApp_all_metrics.dat' using 1:4 name "L3I2PA"
stats 'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_I2PApp0BW_all_metrics.dat' using 1:4 name "L3I2PA0"
stats 'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_I2PApp80BW_all_metrics.dat' using 1:4 name "L3I2PA80"
stats 'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_JonDonym_all_metrics.dat' using 1:4 name "L3JD"
stats 'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_TorApp_all_metrics.dat' using 1:4 name "L3TORA"
stats 'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_TorPT_all_metrics.dat' using 1:4 name "L3TORPT"
stats 'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_Tor_all_metrics.dat' using 1:4 name "L3TOR"

set xrange [L1ROOT_min_x:L1ROOT_max_x]
set xlabel '# features'

if (L1ROOT_max_y*100+10 > 100){
     set yrange [L1ROOT_min_y*100-1:100]
}
else{
     set yrange [L1ROOT_min_y*100-1:L1ROOT_max_y*100+10]
}
set ylabel 'f-measure' offset -4, 0;

set output 'image_wnb/plot_per_x/FMeasurePerFeaturesLv1ROOT.eps';
plot 'data_wnb/metrics/join_fp/multi_flow_level_1_wnb_tag_ROOT_all_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
     'data_wnb/metrics/join_fp/multi_flow_level_1_wnb_tag_ROOT_all_metrics.dat' using 1:($4*100) with linespoints ls 30,\

if (L2I2P_max_y*100+10 > 100){
     set yrange [L2I2P_min_y*100-1:100]
}
else{
     set yrange [L2I2P_min_y*100-1:L2I2P_max_y*100+10]
}
set ylabel 'f-measure' offset -4, 0;

set output 'image_wnb/plot_per_x/FMeasurePerFeaturesLv2I2P.eps';
plot 'data_wnb/metrics/join_fp/multi_flow_level_2_wnb_tag_I2P_all_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
     'data_wnb/metrics/join_fp/multi_flow_level_2_wnb_tag_I2P_all_metrics.dat' using 1:($4*100) with linespoints ls 30,\

if (L2JD_max_y*100+10 > 100){
     set yrange [L2JD_min_y*100-1:100]
}
else{
     set yrange [L2JD_min_y*100-1:L2JD_max_y*100+10]
}
set ylabel 'f-measure' offset -4, 0;

set output 'image_wnb/plot_per_x/FMeasurePerFeaturesLv2JD.eps';
plot 'data_wnb/metrics/join_fp/multi_flow_level_2_wnb_tag_JonDonym_all_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
     'data_wnb/metrics/join_fp/multi_flow_level_2_wnb_tag_JonDonym_all_metrics.dat' using 1:($4*100) with linespoints ls 30,\

if (L2TOR_max_y*100+10 > 100){
     set yrange [L2TOR_min_y*100-1:100]
}
else{
     set yrange [L2TOR_min_y*100-1:L2TOR_max_y*100+10]
}
set ylabel 'f-measure' offset -4, 0;

set output 'image_wnb/plot_per_x/FMeasurePerFeaturesLv2TOR.eps';
plot 'data_wnb/metrics/join_fp/multi_flow_level_2_wnb_tag_Tor_all_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
     'data_wnb/metrics/join_fp/multi_flow_level_2_wnb_tag_Tor_all_metrics.dat' using 1:($4*100) with linespoints ls 30,\

if (L3I2PA_max_y*100+10 > 100){
     set yrange [L3I2PA_min_y*100-1:100]
}
else{
     set yrange [L3I2PA_min_y*100-1:L3I2PA_max_y*100+10]
}
set ylabel 'f-measure' offset -4, 0;
set yrange [50:100]
set output 'image_wnb/plot_per_x/FMeasurePerFeaturesLv3I2PA.eps';
plot 'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_I2PApp_all_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
     'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_I2PApp_all_metrics.dat' using 1:($4*100) with linespoints ls 30,\

if (L3I2PA0_max_y*100+10 > 100){
     set yrange [L3I2PA0_min_y*100-1:100]
}
else{
     set yrange [L3I2PA0_min_y*100-1:L3I2PA0_max_y*100+10]
}
set ylabel 'f-measure' offset -4, 0;

set output 'image_wnb/plot_per_x/FMeasurePerFeaturesLv3I2PA0.eps';
plot 'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_I2PApp0BW_all_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
     'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_I2PApp0BW_all_metrics.dat' using 1:($4*100) with linespoints ls 30,\

if (L3I2PA80_max_y*100+10 > 100){
     set yrange [L3I2PA80_min_y*100-1:100]
}
else{
     set yrange [L3I2PA80_min_y*100-1:L3I2PA80_max_y*100+10]
}
set ylabel 'f-measure' offset -4, 0;

set output 'image_wnb/plot_per_x/FMeasurePerFeaturesLv3I2PA80.eps';
plot 'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_I2PApp80BW_all_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
     'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_I2PApp80BW_all_metrics.dat' using 1:($4*100) with linespoints ls 30,\

if (L3JD_max_y*100+10 > 100){
     set yrange [L3JD_min_y*100-1:100]
}
else{
     set yrange [L3JD_min_y*100-1:L3JD_max_y*100+10]
}
set ylabel 'f-measure' offset -4, 0;

set output 'image_wnb/plot_per_x/FMeasurePerFeaturesLv3JD.eps';
plot 'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_JonDonym_all_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
     'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_JonDonym_all_metrics.dat' using 1:($4*100) with linespoints ls 30,\

if (L3TORA_max_y*100+10 > 100){
     set yrange [L3TORA_min_y*100-1:100]
}
else{
     set yrange [L3TORA_min_y*100-1:L3TORA_max_y*100+10]
}
set ylabel 'f-measure' offset -4, 0;
set yrange [50:100]
set output 'image_wnb/plot_per_x/FMeasurePerFeaturesLv3TORA.eps';
plot 'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_TorApp_all_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
     'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_TorApp_all_metrics.dat' using 1:($4*100) with linespoints ls 30,\

if (L3TORPT_max_y*100+10 > 100){
     set yrange [L3TORPT_min_y*100-1:100]
}
else{
     set yrange [L3TORPT_min_y*100-1:L3TORPT_max_y*100+10]
}
set ylabel 'f-measure' offset -4, 0;

set output 'image_wnb/plot_per_x/FMeasurePerFeaturesLv3TORPT.eps';
plot 'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_TorPT_all_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
     'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_TorPT_all_metrics.dat' using 1:($4*100) with linespoints ls 30,\

if (L3TOR_max_y*100+10 > 100){
     set yrange [L3TOR_min_y*100-1:100]
}
else{
     set yrange [L3TOR_min_y*100-1:L3TOR_max_y*100+10]
}
set ylabel 'f-measure' offset -4, 0;

set output 'image_wnb/plot_per_x/FMeasurePerFeaturesLv3TOR.eps';
plot 'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_Tor_all_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
     'data_wnb/metrics/join_fp/multi_flow_level_3_wnb_tag_Tor_all_metrics.dat' using 1:($4*100) with linespoints ls 30,\

### LEVEL 1 ###

## Accuracy ##

# set xrange [EL1A_min_x:EL1A_max_x]
# set xtics .1 nomirror font ", 26" offset 0, 0;
# set xlabel 'gamma' offset 0, -2;

# if (EL1A_max_y*100+10 > 100){
#      set yrange [EL1A_min_y*100-1:100]
# }
# else{
#      set yrange [EL1A_min_y*100-1:EL1A_max_y*100+10]
# }
# set ytics 1 nomirror font ", 26" offset 0, 0;
# set ylabel 'accuracy' offset -4, 0;

# set output 'image_wnb/plot_per_x/AccuracyPerGammaEarlyLv1.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_1_p_14_all_metrics.dat' using 1:($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_1_p_14_all_metrics.dat' using 1:($2*100) with linespoints ls 30 title 'Flat L1',\

## F-measure ##

# set xrange [EL1F_min_x:EL1F_max_x]

# if (EL1F_max_y*100+10 > 100){
#      set yrange [EL1F_min_y*100-1:100]
# }
# else{
#      set yrange [EL1F_min_y*100-1:EL1F_max_y*100+10]
# }
# set ylabel 'f-measure' offset -4, 0;

# set output 'image_wnb/plot_per_x/FMeasurePerGammaEarlyLv1.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_1_p_5_all_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_1_p_5_all_metrics.dat' using 1:($4*100) with linespoints ls 30 title 'Flat L1',\

## G-mean multiclass ##

# set xrange [EL1GMU_min_x:EL1GMU_max_x]

# if (EL1GMU_max_y*100+10 > 100){
#      set yrange [EL1GMU_min_y*100-1:100]
# }
# else{
#      set yrange [EL1GMU_min_y*100-1:EL1GMU_max_y*100+10]
# }
# set ylabel 'g-mean multiclass' offset -4, 0;

# set output 'image_wnb/plot_per_x/GMeanMuPerGammaEarlyLv1.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_1_p_16_all_metrics.dat' using 1:($6+(3*$7))*100:($6-(3*$7))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_1_p_16_all_metrics.dat' using 1:($6*100) with linespoints ls 30 title 'Flat L1',\

## G-mean macro ##

# set xrange [EL1GMA_min_x:EL1GMA_max_x]

# if (EL1GMA_max_y*100+10 > 100){
#      set yrange [EL1GMA_min_y*100-1:100]
# }
# else{
#      set yrange [EL1GMA_min_y*100-1:EL1GMA_max_y*100+10]
# }
# set ylabel 'g-mean macro' offset -4, 0;

# set output 'image_wnb/plot_per_x/GMeanMaPerGammaEarlyLv1.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_1_p_16_all_metrics.dat' using 1:($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_1_p_16_all_metrics.dat' using 1:($8*100) with linespoints ls 30 title 'Flat L1',\

### LEVEL 2 ###

# set ytics 5 nomirror font ", 26" offset 0, 0;

## Accuracy ##

# set xrange [EL2A_min_x:EL2A_max_x]

# if (EL2A_max_y*100+10 > 100){
#      set yrange [EL2A_min_y*100-1:100]
# }
# else{
#      set yrange [EL2A_min_y*100-1:EL2A_max_y*100+10]
# }
# set ylabel 'accuracy' offset -4, 0;

# set output 'image_wnb/plot_per_x/AccuracyPerGammaEarlyLv2.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_2_p_16_all_metrics.dat' using 1:($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_2_p_16_all_metrics.dat' using 1:($2*100) with linespoints ls 30 title 'Flat L2',\

## F-measure ##

# set xrange [EL2F_min_x:EL2F_max_x]

# if (EL2F_max_y*100+10 > 100){
#      set yrange [EL2F_min_y*100-1:100]
# }
# else{
#      set yrange [EL2F_min_y*100-1:EL2F_max_y*100+10]
# }
# set ylabel 'f-measure' offset -4, 0;

# set output 'image_wnb/plot_per_x/FMeasurePerGammaEarlyLv2.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_2_p_5_all_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_2_p_5_all_metrics.dat' using 1:($4*100) with linespoints ls 30 title 'Flat L2',\

## G-mean multiclass ##

# set xrange [EL2GMU_min_x:EL2GMU_max_x]

# if (EL2GMU_max_y*100+10 > 100){
#      set yrange [EL2GMU_min_y*100-1:100]
# }
# else{
#      set yrange [EL2GMU_min_y*100-1:EL2GMU_max_y*100+10]
# }
# set ylabel 'g-mean multiclass' offset -4, 0;

# set output 'image_wnb/plot_per_x/GMeanMuPerGammaEarlyLv2.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_2_p_16_all_metrics.dat' using 1:($6+(3*$7))*100:($6-(3*$7))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_2_p_16_all_metrics.dat' using 1:($6*100) with linespoints ls 30 title 'Flat L2',\

## G-mean macro ##

# set xrange [EL2GMA_min_x:EL2GMA_max_x]

# if (EL2GMA_max_y*100+10 > 100){
#      set yrange [EL2GMA_min_y*100-1:100]
# }
# else{
#      set yrange [EL2GMA_min_y*100-1:EL2GMA_max_y*100+10]
# }
# set ylabel 'g-mean macro' offset -4, 0;

# set output 'image_wnb/plot_per_x/GMeanMaPerGammaEarlyLv2.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_2_p_16_all_metrics.dat' using 1:($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_2_p_16_all_metrics.dat' using 1:($8*100) with linespoints ls 30 title 'Flat L2',\

### LEVEL 3 ###

# set ytics 8 nomirror font ", 26" offset 0, 0;

## Accuracy ##

# set xrange [EL3A_min_x:EL3A_max_x]

# if (EL3A_max_y*100+10 > 100){
#      set yrange [EL3A_min_y*100-1:100]
# }
# else{
#      set yrange [EL3A_min_y*100-1:EL3A_max_y*100+10]
# }
# set ylabel 'accuracy' offset -4, 0;

# set output 'image_wnb/plot_per_x/AccuracyPerGammaEarlyLv3.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_3_p_6_all_metrics.dat' using 1:($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_3_p_6_all_metrics.dat' using 1:($2*100) with linespoints ls 30 title 'Flat L3',\

## F-measure ##

# set xrange [EL3F_min_x:EL3F_max_x]

# if (EL3F_max_y*100+10 > 100){
#      set yrange [EL3F_min_y*100-1:100]
# }
# else{
#      set yrange [EL3F_min_y*100-1:EL3F_max_y*100+10]
# }
# set ylabel 'f-measure' offset -4, 0;

# set output 'image_wnb/plot_per_x/FMeasurePerGammaEarlyLv3.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_3_p_5_all_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_3_p_5_all_metrics.dat' using 1:($4*100) with linespoints ls 30 title 'Flat L3',\

## G-mean multiclass ##

# set xrange [EL3GMU_min_x:EL3GMU_max_x]

# if (EL3GMU_max_y*100+10 > 100){
#      set yrange [EL3GMU_min_y*100-1:100]
# }
# else{
#      set yrange [EL3GMU_min_y*100-1:EL3GMU_max_y*100+10]
# }
# set ylabel 'g-mean multiclass' offset -4, 0;

# set output 'image_wnb/plot_per_x/GMeanMuPerGammaEarlyLv3.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_3_p_16_all_metrics.dat' using 1:($6+(3*$7))*100:($6-(3*$7))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_3_p_16_all_metrics.dat' using 1:($6*100) with linespoints ls 30 title 'Flat L3',\

## G-mean macro ##

# set xrange [EL3GMA_min_x:EL3GMA_max_x]

# if (EL3GMA_max_y*100+10 > 100){
#      set yrange [EL3GMA_min_y*100-1:100]
# }
# else{
#      set yrange [EL3GMA_min_y*100-1:EL3GMA_max_y*100+10]
# }
# set ylabel 'g-mean macro' offset -4, 0;

# set output 'image_wnb/plot_per_x/GMeanMaPerGammaEarlyLv3.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_3_p_16_all_metrics.dat' using 1:($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_3_p_16_all_metrics.dat' using 1:($8*100) with linespoints ls 30 title 'Flat L3',\


### LEVEL 1 ###

## Accuracy ##

# set xrange [FL1A_min_x:FL1A_max_x]
# set xtics .1 nomirror font ", 26" offset 0, 0;
# set xlabel 'gamma' offset 0, -2;

# if (FL1A_max_y*100+10 > 100){
#      set yrange [FL1A_min_y*100-1:100]
# }
# else{
#      set yrange [FL1A_min_y*100-1:FL1A_max_y*100+10]
# }
# set ytics 1 nomirror font ", 26" offset 0, 0;
# set ylabel 'accuracy' offset -4, 0;

# set output 'image_wnb/plot_per_x/AccuracyPerGammaFlowLv1.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_1_f_61_all_metrics.dat' using 1:($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_1_f_61_all_metrics.dat' using 1:($2*100) with linespoints ls 30 title 'Flat L1',\

## F-measure ##

# set xrange [FL1F_min_x:FL1F_max_x]

# if (FL1F_max_y*100+10 > 100){
#      set yrange [FL1F_min_y*100-1:100]
# }
# else{
#      set yrange [FL1F_min_y*100-1:FL1F_max_y*100+10]
# }
# set ylabel 'f-measure' offset -4, 0;

# set output 'image_wnb/plot_per_x/FMeasurePerGammaFlowLv1.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_1_f_61_all_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_1_f_61_all_metrics.dat' using 1:($4*100) with linespoints ls 30 title 'Flat L1',\

# ## G-mean multiclass ##

# set xrange [FL1GMU_min_x:FL1GMU_max_x]

# if (FL1GMU_max_y*100+10 > 100){
#      set yrange [FL1GMU_min_y*100-1:100]
# }
# else{
#      set yrange [FL1GMU_min_y*100-1:FL1GMU_max_y*100+10]
# }
# set ylabel 'g-mean multiclass' offset -4, 0;

# set output 'image_wnb/plot_per_x/GMeanMuPerGammaFlowLv1.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_1_f_77_all_metrics.dat' using 1:($6+(3*$7))*100:($6-(3*$7))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_1_f_77_all_metrics.dat' using 1:($6*100) with linespoints ls 30 title 'Flat L1',\

# ## G-mean macro ##

# set xrange [FL1GMA_min_x:FL1GMA_max_x]

# if (FL1GMA_max_y*100+10 > 100){
#      set yrange [FL1GMA_min_y*100-1:100]
# }
# else{
#      set yrange [FL1GMA_min_y*100-1:FL1GMA_max_y*100+10]
# }
# set ylabel 'g-mean macro' offset -4, 0;

# set output 'image_wnb/plot_per_x/GMeanMaPerGammaFlowLv1.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_1_f_77_all_metrics.dat' using 1:($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_1_f_77_all_metrics.dat' using 1:($8*100) with linespoints ls 30 title 'Flat L1',\

# ### LEVEL 2 ###

# set ytics 5 nomirror font ", 26" offset 0, 0;

# ## Accuracy ##

# set xrange [FL2A_min_x:FL2A_max_x]

# if (FL2A_max_y*100+10 > 100){
#      set yrange [FL2A_min_y*100-1:100]
# }
# else{
#      set yrange [FL2A_min_y*100-1:FL2A_max_y*100+10]
# }
# set ylabel 'accuracy' offset -4, 0;

# set output 'image_wnb/plot_per_x/AccuracyPerGammaFlowLv2.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_2_f_53_all_metrics.dat' using 1:($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_2_f_53_all_metrics.dat' using 1:($2*100) with linespoints ls 30 title 'Flat L2',\

# ## F-measure ##

# set xrange [FL2F_min_x:FL2F_max_x]

# if (FL2F_max_y*100+10 > 100){
#      set yrange [FL2F_min_y*100-1:100]
# }
# else{
#      set yrange [FL2F_min_y*100-1:FL2F_max_y*100+10]
# }
# set ylabel 'f-measure' offset -4, 0;

# set output 'image_wnb/plot_per_x/FMeasurePerGammaFlowLv2.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_2_f_53_all_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_2_f_53_all_metrics.dat' using 1:($4*100) with linespoints ls 30 title 'Flat L2',\

# ## G-mean multiclass ##

# set xrange [FL2GMU_min_x:FL2GMU_max_x]

# if (FL2GMU_max_y*100+10 > 100){
#      set yrange [FL2GMU_min_y*100-1:100]
# }
# else{
#      set yrange [FL2GMU_min_y*100-1:FL2GMU_max_y*100+10]
# }
# set ylabel 'g-mean multiclass' offset -4, 0;

# set output 'image_wnb/plot_per_x/GMeanMuPerGammaFlowLv2.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_2_f_77_all_metrics.dat' using 1:($6+(3*$7))*100:($6-(3*$7))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_2_f_77_all_metrics.dat' using 1:($6*100) with linespoints ls 30 title 'Flat L2',\

# ## G-mean macro ##

# set xrange [FL3GMA_min_x:FL3GMA_max_x]

# if (FL3GMA_max_y*100+10 > 100){
#      set yrange [FL3GMA_min_y*100-1:100]
# }
# else{
#      set yrange [FL3GMA_min_y*100-1:FL3GMA_max_y*100+10]
# }
# set ylabel 'g-mean macro' offset -4, 0;

# set output 'image_wnb/plot_per_x/GMeanMaPerGammaFlowLv2.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_2_f_77_all_metrics.dat' using 1:($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_2_f_77_all_metrics.dat' using 1:($8*100) with linespoints ls 30 title 'Flat L2',\

# ### LEVEL 3 ###

# set ytics 8 nomirror font ", 26" offset 0, 0;

# ## Accuracy ##

# set xrange [FL3A_min_x:FL3A_max_x]

# if (FL3A_max_y*100+10 > 100){
#      set yrange [FL3A_min_y*100-1:100]
# }
# else{
#      set yrange [FL3A_min_y*100-1:FL3A_max_y*100+10]
# }
# set ylabel 'accuracy' offset -4, 0;

# set output 'image_wnb/plot_per_x/AccuracyPerGammaFlowLv3.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_3_f_53_all_metrics.dat' using 1:($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_3_f_53_all_metrics.dat' using 1:($2*100) with linespoints ls 30 title 'Flat L3',\

# ## F-measure ##

# set xrange [FL3F_min_x:FL3F_max_x]

# if (FL3F_max_y*100+10 > 100){
#      set yrange [FL3F_min_y*100-1:100]
# }
# else{
#      set yrange [FL3F_min_y*100-1:FL3F_max_y*100+10]
# }
# set ylabel 'f-measure' offset -4, 0;

# set output 'image_wnb/plot_per_x/FMeasurePerGammaFlowLv3.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_3_f_53_all_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_3_f_53_all_metrics.dat' using 1:($4*100) with linespoints ls 30 title 'Flat L3',\

# ## G-mean multiclass ##

# set xrange [FL3GMU_min_x:FL3GMU_max_x]

# if (FL3GMU_max_y*100+10 > 100){
#      set yrange [FL3GMU_min_y*100-1:100]
# }
# else{
#      set yrange [FL3GMU_min_y*100-1:FL3GMU_max_y*100+10]
# }
# set ylabel 'g-mean multiclass' offset -4, 0;

# set output 'image_wnb/plot_per_x/GMeanMuPerGammaFlowLv3.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_3_f_77_all_metrics.dat' using 1:($6+(3*$7))*100:($6-(3*$7))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_3_f_77_all_metrics.dat' using 1:($6*100) with linespoints ls 30 title 'Flat L3',\

# ## G-mean macro ##

# set xrange [FL3GMA_min_x:FL3GMA_max_x]

# if (FL3GMA_max_y*100+10 > 100){
#      set yrange [FL3GMA_min_y*100-1:100]
# }
# else{
#      set yrange [FL3GMA_min_y*100-1:FL3GMA_max_y*100+10]
# }
# set ylabel 'g-mean macro' offset -4, 0;

# set output 'image_wnb/plot_per_x/GMeanMaPerGammaFlowLv3.eps';
# plot 'data_wnb/metrics/join_fp/flat_flow_level_3_f_77_all_metrics.dat' using 1:($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 30 notitle,\
#      'data_wnb/metrics/join_fp/flat_flow_level_3_f_77_all_metrics.dat' using 1:($8*100) with linespoints ls 30 title 'Flat L3',\