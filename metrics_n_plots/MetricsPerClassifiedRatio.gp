reset;

set nokey
set mxtics 5
set mytics 5
set grid ytics;
set border 15 linewidth 2

set terminal pdf enhanced color size 10in,6in linewidth 1 dl 0.89;\
set size 1.17, 0.97

set xtics nomirror font ", 35" offset -.5, -1;
set ytics font ", 35" offset 0, 0;
set xlabel font ", 40" offset 0, -2;
set ylabel font ", 40" offset -6, 0;

set macros
dummy="NaN title ' ' lt -3"

#set style data linespoints
set style fill transparent solid 0.15 noborder
set bmargin 7
set rmargin 28
set lmargin 17
set tmargin 1

#Scala rosso
set style line 30 lc rgb '#cd5855' lt 1 lw 2 pt 7 pi 1 ps 1

#Scala verde
set style line 38 lc rgb '#a6c761' lt 1 lw 2 pt 3 pi 1 ps 1

#Scala blu
set style line 34 lc rgb '#255187' lt 1 lw 2 pt 13 pi 1 ps 1

#Scala arancione
set style line 40 lc rgb '#cd8b55' lt 1 lw 2 pt 5 pi 1 ps 1

system "mkdir image"
system "mkdir image/plot_per_cr"

##########################################################################################################
# Alle posizioni pari abbiamo medie, alle dispari le corrispondenti std_dev.               
#
# 2) accuracy
# 4) f-measure
# 6) gmean multiclass
# 8) gmean macro
# 10) classified ratio
##########################################################################################################

# print 'data/metrics_per_cr/flat_early_level_1_p_x_metrics_per_cr.dat'
# stats 'data/metrics_per_cr/flat_early_level_1_p_14_metrics_per_cr.dat' using ($1*100):2 name "EL1A"
# stats 'data/metrics_per_cr/flat_early_level_1_p_5_metrics_per_cr.dat' using ($1*100):4 name "EL1F"
# stats 'data/metrics_per_cr/flat_early_level_1_p_9_metrics_per_cr.dat' using ($1*100):6 name "EL1GMU"
# stats 'data/metrics_per_cr/flat_early_level_1_p_9_metrics_per_cr.dat' using ($1*100):8 name "EL1GMA"

# print 'data/metrics_per_cr/flat_early_level_2_p_x_metrics_per_cr.dat'
# stats 'data/metrics_per_cr/flat_early_level_2_p_16_metrics_per_cr.dat' using ($1*100):2 name "EL2A"
# stats 'data/metrics_per_cr/flat_early_level_2_p_4_metrics_per_cr.dat' using ($1*100):4 name "EL2F"
# stats 'data/metrics_per_cr/flat_early_level_2_p_9_metrics_per_cr.dat' using ($1*100):6 name "EL2GMU"
# stats 'data/metrics_per_cr/flat_early_level_2_p_9_metrics_per_cr.dat' using ($1*100):8 name "EL2GMA"

# print 'data/metrics_per_cr/flat_early_level_3_p_x_metrics_per_cr.dat'
# stats 'data/metrics_per_cr/flat_early_level_3_p_6_metrics_per_cr.dat' using ($1*100):2 name "EL3A"
# stats 'data/metrics_per_cr/flat_early_level_3_p_5_metrics_per_cr.dat' using ($1*100):4 name "EL3F"
# stats 'data/metrics_per_cr/flat_early_level_3_p_9_metrics_per_cr.dat' using ($1*100):6 name "EL3GMU"
# stats 'data/metrics_per_cr/flat_early_level_3_p_9_metrics_per_cr.dat' using ($1*100):8 name "EL3GMA"

print 'data/metrics_per_cr/flat_flow_level_1_f_x_metrics_per_cr.dat'
stats 'data/metrics_per_cr/flat_flow_level_1_f_61_metrics_per_cr.dat' using ($1*100):2 name "FL1A"
stats 'data/metrics_per_cr/flat_flow_level_1_f_61_metrics_per_cr.dat' using ($1*100):4 name "FL1F"
stats 'data/metrics_per_cr/flat_flow_level_1_f_53_metrics_per_cr.dat' using ($1*100):6 name "FL1GMU"
stats 'data/metrics_per_cr/flat_flow_level_1_f_53_metrics_per_cr.dat' using ($1*100):8 name "FL1GMA"

print 'data/metrics_per_cr/flat_flow_level_2_f_x_metrics_per_cr.dat'
stats 'data/metrics_per_cr/flat_flow_level_2_f_53_metrics_per_cr.dat' using ($1*100):2 name "FL2A"
stats 'data/metrics_per_cr/flat_flow_level_2_f_53_metrics_per_cr.dat' using ($1*100):4 name "FL2F"
stats 'data/metrics_per_cr/flat_flow_level_2_f_53_metrics_per_cr.dat' using ($1*100):6 name "FL2GMU"
stats 'data/metrics_per_cr/flat_flow_level_2_f_53_metrics_per_cr.dat' using ($1*100):8 name "FL2GMA"

print 'data/metrics_per_cr/flat_flow_level_3_f_x_metrics_per_cr.dat'
stats 'data/metrics_per_cr/flat_flow_level_3_f_53_metrics_per_cr.dat' using ($1*100):2 name "FL3A"
stats 'data/metrics_per_cr/flat_flow_level_3_f_53_metrics_per_cr.dat' using ($1*100):4 name "FL3F"
stats 'data/metrics_per_cr/flat_flow_level_3_f_53_metrics_per_cr.dat' using ($1*100):6 name "FL3GMU"
stats 'data/metrics_per_cr/flat_flow_level_3_f_53_metrics_per_cr.dat' using ($1*100):8 name "FL3GMA"

### LEVEL 1 ###

## Accuracy ##

# set xrange [EL1A_min_x:EL1A_max_x]
# set xlabel 'classified ratio';

# if (EL1A_max_y*100+10 > 100){
#    set yrange [EL1A_min_y*100-1:100]
# }
# else{
#    set yrange [EL1A_min_y*100-1:EL1A_max_y*100+10]
# }
# set ylabel 'accuracy [%]';

# set output 'image/plot_per_cr/AccuracyEarlyPerCrLv1.eps';
# plot 'data/metrics_per_cr/flat_early_level_1_p_14_metrics_per_cr.dat' using ($1*100):($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 30 notitle,\
#    'data/metrics_per_cr/flat_early_level_1_p_14_metrics_per_cr.dat' using ($1*100):($2*100) with linespoints ls 30 title 'FlatL1',\

# ## F-measure ##

# set xrange [EL1F_min_x:EL1F_max_x]

# if (EL1F_max_y*100+10 > 100){
#    set yrange [EL1F_min_y*100-1:100]
# }
# else{
#    set yrange [EL1F_min_y*100-1:EL1F_max_y*100+10]
# }
# set ylabel 'f-measure [%]';

# set output 'image/plot_per_cr/FMeasureEarlyPerCrLv1.eps';
# plot 'data/metrics_per_cr/flat_early_level_1_p_5_metrics_per_cr.dat' using ($1*100):($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
#    'data/metrics_per_cr/flat_early_level_1_p_5_metrics_per_cr.dat' using ($1*100):($4*100) with linespoints ls 30 title 'FlatL1',\

# ## G-mean multiclass ##

# set xrange [EL1GMU_min_x:EL1GMU_max_x]

# if (EL1GMU_max_y*100+10 > 100){
#    set yrange [EL1GMU_min_y*100-1:100]
# }
# else{
#    set yrange [EL1GMU_min_y*100-1:EL1GMU_max_y*100+10]
# }
# set ylabel 'g-mean multiclass [%]';

# set output 'image/plot_per_cr/GMeanMuEarlyPerCrLv1.eps';
# plot 'data/metrics_per_cr/flat_early_level_1_p_9_metrics_per_cr.dat' using ($1*100):($6+(3*$7))*100:($6-(3*$7))*100 with filledcurves ls 30 notitle,\
#    'data/metrics_per_cr/flat_early_level_1_p_9_metrics_per_cr.dat' using ($1*100):($6*100) with linespoints ls 30 title 'FlatL1',\

# ## G-mean macro ##

# set xrange [EL1GMA_min_x:EL1GMA_max_x]

# if (EL1GMA_max_y*100+10 > 100){
#    set yrange [EL1GMA_min_y*100-1:100]
# }
# else{
#    set yrange [EL1GMA_min_y*100-1:EL1GMA_max_y*100+10]
# }
# set ylabel 'g-mean macro [%]';

# set output 'image/plot_per_cr/GMeanMaEarlyPerCrLv1.eps';
# plot 'data/metrics_per_cr/flat_early_level_1_p_9_metrics_per_cr.dat' using ($1*100):($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 30 notitle,\
#    'data/metrics_per_cr/flat_early_level_1_p_9_metrics_per_cr.dat' using ($1*100):($8*100) with linespoints ls 30 title 'FlatL1',\

# ### LEVEL 2 ###

# ## Accuracy ##

# set xrange [EL2A_min_x:EL2A_max_x]

# if (EL2A_max_y*100+10 > 100){
#    set yrange [EL2A_min_y*100-1:100]
# }
# else{
#    set yrange [EL2A_min_y*100-1:EL2A_max_y*100+10]
# }
# set ylabel 'accuracy [%]';

# set output 'image/plot_per_cr/AccuracyEarlyPerCrLv2.eps';
# plot 'data/metrics_per_cr/flat_early_level_2_p_16_metrics_per_cr.dat' using ($1*100):($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 30 notitle,\
#    'data/metrics_per_cr/flat_early_level_2_p_16_metrics_per_cr.dat' using ($1*100):($2*100) with linespoints ls 30 title 'FlatL2',\

# ## F-measure ##

# set xrange [EL2F_min_x:EL2F_max_x]

# if (EL2F_max_y*100+10 > 100){
#    set yrange [EL2F_min_y*100-1:100]
# }
# else{
#    set yrange [EL2F_min_y*100-1:EL2F_max_y*100+10]
# }
# set ylabel 'f-measure [%]';

# set output 'image/plot_per_cr/FMeasureEarlyPerCrLv2.eps';
# plot 'data/metrics_per_cr/flat_early_level_2_p_5_metrics_per_cr.dat' using ($1*100):($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
#    'data/metrics_per_cr/flat_early_level_2_p_5_metrics_per_cr.dat' using ($1*100):($4*100) with linespoints ls 30 title 'FlatL2',\

# ## G-mean multiclass ##

# set xrange [EL2GMU_min_x:EL2GMU_max_x]

# if (EL2GMU_max_y*100+10 > 100){
#    set yrange [EL2GMU_min_y*100-1:100]
# }
# else{
#    set yrange [EL2GMU_min_y*100-1:EL2GMU_max_y*100+10]
# }
# set ylabel 'g-mean multiclass [%]';

# set output 'image/plot_per_cr/GMeanMuEarlyPerCrLv2.eps';
# plot 'data/metrics_per_cr/flat_early_level_2_p_9_metrics_per_cr.dat' using ($1*100):($6+(3*$7))*100:($6-(3*$7))*100 with filledcurves ls 30 notitle,\
#    'data/metrics_per_cr/flat_early_level_2_p_9_metrics_per_cr.dat' using ($1*100):($6*100) with linespoints ls 30 title 'FlatL2',\

# ## G-mean macro ##

# set xrange [EL2GMA_min_x:EL2GMA_max_x]

# if (EL2GMA_max_y*100+10 > 100){
#    set yrange [EL2GMA_min_y*100-1:100]
# }
# else{
#    set yrange [EL2GMA_min_y*100-1:EL2GMA_max_y*100+10]
# }
# set ylabel 'g-mean macro [%]';

# set output 'image/plot_per_cr/GMeanMaEarlyPerCrLv2.eps';
# plot 'data/metrics_per_cr/flat_early_level_2_p_9_metrics_per_cr.dat' using ($1*100):($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 30 notitle,\
#    'data/metrics_per_cr/flat_early_level_2_p_9_metrics_per_cr.dat' using ($1*100):($8*100) with linespoints ls 30 title 'FlatL2',\

# ### LEVEL 3 ###

# ## Accuracy ##

# set xrange [EL3A_min_x:EL3A_max_x]

# if (EL3A_max_y*100+10 > 100){
#    set yrange [EL3A_min_y*100-1:100]
# }
# else{
#    set yrange [EL3A_min_y*100-1:EL3A_max_y*100+10]
# }
# set ylabel 'accuracy [%]';

# set output 'image/plot_per_cr/AccuracyEarlyPerCrLv3.eps';
# plot 'data/metrics_per_cr/flat_early_level_3_p_6_metrics_per_cr.dat' using ($1*100):($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 30 notitle,\
#    'data/metrics_per_cr/flat_early_level_3_p_6_metrics_per_cr.dat' using ($1*100):($2*100) with linespoints ls 30 title 'FlatL3',\

# ## F-measure ##

# set xrange [EL3F_min_x:EL3F_max_x]

# if (EL3F_max_y*100+10 > 100){
#    set yrange [EL3F_min_y*100-1:100]
# }
# else{
#    set yrange [EL3F_min_y*100-1:EL3F_max_y*100+10]
# }
# set ylabel 'f-measure [%]';

# set output 'image/plot_per_cr/FMeasureEarlyPerCrLv3.eps';
# plot 'data/metrics_per_cr/flat_early_level_3_p_5_metrics_per_cr.dat' using ($1*100):($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
#    'data/metrics_per_cr/flat_early_level_3_p_5_metrics_per_cr.dat' using ($1*100):($4*100) with linespoints ls 30 title 'FlatL3',\

# ## G-mean multiclass ##

# set xrange [EL3GMU_min_x:EL3GMU_max_x]

# if (EL3GMU_max_y*100+10 > 100){
#    set yrange [EL3GMU_min_y*100-1:100]
# }
# else{
#    set yrange [EL3GMU_min_y*100-1:EL3GMU_max_y*100+10]
# }
# set ylabel 'g-mean multiclass [%]';

# set output 'image/plot_per_cr/GMeanMuEarlyPerCrLv3.eps';
# plot 'data/metrics_per_cr/flat_early_level_3_p_9_metrics_per_cr.dat' using ($1*100):($6+(3*$7))*100:($6-(3*$7))*100 with filledcurves ls 30 notitle,\
#    'data/metrics_per_cr/flat_early_level_3_p_9_metrics_per_cr.dat' using ($1*100):($6*100) with linespoints ls 30 title 'FlatL3',\

# ## G-mean macro ##

# set xrange [EL3GMA_min_x:EL3GMA_max_x]

# if (EL3GMA_max_y*100+10 > 100){
#    set yrange [EL3GMA_min_y*100-1:100]
# }
# else{
#    set yrange [EL3GMA_min_y*100-1:EL3GMA_max_y*100+10]
# }
# set ylabel 'g-mean macro [%]';

# set output 'image/plot_per_cr/GMeanMaEarlyPerCrLv3.eps';
# plot 'data/metrics_per_cr/flat_early_level_3_p_9_metrics_per_cr.dat' using ($1*100):($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 30 notitle,\
#    'data/metrics_per_cr/flat_early_level_3_p_9_metrics_per_cr.dat' using ($1*100):($8*100) with linespoints ls 30 title 'FlatL3',\

### LEVEL 1 ###

## Accuracy ##

set xrange [FL1A_min_x:FL1A_max_x]
set xlabel 'classified ratio';

if (FL1A_max_y*100+10 > 100){
   set yrange [FL1A_min_y*100-1:100]
}
else{
   set yrange [FL1A_min_y*100-1:FL1A_max_y*100+10]
}
set ylabel 'accuracy [%]';

set output 'image/plot_per_cr/AccuracyFlowPerCrLv1.eps';
plot 'data/metrics_per_cr/flat_flow_level_1_f_61_metrics_per_cr.dat' using ($1*100):($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 30 notitle,\
   'data/metrics_per_cr/flat_flow_level_1_f_61_metrics_per_cr.dat' using ($1*100):($2*100) with linespoints ls 30 title 'FlatL1',\

## F-measure ##

set xrange [FL1F_min_x:FL1F_max_x]

if (FL1F_max_y*100+10 > 100){
   set yrange [FL1F_min_y*100-1:100]
}
else{
   set yrange [FL1F_min_y*100-1:FL1F_max_y*100+10]
}
set ylabel 'f-measure [%]';

set output 'image/plot_per_cr/FMeasureFlowPerCrLv1.eps';
plot 'data/metrics_per_cr/flat_flow_level_1_f_61_metrics_per_cr.dat' using ($1*100):($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
   'data/metrics_per_cr/flat_flow_level_1_f_61_metrics_per_cr.dat' using ($1*100):($4*100) with linespoints ls 30 title 'FlatL1',\

## G-mean multiclass ##

set xrange [FL1GMU_min_x:FL1GMU_max_x]

if (FL1GMU_max_y*100+10 > 100){
   set yrange [FL1GMU_min_y*100-1:100]
}
else{
   set yrange [FL1GMU_min_y*100-1:FL1GMU_max_y*100+10]
}
set ylabel 'g-mean multiclass [%]';

set output 'image/plot_per_cr/GMeanMuFlowPerCrLv1.eps';
plot 'data/metrics_per_cr/flat_flow_level_1_f_53_metrics_per_cr.dat' using ($1*100):($6+(3*$7))*100:($6-(3*$7))*100 with filledcurves ls 30 notitle,\
   'data/metrics_per_cr/flat_flow_level_1_f_53_metrics_per_cr.dat' using ($1*100):($6*100) with linespoints ls 30 title 'FlatL1',\

## G-mean macro ##

set xrange [FL1GMA_min_x:FL1GMA_max_x]

if (FL1GMA_max_y*100+10 > 100){
   set yrange [FL1GMA_min_y*100-1:100]
}
else{
   set yrange [FL1GMA_min_y*100-1:FL1GMA_max_y*100+10]
}
set ylabel 'g-mean macro [%]';

set output 'image/plot_per_cr/GMeanMaFlowPerCrLv1.eps';
plot 'data/metrics_per_cr/flat_flow_level_1_f_53_metrics_per_cr.dat' using ($1*100):($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 30 notitle,\
   'data/metrics_per_cr/flat_flow_level_1_f_53_metrics_per_cr.dat' using ($1*100):($8*100) with linespoints ls 30 title 'FlatL1',\

### LEVEL 2 ###

## Accuracy ##

set xrange [FL2A_min_x:FL2A_max_x]

if (FL2A_max_y*100+10 > 100){
   set yrange [FL2A_min_y*100-1:100]
}
else{
   set yrange [FL2A_min_y*100-1:FL2A_max_y*100+10]
}
set ylabel 'accuracy [%]';

set output 'image/plot_per_cr/AccuracyFlowPerCrLv2.eps';
plot 'data/metrics_per_cr/flat_flow_level_2_f_53_metrics_per_cr.dat' using ($1*100):($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 30 notitle,\
   'data/metrics_per_cr/flat_flow_level_2_f_53_metrics_per_cr.dat' using ($1*100):($2*100) with linespoints ls 30 title 'FlatL2',\

## F-measure ##

set xrange [FL2F_min_x:FL2F_max_x]

if (FL2F_max_y*100+10 > 100){
   set yrange [FL2F_min_y*100-1:100]
}
else{
   set yrange [FL2F_min_y*100-1:FL2F_max_y*100+10]
}
set ylabel 'f-measure [%]';

set output 'image/plot_per_cr/FMeasureFlowPerCrLv2.eps';
plot 'data/metrics_per_cr/flat_flow_level_2_f_53_metrics_per_cr.dat' using ($1*100):($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
   'data/metrics_per_cr/flat_flow_level_2_f_53_metrics_per_cr.dat' using ($1*100):($4*100) with linespoints ls 30 title 'FlatL2',\

## G-mean multiclass ##

set xrange [FL2GMU_min_x:FL2GMU_max_x]

if (FL2GMU_max_y*100+10 > 100){
   set yrange [FL2GMU_min_y*100-1:100]
}
else{
   set yrange [FL2GMU_min_y*100-1:FL2GMU_max_y*100+10]
}
set ylabel 'g-mean multiclass [%]';

set output 'image/plot_per_cr/GMeanMuFlowPerCrLv2.eps';
plot 'data/metrics_per_cr/flat_flow_level_2_f_53_metrics_per_cr.dat' using ($1*100):($6+(3*$7))*100:($6-(3*$7))*100 with filledcurves ls 30 notitle,\
   'data/metrics_per_cr/flat_flow_level_2_f_53_metrics_per_cr.dat' using ($1*100):($6*100) with linespoints ls 30 title 'FlatL2',\

## G-mean macro ##

set xrange [FL2GMA_min_x:FL2GMA_max_x]

if (FL2GMA_max_y*100+10 > 100){
   set yrange [FL2GMA_min_y*100-1:100]
}
else{
   set yrange [FL2GMA_min_y*100-1:FL2GMA_max_y*100+10]
}
set ylabel 'g-mean macro [%]';

set output 'image/plot_per_cr/GMeanMaFlowPerCrLv2.eps';
plot 'data/metrics_per_cr/flat_flow_level_2_f_53_metrics_per_cr.dat' using ($1*100):($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 30 notitle,\
   'data/metrics_per_cr/flat_flow_level_2_f_53_metrics_per_cr.dat' using ($1*100):($8*100) with linespoints ls 30 title 'FlatL2',\

### LEVEL 3 ###

## Accuracy ##

set xrange [FL3A_min_x:FL3A_max_x]

if (FL3A_max_y*100+10 > 100){
   set yrange [FL3A_min_y*100-1:100]
}
else{
   set yrange [FL3A_min_y*100-1:FL3A_max_y*100+10]
}
set ylabel 'accuracy [%]';

set output 'image/plot_per_cr/AccuracyFlowPerCrLv3.eps';
plot 'data/metrics_per_cr/flat_flow_level_3_f_53_metrics_per_cr.dat' using ($1*100):($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 30 notitle,\
   'data/metrics_per_cr/flat_flow_level_3_f_53_metrics_per_cr.dat' using ($1*100):($2*100) with linespoints ls 30 title 'FlatL3',\

## F-measure ##

set xrange [FL3F_min_x:FL3F_max_x]

if (FL3F_max_y*100+10 > 100){
   set yrange [FL3F_min_y*100-1:100]
}
else{
   set yrange [FL3F_min_y*100-1:FL3F_max_y*100+10]
}
set ylabel 'f-measure [%]';

set output 'image/plot_per_cr/FMeasureFlowPerCrLv3.eps';
plot 'data/metrics_per_cr/flat_flow_level_3_f_53_metrics_per_cr.dat' using ($1*100):($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
   'data/metrics_per_cr/flat_flow_level_3_f_53_metrics_per_cr.dat' using ($1*100):($4*100) with linespoints ls 30 title 'FlatL3',\

## G-mean multiclass ##

set xrange [FL3GMU_min_x:FL3GMU_max_x]

if (FL3GMU_max_y*100+10 > 100){
   set yrange [FL3GMU_min_y*100-1:100]
}
else{
   set yrange [FL3GMU_min_y*100-1:FL3GMU_max_y*100+10]
}
set ylabel 'g-mean multiclass [%]';

set output 'image/plot_per_cr/GMeanMuFlowPerCrLv3.eps';
plot 'data/metrics_per_cr/flat_flow_level_3_f_53_metrics_per_cr.dat' using ($1*100):($6+(3*$7))*100:($6-(3*$7))*100 with filledcurves ls 30 notitle,\
   'data/metrics_per_cr/flat_flow_level_3_f_53_metrics_per_cr.dat' using ($1*100):($6*100) with linespoints ls 30 title 'FlatL3',\

## G-mean macro ##

set xrange [FL3GMA_min_x:FL3GMA_max_x]

if (FL3GMA_max_y*100+10 > 100){
   set yrange [FL3GMA_min_y*100-1:100]
}
else{
   set yrange [FL3GMA_min_y*100-1:FL3GMA_max_y*100+10]
}
set ylabel 'g-mean macro [%]';

set output 'image/plot_per_cr/GMeanMaFlowPerCrLv3.eps';
plot 'data/metrics_per_cr/flat_flow_level_3_f_53_metrics_per_cr.dat' using ($1*100):($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 30 notitle,\
   'data/metrics_per_cr/flat_flow_level_3_f_53_metrics_per_cr.dat' using ($1*100):($8*100) with linespoints ls 30 title 'FlatL3',\