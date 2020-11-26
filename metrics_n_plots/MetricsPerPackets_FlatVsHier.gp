reset;

set key bottom right maxrows 2;
set key samplen 4 spacing 0.95
set mxtics 5
set mytics 5
set grid ytics;
set border 15 linewidth 2

set terminal eps enhanced color size 10in,6in linewidth 1 dl 0.89;\
set size 1.2, 0.97

set xtics 1,1,16 nomirror font ", 38" offset -0.5, -0.5;
set ytics font ", 38";
set xlabel font ", 45" offset 0, -2;
set ylabel font ", 45" offset -5, 0;
set key font ", 43";

set macros
dummy="NaN title ' ' lt -3"

#set style data_wrf linespoints
set style fill transparent solid 0.15 noborder
set bmargin 6
set rmargin 32
set lmargin 15

#Scala rosso
set style line 30 lc rgb '#CC0000' lt 1        lw 5 pt 4 pi 1 ps 1.5

#Scala verde
set style line 40 lc rgb '#008000' lt 1        lw 5 pt 8 pi 1 ps 1.5

#Scala blu
set style line 50 lc rgb '#255187' lt 1        lw 5 pt 14 pi 1 ps 1.5

#Scala arancione
set style line 60 lc rgb '#FF8100' lt 1        lw 5 pt 6 pi 1 ps 1.5

system "mkdir image_new"
system "mkdir image_new/plot_per_x"

# Statistics
print 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_1_metrics.dat'
stats 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_1_metrics.dat' using 1:2 name "L1Am"
print 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_1_metrics.dat'
stats 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_1_metrics.dat' using 1:4 name "L1Fm"
stats 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_1_metrics.dat' using 1:8 name "L1Gm"
print 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_2_metrics.dat'
stats 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_2_metrics.dat' using 1:2 name "L2Am"
print 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_2_metrics.dat'
stats 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_2_metrics.dat' using 1:4 name "L2Fm"
stats 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_2_metrics.dat' using 1:8 name "L2Gm"
print 'data_wbn/metrics/join_fp/flat_early_level_3_metrics.dat'
stats 'data_wbn/metrics/join_fp/flat_early_level_3_metrics.dat' using 1:2 name "L3Am"
print 'data_wbn/metrics/join_fp/flat_early_level_3_metrics.dat'
stats 'data_wbn/metrics/join_fp/flat_early_level_3_metrics.dat' using 1:4 name "L3Fm"
stats 'data_wbn/metrics/join_fp/flat_early_level_3_metrics.dat' using 1:8 name "L3Gm"

print 'data_wbn/metrics/join_fp/flat_early_level_2_inferred_1_metrics.dat'
stats 'data_wbn/metrics/join_fp/flat_early_level_2_inferred_1_metrics.dat' using 1:2
print 'data_wbn/metrics/join_fp/flat_early_level_2_inferred_1_metrics.dat'
stats 'data_wbn/metrics/join_fp/flat_early_level_2_inferred_1_metrics.dat' using 1:4
stats 'data_wbn/metrics/join_fp/flat_early_level_2_inferred_1_metrics.dat' using 1:8
print 'data_wbn/metrics/join_fp/flat_early_level_2_metrics.dat'
stats 'data_wbn/metrics/join_fp/flat_early_level_2_metrics.dat' using 1:2
print 'data_wbn/metrics/join_fp/flat_early_level_2_metrics.dat'
stats 'data_wbn/metrics/join_fp/flat_early_level_2_metrics.dat' using 1:4
stats 'data_wbn/metrics/join_fp/flat_early_level_2_metrics.dat' using 1:8

print 'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat'
stats 'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:2
print 'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat'
stats 'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:4
stats 'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:8

print 'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat'
stats 'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:2 name "L1AM"
print 'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat'
stats 'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:4 name "L1FM"
stats 'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:8 name "L1GM"
print 'data_wbn/metrics/join_fp/multi_early_level_2_metrics.dat'
stats 'data_wbn/metrics/join_fp/multi_early_level_2_metrics.dat' using 1:2 name "L2AM"
print 'data_wbn/metrics/join_fp/multi_early_level_2_metrics.dat'
stats 'data_wbn/metrics/join_fp/multi_early_level_2_metrics.dat' using 1:4 name "L2FM"
stats 'data_wbn/metrics/join_fp/multi_early_level_2_metrics.dat' using 1:8 name "L2GM"
print 'data_wbn/metrics/join_fp/multi_early_level_3_metrics.dat'
stats 'data_wbn/metrics/join_fp/multi_early_level_3_metrics.dat' using 1:2 name "L3AM"
print 'data_wbn/metrics/join_fp/multi_early_level_3_metrics.dat'
stats 'data_wbn/metrics/join_fp/multi_early_level_3_metrics.dat' using 1:4 name "L3FM"
stats 'data_wbn/metrics/join_fp/multi_early_level_3_metrics.dat' using 1:8 name "L3GM"


# L1 Classification
set ylabel 'Accuracy [%]';
set xlabel 'Packet Count';

set xrange [L1Am_min_x:L1AM_max_x]
if (L1AM_max_y*100+10 > 100){
     set yrange [L1Am_min_y*100-1:100]
}
else{
     set yrange [L1Am_min_y*100-1:L1AM_max_y*100+10]
}
set yrange [95:100]
set output 'image_new/plot_per_x/FlatVsHierarchicalEarlyL1Accuracy.eps';
plot 'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 60 notitle,\
     'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:($2*100) with linespoints ls 60 title 'Flat L1',\
     'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 40 notitle,\
     'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:($2*100) with linespoints ls 40 title 'Hierarchical',\
     # 'data_wbn/metrics/join_fp/flat_early_level_2_inferred_1_metrics.dat' using 1:($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 40 notitle,\
     # 'data_wbn/metrics/join_fp/flat_early_level_2_inferred_1_metrics.dat' using 1:($2*100) with linespoints ls 40 title 'Flat L2',\
     # 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_1_metrics.dat' using 1:($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 50 notitle,\
     # 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_1_metrics.dat' using 1:($2*100) with linespoints ls 50 title 'Flat L3',\

set xrange [L1Fm_min_x:L1FM_max_x]
if (L1FM_max_y*100+10 > 100){
     set yrange [L1Fm_min_y*100-1:100]
}
else{
     set yrange [L1Fm_min_y*100-1:L1FM_max_y*100+10]
}
set yrange [95:100]
set output 'image_new/plot_per_x/FlatVsHierarchicalEarlyL1FMeasure.eps';
set ylabel  'F-measure [%]';
plot 'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 60 notitle,\
     'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:($4*100) with linespoints ls 60 title 'Flat L1',\
     'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 40 notitle,\
     'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:($4*100) with linespoints ls 40 title 'Hierarchical',\
     #      'data_wbn/metrics/join_fp/flat_early_level_2_inferred_1_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 40 notitle,\
     # 'data_wbn/metrics/join_fp/flat_early_level_2_inferred_1_metrics.dat' using 1:($4*100) with linespoints ls 40 title 'Flat L2',\
     # 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_1_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 50 notitle,\
     # 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_1_metrics.dat' using 1:($4*100) with linespoints ls 50 title 'Flat L3',\

set xrange [L1Gm_min_x:L1GM_max_x]
if (L1GM_max_y*100+10 > 100){
     set yrange [L1Gm_min_y*100-1:100]
}
else{
     set yrange [L1Gm_min_y*100-1:L1GM_max_y*100+10]
}
set yrange [95:100]
set output 'image_new/plot_per_x/FlatVsHierarchicalEarlyL1Gmean.eps';
set ylabel 'G-mean [%]';
plot 'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 60 notitle,\
     'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:($8*100) with linespoints ls 60 title 'Flat L1',\
     'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 40 notitle,\
     'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:($8*100) with linespoints ls 40 title 'Hierarchical',\
     #      'data_wbn/metrics/join_fp/flat_early_level_2_inferred_1_metrics.dat' using 1:($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 40 notitle,\
     # 'data_wbn/metrics/join_fp/flat_early_level_2_inferred_1_metrics.dat' using 1:($8*100) with linespoints ls 40 title 'Flat L2',\
     # 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_1_metrics.dat' using 1:($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 50 notitle,\
     # 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_1_metrics.dat' using 1:($8*100) with linespoints ls 50 title 'Flat L3',\

# set xrange [L1Fm_min_x:L1FM_max_x]
# if (L1FM_max_y*100+10 > 100){
#      set yrange [L1FM_min_y*100-1:100]
# }
# else{
#      set yrange [L1FM_min_y*100-1:L1FM_max_y*100+10]
# }
# set output 'image_new/plot_per_x/FlatVsHierarchicalEarlyL1FMeasureNoL3.eps';
# set ylabel  'F-measure [%]';
# plot 'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
#      'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:($4*100) with linespoints ls 30 title 'Flat L1',\
#      'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 60 notitle,\
#      'data_wbn/metrics/join_fp/multi_early_level_1_metrics.dat' using 1:($4*100) with linespoints ls 60 title 'Hierarchical',\
#      'data_wbn/metrics/join_fp/flat_early_level_2_inferred_1_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 40 notitle,\
#      'data_wbn/metrics/join_fp/flat_early_level_2_inferred_1_metrics.dat' using 1:($4*100) with linespoints ls 40 title 'Flat L2',\


# L2 Classification
set xrange [L2Am_min_x:L2AM_max_x]
if (L2AM_max_y*100+10 > 100){
     set yrange [L2Am_min_y*100-1:100]
}
else{
     set yrange [L2Am_min_y*100-1:L2AM_max_y*100+10]
}
set yrange [70:90]
set output 'image_new/plot_per_x/FlatVsHierarchicalEarlyL2Accuracy.eps';
set ylabel 'Accuracy [%]';
plot 'data_wbn/metrics/join_fp/flat_early_level_2_metrics.dat' using 1:($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 60 notitle,\
     'data_wbn/metrics/join_fp/flat_early_level_2_metrics.dat' using 1:($2*100) with linespoints ls 60 title 'Flat L2',\
     'data_wbn/metrics/join_fp/multi_early_level_2_metrics.dat' using 1:($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 40 notitle,\
     'data_wbn/metrics/join_fp/multi_early_level_2_metrics.dat' using 1:($2*100) with linespoints ls 40 title 'Hierarchical',\
     #      'data_wbn/metrics/join_fp/flat_early_level_3_inferred_2_metrics.dat' using 1:($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 50 notitle,\
     # 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_2_metrics.dat' using 1:($2*100) with linespoints ls 50 title 'Flat L3',\

set xrange [L2Fm_min_x:L2FM_max_x]
if (L2FM_max_y*100+10 > 100){
     set yrange [L2Fm_min_y*100-1:100]
}
else{
     set yrange [L2Fm_min_y*100-1:L2FM_max_y*100+10]
}
set yrange [50:90]
set output 'image_new/plot_per_x/FlatVsHierarchicalEarlyL2FMeasure.eps';
set ylabel 'F-measure [%]';
plot 'data_wbn/metrics/join_fp/flat_early_level_2_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 60 notitle,\
     'data_wbn/metrics/join_fp/flat_early_level_2_metrics.dat' using 1:($4*100) with linespoints ls 60 title 'Flat L2',\
     'data_wbn/metrics/join_fp/multi_early_level_2_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 40 notitle,\
     'data_wbn/metrics/join_fp/multi_early_level_2_metrics.dat' using 1:($4*100) with linespoints ls 40 title 'Hierarchical',\
     #      'data_wbn/metrics/join_fp/flat_early_level_3_inferred_2_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 50 notitle,\
     # 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_2_metrics.dat' using 1:($4*100) with linespoints ls 50 title 'Flat L3',\

set xrange [L2Gm_min_x:L2GM_max_x]
if (L2GM_max_y*100+10 > 100){
     set yrange [L2Gm_min_y*100-1:100]
}
else{
     set yrange [L2Gm_min_y*100-1:L2GM_max_y*100+10]
}
set yrange [70:95]
set output 'image_new/plot_per_x/FlatVsHierarchicalEarlyL2Gmean.eps';
set ylabel 'G-mean [%]';
plot 'data_wbn/metrics/join_fp/flat_early_level_2_metrics.dat' using 1:($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 60 notitle,\
     'data_wbn/metrics/join_fp/flat_early_level_2_metrics.dat' using 1:($8*100) with linespoints ls 60 title 'Flat L2',\
     'data_wbn/metrics/join_fp/multi_early_level_2_metrics.dat' using 1:($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 40 notitle,\
     'data_wbn/metrics/join_fp/multi_early_level_2_metrics.dat' using 1:($8*100) with linespoints ls 40 title 'Hierarchical',\
     #      'data_wbn/metrics/join_fp/flat_early_level_3_inferred_2_metrics.dat' using 1:($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 50 notitle,\
     # 'data_wbn/metrics/join_fp/flat_early_level_3_inferred_2_metrics.dat' using 1:($8*100) with linespoints ls 50 title 'Flat L3',\


# L3 Classification
set xrange [L3Am_min_x:L3AM_max_x]
if (L3AM_max_y*100+10 > 100){
     set yrange [L3Am_min_y*100-1:100]
}
else{
     set yrange [L3Am_min_y*100-1:L3AM_max_y*100+10]
}
set yrange [50:70]
set output 'image_new/plot_per_x/FlatVsHierarchicalEarlyL3Accuracy.eps';
set ylabel  'Accuracy [%]';
plot 'data_wbn/metrics/join_fp/flat_early_level_3_metrics.dat' using 1:($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 60 notitle,\
     'data_wbn/metrics/join_fp/flat_early_level_3_metrics.dat' using 1:($2*100) with linespoints ls 60 title 'Flat L3',\
     'data_wbn/metrics/join_fp/multi_early_level_3_metrics.dat' using 1:($2+(3*$3))*100:($2-(3*$3))*100 with filledcurves ls 40 notitle,\
     'data_wbn/metrics/join_fp/multi_early_level_3_metrics.dat' using 1:($2*100) with linespoints ls 40 title 'Hierarchical',\

set xrange [L3Fm_min_x:L3FM_max_x]
if (L3FM_max_y*100+10 > 100){
     set yrange [L3Fm_min_y*100-1:100]
}
else{
     set yrange [L3Fm_min_y*100-1:L3FM_max_y*100+10]
}
set yrange [20:60]
set output 'image_new/plot_per_x/FlatVsHierarchicalEarlyL3FMeasure.eps';
set ylabel  'F-measure [%]';
plot 'data_wbn/metrics/join_fp/flat_early_level_3_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 60 notitle,\
     'data_wbn/metrics/join_fp/flat_early_level_3_metrics.dat' using 1:($4*100) with linespoints ls 60 title 'Flat L3',\
     'data_wbn/metrics/join_fp/multi_early_level_3_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 40 notitle,\
     'data_wbn/metrics/join_fp/multi_early_level_3_metrics.dat' using 1:($4*100) with linespoints ls 40 title 'Hierarchical',\

set xrange [L3Gm_min_x:L3GM_max_x]
if (L3GM_max_y*100+10 > 100){
     set yrange [L3Gm_min_y*100-1:100]
}
else{
     set yrange [L3Gm_min_y*100-1:L3GM_max_y*100+10]
}
set yrange [50:75]
set output 'image_new/plot_per_x/FlatVsHierarchicalEarlyL3Gmean.eps';
set ylabel 'G-mean [%]';
plot 'data_wbn/metrics/join_fp/flat_early_level_3_metrics.dat' using 1:($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 60 notitle,\
     'data_wbn/metrics/join_fp/flat_early_level_3_metrics.dat' using 1:($8*100) with linespoints ls 60 title 'Flat L3',\
     'data_wbn/metrics/join_fp/multi_early_level_3_metrics.dat' using 1:($8+(3*$9))*100:($8-(3*$9))*100 with filledcurves ls 40 notitle,\
     'data_wbn/metrics/join_fp/multi_early_level_3_metrics.dat' using 1:($8*100) with linespoints ls 40 title 'Hierarchical',\
