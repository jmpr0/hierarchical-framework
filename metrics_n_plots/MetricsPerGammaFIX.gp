reset;

set key at .45, 42 maxcols 1;
set key samplen 4 spacing 0.95 width -4.80
set mxtics 5
set mytics 5
set grid ytics;
set border 15 linewidth 2

set terminal eps enhanced color size 10in,6in linewidth 1 dl 1;\
set size 1.2, 0.97

set xtics nomirror font ", 38" offset -0.85, -0.5;
set ytics font ", 38" offset -0.5, 0;
set xlabel font ", 45" offset 0, -1.5;
set ylabel font ", 45" offset -6, 0;
set key font ", 32";

# set macros
# dummy="NaN title ' ' lt -3"

#set style data_wrf linespoints
set style fill transparent solid 0.15 noborder
set bmargin 6
set rmargin 35
set lmargin 15

#Scala rosso
set style line 30 lc rgb '#CC0000' lt 1 dt 2 lw 5 pt 4 pi 1 ps 1.5
set style line 31 lc rgb '#CC0000' lt 1 dt 1 lw 5 pt 5 pi 1 ps 1.5

#Scala verde
set style line 32 lc rgb '#008000' lt 1 dt 2 lw 5 pt 6 pi 1 ps 1.5
set style line 33 lc rgb '#008000' lt 1 dt 1 lw 5 pt 7 pi 1 ps 1.5

#Scala blu
set style line 34 lc rgb '#255187' lt 1 dt 2 lw 5 pt 6 pi 1 ps 1.5
set style line 35 lc rgb '#255187' lt 1 dt 1 lw 5 pt 7 pi 1 ps 1.5

#Scala arancione
set style line 36 lc rgb '#FF8100' lt 1 dt 2 lw 5 pt 8 pi 1 ps 1.5
set style line 37 lc rgb '#FF8100' lt 1 dt 1 lw 5 pt 9 pi 1 ps 1.5

system "mkdir image_FIXLV3FLOW"
system "mkdir image_FIXLV3FLOW/plot_per_gamma"

system "mkdir image_wrf"
system "mkdir image_wrf/plot_per_gamma"

##########################################################################################################
# Alle posizioni pari abbiamo medie, alle dispari le corrispondenti std_dev.                             
#
# 2) accuracy
# 4) f-measure
# 6) gmean multiclass
# 8) gmean macro
##########################################################################################################

print 'data_FIXLV3FLOW/metrics/join_gamma/multi_flow_level_3_c_FIXLV3FLOW_metrics.dat'
stats 'data_FIXLV3FLOW/metrics/join_gamma/multi_flow_level_3_c_FIXLV3FLOW_metrics.dat' using 1:4 name "FIXL3F"

print 'data_OPTALLEARLY/metrics/join_gamma/multi_flow_level_2_c_OPTALLEARLY_metrics.dat'
stats 'data_OPTALLEARLY/metrics/join_gamma/multi_flow_level_2_c_OPTALLEARLY_metrics.dat' using 1:4 name "OPTL3F"

### LEVEL 3 ###

set xlabel '{/Symbol g}'
set ylabel 'Percentage'

set xrange [FIXL3F_min_x:FIXL3F_max_x]
set yrange [40:100]

set output 'image_FIXLV3FLOW/plot_per_gamma/MetricsPerGammaFlowLv2.eps';
plot 'data_FIXLV3FLOW/metrics/join_gamma/multi_flow_level_2_c_FIXLV3FLOW_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
     'data_FIXLV3FLOW/metrics/join_gamma/multi_flow_level_2_c_FIXLV3FLOW_metrics.dat' using 1:($4*100) with linespoints ls 30 title 'F-measure (HC_{all})',\
     'data_FIXLV3FLOW/metrics/join_gamma/multi_flow_level_2_c_FIXLV3FLOW_metrics.dat' using 1:($10+(3*$11))*100:($10-(3*$11))*100 with filledcurves ls 31 notitle,\
     'data_FIXLV3FLOW/metrics/join_gamma/multi_flow_level_2_c_FIXLV3FLOW_metrics.dat' using 1:($10*100) with linespoints ls 31 title 'CR (HC_{all})',\
     'data_FIXLV3FLOW/metrics/join_gamma/multi_flow_level_3_c_FIXLV3FLOW_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 32 notitle,\
     'data_FIXLV3FLOW/metrics/join_gamma/multi_flow_level_3_c_FIXLV3FLOW_metrics.dat' using 1:($4*100) with linespoints ls 32 title 'F-measure (HC_3)',\
     'data_FIXLV3FLOW/metrics/join_gamma/multi_flow_level_3_c_FIXLV3FLOW_metrics.dat' using 1:($10+(3*$11))*100:($10-(3*$11))*100 with filledcurves ls 33 notitle,\
     'data_FIXLV3FLOW/metrics/join_gamma/multi_flow_level_3_c_FIXLV3FLOW_metrics.dat' using 1:($10*100) with linespoints ls 33 title 'CR (HC_3)',\
     'data_wrf/metrics/join_gamma/flat_flow_level_3_f_65_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 34 notitle,\
     'data_wrf/metrics/join_gamma/flat_flow_level_3_f_65_metrics.dat' using 1:($4*100) with linespoints ls 34 title 'F-measure (FC_3)',\
     'data_wrf/metrics/join_gamma/flat_flow_level_3_f_65_metrics.dat' using 1:($10+(3*$11))*100:($10-(3*$11))*100 with filledcurves ls 35 notitle,\
     'data_wrf/metrics/join_gamma/flat_flow_level_3_f_65_metrics.dat' using 1:($10*100) with linespoints ls 35 title 'CR (FC_3)',\

set xrange [OPTL3F_min_x:OPTL3F_max_x]
set yrange [0:100]

set output 'image_OPTALLEARLY/plot_per_gamma/MetricsPerGammaFlowLv2.eps';
plot 'data_OPTALLEARLY/metrics/join_gamma/multi_flow_level_2_c_OPTALLEARLY_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 30 notitle,\
     'data_OPTALLEARLY/metrics/join_gamma/multi_flow_level_2_c_OPTALLEARLY_metrics.dat' using 1:($4*100) with linespoints ls 30 title 'F-measure (HC_{all})',\
     'data_OPTALLEARLY/metrics/join_gamma/multi_flow_level_2_c_OPTALLEARLY_metrics.dat' using 1:($10+(3*$11))*100:($10-(3*$11))*100 with filledcurves ls 31 notitle,\
     'data_OPTALLEARLY/metrics/join_gamma/multi_flow_level_2_c_OPTALLEARLY_metrics.dat' using 1:($10*100) with linespoints ls 31 title 'CR (HC_{all})',\
     'data_OPTALLEARLY/metrics/join_gamma/multi_flow_level_3_c_OPTALLEARLY_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 36 notitle,\
     'data_OPTALLEARLY/metrics/join_gamma/multi_flow_level_3_c_OPTALLEARLY_metrics.dat' using 1:($4*100) with linespoints ls 36 title 'F-measure (HC_3)',\
     'data_OPTALLEARLY/metrics/join_gamma/multi_flow_level_3_c_OPTALLEARLY_metrics.dat' using 1:($10+(3*$11))*100:($10-(3*$11))*100 with filledcurves ls 37 notitle,\
     'data_OPTALLEARLY/metrics/join_gamma/multi_flow_level_3_c_OPTALLEARLY_metrics.dat' using 1:($10*100) with linespoints ls 37 title 'CR (HC_3)',\
     'data_wbn/metrics/join_gamma/flat_early_level_3_p_11_metrics.dat' using 1:($4+(3*$5))*100:($4-(3*$5))*100 with filledcurves ls 34 notitle,\
     'data_wbn/metrics/join_gamma/flat_early_level_3_p_11_metrics.dat' using 1:($4*100) with linespoints ls 34 title 'F-measure (FC_3)',\
     'data_wbn/metrics/join_gamma/flat_early_level_3_p_11_metrics.dat' using 1:($10+(3*$11))*100:($10-(3*$11))*100 with filledcurves ls 35 notitle,\
     'data_wbn/metrics/join_gamma/flat_early_level_3_p_11_metrics.dat' using 1:($10*100) with linespoints ls 35 title 'CR (FC_3)',\
