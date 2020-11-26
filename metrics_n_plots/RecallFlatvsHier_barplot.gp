reset;

# set key right maxrows 4;
# set key samplen 4 spacing 6 width 10
set key font ", 40";
set key bottom left box opaque
set mytics 10
set grid ytics;
set border 15 linewidth 2

set terminal eps enhanced color size 10in,6in linewidth 1 dl 0.89;\
set size 1.2, 0.97

set xtics scale 0 1 nomirror font ", 35" offset .75, -1;
set ytics font ", 35" offset 0, 0;
set xlabel font ", 40" offset 0, -2;
set ylabel font ", 40" offset -3, 0;

# set macros

#set style data linespoints
set style data histogram
set style histogram errorbars gap 2 lw 2
set style fill solid 1.0 border -1
# set boxwidth 0.9
set xtic format ""
set bars 1
set grid ytics

set bmargin 6
set rmargin 32
set lmargin 13
set tmargin 2

#Scala rosso
set style line 30 lc rgb '#cd5855' lt 1 lw 2 pt 7 pi 1 ps 1

#Scala verde
set style line 38 lc rgb '#a6c761' lt 1 lw 2 pt 3 pi 1 ps 1

#Scala blu
set style line 34 lc rgb '#255187' lt 1 lw 2 pt 13 pi 1 ps 1

#Scala arancione
set style line 40 lc rgb '#cd8b55' lt 1 lw 2 pt 5 pi 1 ps 1

system "mkdir image"
system "mkdir image/histogram"

# Accuracy
# Classifiers
set yrange [40:100]

set output 'image/histogram/FlatVsHierarchicalRecall.eps';
set ylabel  'racall [%]';
plot 'data_wrf/barplot.dat' using ($2*100):(($2-$3*3)*100):(($2+$3*3)*100):xtic(1) ls 30 title 'Flat L3',\
     '' using ($5*100):(($5-$6*3)*100):(($5+$6*3)*100) ls 38 title 'Hierarchical',\

# set output 'image/histogram/FlatVsHierarchicalEarlyFMeasure.eps';
# set ylabel  'f-measure [%]';
# plot 'data_wrf/metrics/join_fp/optimal_early_bar.dat' using ($3*100):(($3-$7*3)*100):(($3+$7*3)*100):xtic(1) ls 40 title 'Flat L1',\
#      '' using ($15*100):(($15-$19*3)*100):(($15+$19*3)*100) ls 34 title 'Flat L2',\
#      '' using ($27*100):(($27-$31*3)*100):(($27+$31*3)*100) ls 30 title 'Flat L3',\
#      '' using ($39*100):(($39-$43*3)*100):(($39+$43*3)*100) ls 38 title 'Hierarchical',\

# set output 'image/histogram/FlatVsHierarchicalEarlyGmean.eps';
# set ylabel  'g-mean [%]';
# plot 'data_wrf/metrics/join_fp/optimal_early_bar.dat' using ($5*100):(($5-$9*3)*100):(($5+$9*3)*100):xtic(1) ls 40 title 'Flat L1',\
#      '' using ($17*100):(($17-$21*3)*100):(($17+$21*3)*100) ls 34 title 'Flat L2',\
#      '' using ($29*100):(($29-$33*3)*100):(($29+$33*3)*100) ls 30 title 'Flat L3',\
#      '' using ($41*100):(($41-$45*3)*100):(($41+$45*3)*100) ls 38 title 'Hierarchical',\
