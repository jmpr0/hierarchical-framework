reset;

set key at 3, 109 samplen 2 spacing 0.95 font ", 34" maxrows 2;
set mytics 5
set grid ytics;
set border 15 linewidth 2

set terminal postscript eps enhanced color linewidth 1;\
#set size 1.2, 0.97
set size 1.2, 1

set xtics scale 0 1 nomirror font ", 34" offset 2, -0.5;
set ytics font ", 34" add (" " 105, " " 110);
set xlabel font ", 38" offset 0, -1.5;
set ylabel font ", 38" offset -3.5, -3.5;

set style data histogram
set style histogram errorbars gap 2 lw 2.5
#set style histogram cluster
set style fill pattern border -1
set boxwidth 0.9

set bmargin 5
set rmargin 5
set lmargin 13
#set tmargin 5

#Scala rosso
set style line 30 lc rgb '#CC0000' lt 1 lw 2.5 pt 7 pi 1 ps 1

#Scala verde
set style line 40 lc rgb '#008000' lt 1 lw 2.5 pt 3 pi 1 ps 1

#Scala blu
set style line 50 lc rgb '#255187' lt 1 lw 2.5 pt 13 pi 1 ps 1

#Scala arancione
set style line 60 lc rgb '#FF8100' lt 1 lw 2.5 pt 5 pi 1 ps 1

system "mkdir image_new"
system "mkdir image_new/histogram"

set yrange [65:110]

set xlabel 'Classification Level';

# # Accuracy
# set ylabel 'Accuracy [%]';
# set output 'image_new/histogram/FlatVsHierarchicalFlowAccuracy.eps';
# plot 'data_new/optimal_flow_bar.dat' using ($2*100):(($2-$6*3)*100):(($2+$6*3)*100):xtic(1) ls 30 fs pattern 6 title 'Flat L1',\
#      '' using ($12*100):(($12-$16*3)*100):(($12+$16*3)*100) ls 40 fs pattern 7 title 'Flat L2',\
#      '' using ($22*100):(($22-$26*3)*100):(($22+$26*3)*100) ls 50 fs pattern 2 title 'Flat L3',\
#      '' using ($32*100):(($32-$36*3)*100):(($32+$36*3)*100) ls 60 fs pattern 3 title 'H.',\

# # F-measure
# set output 'image_new/histogram/FlatVsHierarchicalFlowFMeasure.eps';
# set ylabel 'F-measure [%]';
# plot 'data_new/optimal_flow_bar.dat' using ($3*100):(($3-$7*3)*100):(($3+$7*3)*100):xtic(1) ls 30 fs pattern 6 title 'Flat L1',\
#      '' using ($13*100):(($13-$17*3)*100):(($13+$17*3)*100) ls 40 fs pattern 7 title 'Flat L2',\
#      '' using ($23*100):(($23-$27*3)*100):(($23+$27*3)*100) ls 50 fs pattern 2 title 'Flat L3',\
#      '' using ($33*100):(($33-$37*3)*100):(($33+$37*3)*100) ls 60 fs pattern 3 title 'H.',\

# # G-mean
# set output 'image_new/histogram/FlatVsHierarchicalFlowGmean.eps';
# set ylabel 'G-mean [%]';
# plot 'data_new/optimal_flow_bar.dat' using ($5*100):(($5-$9*3)*100):(($5+$9*3)*100):xtic(1) ls 30 fs pattern 6 title 'Flat L1',\
#      '' using ($15*100):(($15-$19*3)*100):(($15+$19*3)*100) ls 40 fs pattern 7 title 'Flat L2',\
#      '' using ($25*100):(($25-$29*3)*100):(($25+$29*3)*100) ls 50 fs pattern 2 title 'Flat L3',\
#      '' using ($35*100):(($35-$39*3)*100):(($35+$39*3)*100) ls 60 fs pattern 3 title 'H.',\

# Accuracy
set ylabel 'Accuracy [%]';
set output 'image_new/histogram/FlatVsHierarchicalFlowAccuracy.eps';
plot 'data_new/optimal_flow_bar.dat' using ($2*100):(($2-$6*3)*100):(($2+$6*3)*100):xtic(1) ls 30 fs pattern 6 title 'Flat',\
     '' using ($12*100):(($12-$16*3)*100):(($12+$16*3)*100) ls 40 fs pattern 7 title 'H. Naive',\
     '' using ($22*100):(($22-$26*3)*100):(($22+$26*3)*100) ls 50 fs pattern 2 title 'H. Local',\
     '' using ($32*100):(($32-$36*3)*100):(($32+$36*3)*100) ls 60 fs pattern 3 title 'H. Parent-Dependent',\

# F-measure
set output 'image_new/histogram/FlatVsHierarchicalFlowFMeasure.eps';
set ylabel 'F-measure [%]';
plot 'data_new/optimal_flow_bar.dat' using ($3*100):(($3-$7*3)*100):(($3+$7*3)*100):xtic(1) ls 30 fs pattern 6 title 'Flat',\
     '' using ($13*100):(($13-$17*3)*100):(($13+$17*3)*100) ls 40 fs pattern 7 title 'H. Naive',\
     '' using ($23*100):(($23-$27*3)*100):(($23+$27*3)*100) ls 50 fs pattern 2 title 'H. Local',\
     '' using ($33*100):(($33-$37*3)*100):(($33+$37*3)*100) ls 60 fs pattern 3 title 'H. Parent-Dependent',\

# G-mean
set output 'image_new/histogram/FlatVsHierarchicalFlowGmean.eps';
set ylabel 'G-mean [%]';
plot 'data_new/optimal_flow_bar.dat' using ($5*100):(($5-$9*3)*100):(($5+$9*3)*100):xtic(1) ls 30 fs pattern 6 title 'Flat',\
     '' using ($15*100):(($15-$19*3)*100):(($15+$19*3)*100) ls 40 fs pattern 7 title 'H. Naive',\
     '' using ($25*100):(($25-$29*3)*100):(($25+$29*3)*100) ls 50 fs pattern 2 title 'H. Local',\
     '' using ($35*100):(($35-$39*3)*100):(($35+$39*3)*100) ls 60 fs pattern 3 title 'H. Parent-Dependent',\
