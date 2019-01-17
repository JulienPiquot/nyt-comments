reset
#set autoscale
#set grid xtics lt 0 lw 1 lc rgb "#bbbbb"
#set grid ytics lt 0 lw 1 lc rgb "#bbbbb"

set title "distribution des recommandation"
set xlabel "nombre de recommandations (log)"
set ylabel "effectif"

set datafile separator ' '
#set boxwidth 0.9 relative
#set style fill solid 1.0
plot "logrecommandation_hist.txt" with boxes