reset
#set autoscale
#set grid xtics lt 0 lw 1 lc rgb "#bbbbb"
#set grid ytics lt 0 lw 1 lc rgb "#bbbbb"

set title "distribution de la taille des articles"
set xlabel "nombre de caractères"
set ylabel "effectif"

set datafile separator ' '
#set boxwidth 0.9 relative
#set style fill solid 1.0
plot "count_word_hist.txt" with boxes