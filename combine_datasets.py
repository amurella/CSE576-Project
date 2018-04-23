
f_combined = open("dataset/combined_spa.txt", "w")
f_tatoeba = open("dataset/spa.txt", "r")
f_europarl = open("dataset/europarl_spa.txt", "r")

f_combined.write(f_tatoeba.read())
f_combined.write(f_europarl.read())

