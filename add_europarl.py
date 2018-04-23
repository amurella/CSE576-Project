import string
import re

#load doc into memory
def load_doc(filename):
        # open the file as read only
        file = open(filename, mode='rt', encoding='utf-8')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text

#split a loaded document into sentences
def to_lines(doc):
	lines = doc.strip().split('\n')
	return lines

#combine sentences from two loaded documents into pairs
def combine_into_pairs(doc1, doc2):
	#doc1 and doc2 must have an equal number of lines
	lines1 = to_lines(doc1)
	lines2 = to_lines(doc2)

	f = open('dataset/europarl_spa.txt','w')

	for index in range(len(lines1)):
		f.write(lines1[index])
		f.write('\t')
		f.write(lines2[index])
		f.write('\n')

	f.close()


eng_doc = load_doc('dataset/es-en/europarl-v7.es-en.en')
spa_doc = load_doc('dataset/es-en/europarl-v7.es-en.es')

combine_into_pairs(eng_doc, spa_doc)

