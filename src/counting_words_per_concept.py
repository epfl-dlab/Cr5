import gzip
import pickle
import sys
import argparse
from utils import *

parser = argparse.ArgumentParser(description='Words counter')
parser.add_argument("--voc_size", type=int, default=65000, help="Vocabulary size")
parser.add_argument("--lang_code", type=str, default=None, help="Language code")
params = parser.parse_args()

assert(params.lang_code is not None)

lang_code = params.lang_code
words_considered = params.voc_size


lang_id = '%s_%d' % (lang_code, words_considered)
dump_file_name = "unique_words_per_concept_%s.p" % lang_id
dump_file_name = get_word_counts_file_path(dump_file_name)

if os.path.isfile(dump_file_name):
	print("Dump file already exists.")
	sys.exit()

concepts = {} # concepts that have at least one word with ID less than or equal to words_considered
all_concepts = set() # the previous plus the concepts that don't have any words complying to the criterion 


print("Starting %s" % lang_id)


with gzip.open(get_sparse_matrix_file_path(lang_code), 'r') as f:
	c = 0

	for line in codecs.getreader("utf-8")(f):
		c += 1
		if c % 1000000 == 0:
			print("%d rows processed" % c)
			sys.stdout.flush()
		    
		parts = line.split()
		all_concepts.add(parts[0])
		if int(parts[1]) > words_considered:
			continue
		if parts[0] in concepts:
			concepts[parts[0]] += 1
		else:
			concepts[parts[0]] = 1
    
print("Finished with %s" % lang_code)

result = {'all_concepts' : all_concepts, 'concepts' : concepts}

print('Number of concepts in range: ', len(concepts))

pickle.dump(result, open(dump_file_name, 'wb'))