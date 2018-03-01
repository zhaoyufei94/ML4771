import marshal
import operator

from os.path import dirname, abspath

d = dirname(abspath(__file__))
d = d.replace('\\', '/')

with open (d+"/unigram", "rb") as f:
	classifier = marshal.load(f)

sorted_classifier = sorted(classifier.items(), key=operator.itemgetter(1))
print(sorted_classifier[0:10])
print(sorted_classifier[-10:])