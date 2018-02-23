import time
import sys
import random
import math
#import marshal
from ipywidgets import IntProgress
from IPython.display import display
from os.path import dirname, abspath
d = dirname(abspath(__file__))
d = d.replace('\\','/')

def get_idf_para(lines):
    n = len(lines)
    word_count_in_docu = {}
    for line in lines:
        line = line[0:-1].split(",")
        text = line[1].split(" ")
        flag = {}
        for word in text:
            flag[word] = 0
        for word in text:
            if word_count_in_docu.get(word)==None:
                word_count_in_docu[word] = 1
                flag[word] = 1
            elif flag[word] != 1:
                word_count_in_docu[word] += 1
                flag[word] = 1
    return n, word_count_in_docu
		
def get_label_feature(line,n,word_count_in_docu):
    line = line[0:-1].split(",")
    if "1" == line[0]:
        label = 1
    else:
        label = -1
    text = line[1].split(" ")
    tf = {}
    feature = {"BIAS": 1}
    for word in text:
        if None == feature.get(word):
            tf[word] = 1
        else:
            tf[word] += 1
    
    for key in tf:
        feature[key] = tf[key]*math.log10(n/word_count_in_docu[key])
    
    return label, feature

def get_value(classifier, feature):
    value = 0
    for word in feature:
        if None == classifier.get(word):
            continue
        else:
            value += classifier[word] * feature[word]
    return value


def perceptron_pass1(lines, classifier,n,word_count_in_docu):
    p = IntProgress(max = len(lines))
    display(p)
    random.shuffle(lines)
    for line in lines:
        label, feature = get_label_feature(line,n,word_count_in_docu)
        value = get_value(classifier, feature)
        if label * value > 0:
            pass
        else:
            for word in feature:
                if None == classifier.get(word):
                    classifier[word] = label * feature[word]
                else:
                    classifier[word] += label * feature[word]
        p.value += 1
        p.description = "{}%".format(round(100*(p.value/p.max), 1))
    return classifier

def perceptron_pass2(lines, classifier,n,word_count_in_docu):
    n = len(lines)
    count = 0
    p = IntProgress(max = n)
    display(p)
    c = classifier
    random.shuffle(lines)
    for line in lines:
        count += 1
        label, feature = get_label_feature(line,n,word_count_in_docu)
        value = get_value(classifier, feature)
        if label * value > 0:
            pass
        else:
            for word in feature:
                if None == classifier.get(word):
                    classifier[word] = label * feature[word]
                    c[word] = label * feature[word] * (n - count) / n
                else:
                    classifier[word] += label * feature[word]
                    c[word] += label * feature[word] * (n - count) / n
        p.value = count
        p.description = "{}%".format(round(100*p.value/p.max, 2))
    p.description = "Done"
    return c

def test(lines, classifier,n,word_count_in_docu):
    count = 0
    correct = 0
    p = IntProgress(max = len(lines))
    display(p)
    for line in lines:
        count += 1
        label, feature = get_label_feature(line,n,word_count_in_docu)
        value = get_value(classifier, feature)
        if label * value >= 0:
            correct += 1
        p.value = count
    return correct / count

start_time = time.time()
train_path = d+"/reviews_tr.csv"
test_path = d+"/reviews_te.csv"

file = open(train_path, "r")
_ = file.readline()

classifier = {}
lines = file.readlines()

print("pass1")
n, word_count_in_docu = get_idf_para(lines)
classifier = perceptron_pass1(lines, classifier,n,word_count_in_docu)

print("pass2")
classifier = perceptron_pass2(lines, classifier,n,word_count_in_docu)

file.close()

file = open(test_path, "r")
_ = file.readline()
lines = file.readlines()
n, word_count_in_docu = get_idf_para(lines)

print("test")
accuracy = test(lines, classifier,n,word_count_in_docu)
file.close()
print(accuracy)
# your code
elapsed_time = time.time() - start_time
print(elapsed_time)
"""
open ("./unigram", "wb") as f:
    f.write(marshal.dump(classifier))
f.close()
"""