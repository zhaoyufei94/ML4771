import time
import sys
import random
import marshal
from ipywidgets import IntProgress
from IPython.display import display

def get_label_feature(line):
    line = line[0:-1].split(",")
    if "1" == line[0]:
        label = 1
    else:
        label = -1
    text = line[1].split(" ")
    feature = {"BIAS": 1}
    for i in range(len(text)-1):
        pair = (text[i], text[i+1])
        if None == feature.get(pair):
            feature[pair] = 1
        else:
            feature[pair] += 1
    return label, feature

def get_value(classifier, feature):
    value = 0
    for word in feature:
        if None == classifier.get(word):
            continue
        else:
            value += classifier[word] * feature[word]
    return value


def perceptron_pass1(lines, classifier):
    p = IntProgress(max = len(lines))
    display(p)
    random.shuffle(lines)
    for line in lines:
        label, feature = get_label_feature(line)
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

def perceptron_pass2(lines, classifier):
    n = len(lines)
    count = 0
    p = IntProgress(max = n)
    display(p)
    c = classifier
    random.shuffle(lines)
    for line in lines:
        count += 1
        label, feature = get_label_feature(line)
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

def test(lines, classifier):
    count = 0
    correct = 0
    p = IntProgress(max = len(lines))
    display(p)
    for line in lines:
        count += 1
        label, feature = get_label_feature(line)
        value = get_value(classifier, feature)
        if label * value >= 0:
            correct += 1
        p.value = count
    return correct / count


train_path = "../hw2data_1/reviews_tr.csv"
test_path = "../hw2data_1/reviews_te.csv"

file = open(train_path, "r")
_ = file.readline()

classifier = {}
lines = file.readlines()

print("pass1")
classifier = perceptron_pass1(lines, classifier)

print("pass2")
classifier = perceptron_pass2(lines, classifier)

file.close()

file = open(test_path, "r")
_ = file.readline()
lines = file.readlines()
print("test")
accuracy = test(lines, classifier)
file.close()
print(accuracy)
"""
open ("./bigram", "wb") as f:
    f.write(marshal.dump(classifier))
f.close()
"""
