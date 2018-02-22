import sys
import random

def get_label_feature(line):
    line = line[0:-1].split(",")
    if "1" == line[0]:
        label = 1
    else:
        label = -1
    text = line[1].split(" ")
    feature = {"BIAS": 1}
    for word in text:
        if None == feature.get(word):
            feature[word] = 1
        else:
            feature[word] += 1
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
    return classifier

def perceptron_pass2(lines, classifier):
    c = {}
    random.shuffle(lines)
    n = len(lines)
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
        for word in classifier:
            if None == c.get(word):
                c[word] = classifier[word] / n
            else:
                c[word] += classifier[word] / n
        bar.move()
        bar.log()
    return c

def test(lines, classifier):
    count = 0
    correct = 0
    for line in lines:
        count += 1
        label, feature = get_label_feature(line)
        value = get_value(classifier, feature)
        if label * value > 0:
            correct += 1
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
#print(count)
#print(classifier)
