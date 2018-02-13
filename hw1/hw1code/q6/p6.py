# please use python 2.7 to run this program

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

def load_data():
    dictmat = scipy.io.loadmat("./hw1data.mat")
    images = dictmat['X']
    labels = dictmat['Y']
    labels = np.reshape(labels,([10000]))


    # In order to speed up the process, I picked most various pixels
    variance = np.var(images, axis=0)
    pick_pixels = np.argsort(variance)[-200::]
    images = np.take(images, pick_pixels, axis=1)


    return images, labels

def train_test_split(images, labels, split_rate):

    #ensure that there is a random split between training and test data
    permutation = np.random.permutation(labels.shape[0])
    images = images[permutation]
    labels = labels[permutation]

    idx_sort = np.argsort(labels)
    sorted_records_array = labels[idx_sort]
    vals, idx_start, count = np.unique(sorted_records_array, return_index=True, return_counts=True)
    index_sets = np.split(idx_sort, idx_start[1:])
    data_size = len(labels)
    one_class_num = int(split_rate*data_size/10)

    training_images = np.empty([0,int(images.shape[1])],int)
    training_labels = np.empty([0],int)

    test_images = np.empty([0,int(images.shape[1])],int)
    test_labels = np.empty([0],int)


    for i in range(10):
        training_images = np.concatenate((training_images,images[index_sets[i][:one_class_num]]),axis=0)
        training_labels=np.concatenate((training_labels,labels[index_sets[i][:one_class_num]]),axis=0)

        test_images = np.concatenate((test_images,images[index_sets[i][one_class_num:]]),axis=0)
        test_labels = np.concatenate((test_labels,labels[index_sets[i][one_class_num:]]),axis=0)
    training_labels=training_labels[:,np.newaxis]
    test_labels=test_labels[:,np.newaxis]

    return training_images,training_labels,test_images,test_labels

# Calculate gini score
def gini_index(dataset, labels):
    data_size = float(sum([len(group) for group in dataset]))
    gini = 0.0
    for group in dataset:
        group_size = float(len(group))
        # avoid divide by zero
        if group_size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for label in labels:
            p = [row[-1] for row in group].count(label)/group_size
            score = score + p * p
        # weight the group score by its relative size
        gini = gini+(1.0-score)*(group_size/data_size)
    return gini

# Split dataset
def split_data(index, threshold, dataset):
    left, right = [], []
    for data in dataset:
        if data[index] < threshold:
            left.append(data)
        else:
            right.append(data)
    return left, right

def get_split(dataset):
    # get all the distinct labels
    labels = list(set(data[-1] for data in dataset))

    best_index, best_threshold, best_score, best_groups = 1000, 1000, 1000, 1

    # try all kind of split to find the best feature
    for index in range(len(dataset[0])-1):
        for threshold in (0,50,100,150,200,255):
            groups = split_data(index, threshold, dataset)
            gini = gini_index(groups, labels)
            if gini < best_score:
                best_index, best_threshold, best_score, best_groups = index, threshold, gini, groups
    return {'index':best_index, 'threshold':best_threshold, 'groups':best_groups}

# Create a leaf node
def to_leaf(group):
    classes = [data[-1] for data in group]
    return max(set(classes), key=classes.count)

# Create child node or make leaf
def split(node, max_depth, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a leaf
    if not left or not right:
        node['left'] = node['right'] = to_leaf(left + right)
        return
    # check for max_depth
    if depth >= max_depth:
        node['left'], node['right'] = to_leaf(left), to_leaf(right)
        return

    node['left'] = get_split(left)
    split(node['left'], max_depth, depth+1)

    node['right'] = get_split(right)
    split(node['right'], max_depth, depth+1)

# Build a decision tree
def build_tree(train_data, max_depth):
    root = get_split(train_data)
    split(root, max_depth, 1)
    return root

# Make a prediction
def predict(node, row):
    if row[node['index']] < node['threshold']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# CART
def decision_tree(train_data, test_data, max_depth):
    tree = build_tree(train_data, max_depth)
    test_predictions = []
    for row in test_data:
        prediction = predict(tree, row)
#         print 'predicted=' + str(prediction) + ', real=' + str(row[-1])
        test_predictions.append(prediction)
    train_predictions = []
    for row in train_data:
        train_prediction = predict(tree, row)
#         print 'predicted=' + str(prediction) + ', real=' + str(row[-1])
        train_predictions.append(train_prediction)
    return test_predictions, train_predictions

def accuracy_metric(real, predicted):
    correct = 0
    for i in range(len(real)):
        if real[i] == predicted[i]:
            correct += 1
    return correct/float(len(real))*100.0

def eval_tree(train_data, test_data, algorithm, *arg):
    test_predicted,train_predicted = algorithm(train_data, test_data, *arg)
    test_real = [row[-1] for row in test_data]
    test_accuracy = accuracy_metric(test_real, test_predicted)
    train_real = [row[-1] for row in train_data]
    train_accuracy = accuracy_metric(train_real, train_predicted)

    return test_accuracy,train_accuracy

def main(max_depth):
    split_rate=0.7
    X,y=load_data()
    training_images,training_labels,test_images,test_labels=train_test_split(X, y, split_rate)

    training_data=np.concatenate((training_images, training_labels),axis=1)
    test_data=np.concatenate((test_images, test_labels),axis=1)
    training_data=training_data.tolist()
    test_data=test_data.tolist()


    test_accuracy,train_accuracy = eval_tree(training_data, test_data, decision_tree, max_depth)
    return int(test_accuracy),int(train_accuracy)

test_accuracy,train_accuracy = main(10)
print('test_accuracy='+str(test_accuracy)+"% \n" "train_accuracy="+str(train_accuracy)+"%")
# testset_error_rates=[]
# trainset_error_rates=[]
# for K in (range(1,25,1)):
#     test_accuracy,train_accuracy = main(K)
#     testset_error_rate = 100 - test_accuracy
#     trainset_error_rate = 100 - train_accuracy
#     testset_error_rates.append(testset_error_rate)
#     trainset_error_rates.append(trainset_error_rate)

# disc = {"test_error_rates": testset_error_rates, "train_error_rates": trainset_error_rates}
# df = pd.DataFrame(disc, index=[i for i in range(len(testset_error_rates))])

# df.test_error_rates.plot(legend=True, style='r-')
# df.train_error_rates.plot(style='b--', legend=True)

# plt.xlabel('K (Depth of the decision tree)')
# plt.xticks(np.arange(1, 24, 1))
# plt.ylabel('Error Rate(%)')
# figure = plt.gcf()
# plt.title('Decision Tree Classifier(Training set ratio 70%)')

# figure.savefig('p7_70%.png')
