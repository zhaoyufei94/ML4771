The p6.py script in this folder is the code part of problem 6.

This python code is based on python 2.7 and need the library of numpy, marplot.py， pandas and spicy. The training procedure make take much time(about 3 minutes). After running the whole procedure, the program will return training and test error percentage when training set ratio is 70% and the maximum depth is 10.

Here is the esign procedure:
    1.Process the data:
	(a) Load the data from the .mat file. In order to the speed up the data, this program only picks the pixels with the biggest variance;
	(b) Split the data. First needs to shuffle the data. Then, this program use the idea of stratified split to make sure that the ratio of each classes in training set is the same as the ratio in the test set.
    2. Write algorithm to evaluate the uncertainty. This program choose to use Gini Index as measurement. This measurement of uncertainty is used to evaluate which kind of splits is better and find the best threshold. 
    3. In the tree algorithm, the tree first split the whole training set and do it recursively until there is only one data or the depth exceed the maximum depth K.
    4. According to the decision tree built by the training data, do predict according to the data and get the predict result.
    5. In the end, compute the error rate of train and test set and plot the line.