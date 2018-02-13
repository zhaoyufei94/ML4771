Before running these codes, you should have loaded hw1data.mat in MATLAB. So there should be X for images and Y for corresponding labels in workspace.

p5_1.m and p5_2.m is main script for the first two subproblems of problem 5, respectively.

p5_1.m first runs preprocessing.m to prepare the trainable data as XX in workspace. Then, it calls function mle which XX, Y and number of samples used to train as parameters. And train the Multiple Gaussian model with the first train_size samples of XX associated with Y, test with rest of them and returns the test accuracy.

p5_2.m simply calls knn function to run k-NN algorithm on X and Y, with a split training/test of train_size/10000-train_size