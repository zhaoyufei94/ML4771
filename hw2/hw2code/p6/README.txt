For problem 6, we have 3 Matlab scripts.

1. nn.m: 
main script for build and train neural network.
call it by [w1, w2, b1, b2] = nn(k, r, X, Y);
where k is the number of neurons in hidden layer and r is learning rate,
X and Y should be sorted data points(see below for more details).

2. sigmoid.m:
express sigmoid function here for better layout in nn.m

3. sort.m:
the script to re-arrange given X, Y data points in an incremental order of X.
So that when ploting the data points, it will result in a curve instead of a block.


When running the whole process(with X and Y already loaded in workspace):
> sort;
> [w1, w2, b1, b2] = nn(12, 0.8, XX, YY);
