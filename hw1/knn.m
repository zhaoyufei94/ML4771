function accuracy = knn(train_size, k, X, Y)
% Compute accuracy of k-NN on given data
% train_size: number of data(out of 10000) used to train k-NN
% k: parameter in k-NN
% X, Y: data (10000*784) and label (10000*1)

correct = 0;

for i = train_size + 1: 10000
    d = zeros(1, train_size);
    count = zeros(1, 10);
    
    for j = 1: train_size
        dd = (X(i, :) - X(j, :)) * (X(i, :) - X(j, :))';
        d(j) = sqrt(dd);
    end

    [~, n] = sort(d);
    for j = 1:k
        count(Y(n(j)) + 1) = count(Y(n(j)) + 1) + 1;
    end
    [~, n] = sort(count, 'descend');
    if n(1) == Y(i) + 1
        correct = correct + 1;
    end

end
accuracy = correct / (10000 - train_size);
