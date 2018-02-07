function accuracy = knn(train_size, start_point, k, X, Y)
% Compute accuracy of k-NN on given data
% train_size: number of data(out of 10000) used to train k-NN
% k: parameter in k-NN
% X, Y: data (10000*784) and label (10000*1)
tic;
correct = 0;

if train_size + start_point < 10000
for i = train_size + start_point: 10000
    d = zeros(1, train_size);
    count = zeros(1, 10);
    
    for j = start_point: train_size + start_point - 1
        %dd = (X(i, :) - X(j, :)) * (X(i, :) - X(j, :))';
        %d(j - start_point + 1) = sqrt(dd);
        d(j - start_point + 1) = norm((X(i, :) - X(j, :)), Inf);
    end

    [~, n] = sort(d);
    for j = 1:k
        count(Y(n(j) + start_point - 1) + 1) = count(Y(n(j) + start_point - 1) + 1) + 1;
    end
    [~, n] = sort(count, 'descend');
    if n(1) == Y(i) + 1
        correct = correct + 1;
    end

end
end

if start_point > 1
for i = 1: start_point - 1
    d = zeros(1, train_size);
    count = zeros(1, 10);
    
    for j = start_point: train_size + start_point - 1
        %dd = (X(i, :) - X(j, :)) * (X(i, :) - X(j, :))';
        %d(j - start_point + 1) = sqrt(dd);
        d(j - start_point + 1) = norm((X(i, :) - X(j, :)), Inf);
    end

    [~, n] = sort(d);
    for j = 1:k
        count(Y(n(j) + start_point - 1) + 1) = count(Y(n(j) + start_point - 1) + 1) + 1;
    end
    [~, n] = sort(count, 'descend');
    if n(1) == Y(i) + 1
        correct = correct + 1;
    end

end
end
t = toc;
accuracy = correct / (10000 - train_size);
disp(accuracy);
disp(t / 60);