%acc_L1 = zeros(1, 7);
acc_LInf = zeros(1, 7);
k = [1, 2, 3, 4, 5, 6, 7];

for i = 1: 7
    %acc_L1(i) = knn(9000, 1, k(i), X, Y);
    acc_LInf(i) = knn(9000, 1, k(i), X, Y);
end