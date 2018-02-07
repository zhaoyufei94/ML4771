train_size = [6000, 6500, 7000, 7500, 8000, 8500, 9000];
acc_mle = zeros(1, 7);
acc_knn = zeros(1, 7);

for i = 1:7
    acc_mle(i) = mle(train_size(i), 1, XX, Y);
    acc_knn(i) = knn(train_size(i), 1, 4, X, Y);
end

figure;
plot(train_size, [acc_mle; acc_knn]);
xlabel("train size");
ylabel("accuracy");
legend("MLE", "kNN");

acc_mle = zeros(1, 6);
acc_knn = zeros(1, 6);
start_point = [1, 201, 401, 601, 801, 1001];
for i = 1:6
    acc_mle(i) = mle(9000, start_point(i), XX, Y);
    acc_knn(i) = knn(9000, start_point(i), 4, X, Y);
end

figure;
plot(start_point, [acc_mle; acc_knn]);
xlabel("start point");
ylabel("accuracy");
legend("MLE", "kNN");