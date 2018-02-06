f_var = zeros(1, 784);
a = ones(10000, 1);

f_mean = mean(X);

for i = 1: 10000
    f_var = f_var + (X(i, :) - f_mean).^2 / 10000;
end

[~, f] = sort(f_var, 'descend');
f = f(1, 1:200);
f = sort(f);

XX = (X(:, f(1)) - f_mean(f(1)) * a) / sqrt(f_var(f(1)));
for i = 2:200
    XX = [XX, (X(:, f(i)) - f_mean(f(i)) * a) / sqrt(f_var(f(i)))];
end
