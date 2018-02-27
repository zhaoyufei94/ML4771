XX = zeros(2000, 1);
YY = zeros(2000, 1);

[~, b] = sort(X);

for i = 1: 2000
    XX(i) = X(b(i));
    YY(i) = Y(b(i));
end