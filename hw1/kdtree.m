classdef kdtree
    properties
        index;
        value;
        ltree;
        rtree;
        lcell;
        rcell;
    end
    methods
        function kd = kdtree()
            kd.index = 0;
            kd.value = 0;
            kd.ltree = NaN;
            kd.rtree = NaN;
            kd.lcell = NaN;
            kd.rcell = NaN;
        end
        function print(self)
            disp(self.ltree);
        end
        
        function self = build(self, X, k)
            ssize = size(X);
            f_mean = mean(X);
            f_var = zeros(1, size(2));
            for i = 1:size(1)
                f_var = f_var + (X(i, :) - f_mean).^2 / size(1);
            end
            [~, f] = sort(f_var, 'descend');
            self.index = f(1);
            [v, r] = sort(X(:, f(1)));
            self.value = v(round(size(1) / 2));
            if ssize(1) < 2 * k
            end
        end
    end
end