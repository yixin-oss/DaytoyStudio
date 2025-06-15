% 定义符号变量
syms x;

% 定义待分解的多项式
f = 2*x^5 + 3*x^4 + 4*x^3 + 5*x^2 + 6*x + 7;

% 定义基多项式
g = x^3 + 1;

% 初始化结果列表
quotients = {};
remainders = {};
currentPoly = f;

% 求多项式的次数
getDegree = @(poly) feval(symengine, 'degree', poly, x);

% 辗转相除法
while true
    [q, r] = quorem(currentPoly, g);
    quotients{end+1} = q;
    remainders{end+1} = r;
    currentPoly = q;
    
    if getDegree(currentPoly) < getDegree(g)
        break;
    end
end

coefficients=[quotients{end}];
for i=length(remainders):-1:1
    temp=remainders{i};
    coefficients=[coefficients,temp];
end

% 构建多项式
polynomialStr = '';
n = length(coefficients);

for i = 1:n
    if coefficients(i) ~= 0
        termStr = sprintf('(%s)*(%s)^%d', char(coefficients(i)), g, n-i);
        
        if ~isempty(polynomialStr)
                termStr = [' + ' termStr];
        end

        polynomialStr = [polynomialStr termStr];
    end
end

% 显示结果
disp('多项式为: ');
disp(polynomialStr);
