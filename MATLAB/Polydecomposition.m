% ������ű���
syms x;

% ������ֽ�Ķ���ʽ
f = 2*x^5 + 3*x^4 + 4*x^3 + 5*x^2 + 6*x + 7;

% ���������ʽ
g = x^3 + 1;

% ��ʼ������б�
quotients = {};
remainders = {};
currentPoly = f;

% �����ʽ�Ĵ���
getDegree = @(poly) feval(symengine, 'degree', poly, x);

% շת�����
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

% ��������ʽ
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

% ��ʾ���
disp('����ʽΪ: ');
disp(polynomialStr);
