clear
fileName = 'coefs_db97.txt';
%%
low = sqrt(2) .* [0.602949, 0.266864, -0.078223, -0.016864, 0.026749]';
low = [low(end:-1:2); low];
high = 1/sqrt(2) .* [1.115087, -0.591272, -0.057544, 0.091272]';
high = [high(end:-1:2); high];

low_syn = 1/sqrt(2) .* [1.115087, 0.591272, -0.057544, -0.091272]';
low_syn = [low_syn(end:-1:2); low_syn];
high_syn = sqrt(2) .* [0.602949, -0.266864, -0.078223, 0.016864, 0.026749]';
high_syn = [high_syn(end:-1:2); high_syn];

[m_0, m_1] = getScalingMat(low);
[n_0, n_1] = getWaveletMat(low, high);
%%
fileID = fopen(fileName, 'w');
fprintf(fileID, 'low');
fprintf(fileID, '\t%.6f', low);
fprintf(fileID, '\n');

fprintf(fileID, 'high');
fprintf(fileID, '\t%.6f', high);
fprintf(fileID, '\n');

fprintf(fileID, 'low_syn');
fprintf(fileID, '\t%.6f', low_syn);
fprintf(fileID, '\n');

fprintf(fileID, 'high_syn');
fprintf(fileID, '\t%.6f', high_syn);
fprintf(fileID, '\n');

for k = 2 : 5
    result = scale_recursion(low, k, m_0, m_1);
%     figure
%     plot(result);
%     title("K = " + k)
    count = 1;
    fprintf(fileID, '\nlow_k %d',k);
    for i = 1:length(result)
        if count == 9
            fprintf(fileID, '\nlow_k %d',k);
            count = 1;
        end
        fprintf(fileID, '\t%.6f', result(i));
        count = count + 1;
    end
    fprintf(fileID, '\n');
end

for k = 2 : 5
    result = wavelet_coef(low, k, m_0, m_1, n_0, n_1);
%     figure
%     plot(result);
%     title("K = " + k)
    count = 1;
    fprintf(fileID, '\nhigh_k %d',k);
    for i = 1:length(result)
        if count == 9
            fprintf(fileID, '\nhigh_k %d',k);
            count = 1;
        end
        fprintf(fileID, '\t%.6f', result(i));
        count = count + 1;
    end
    fprintf(fileID, '\n');
end
%%
function result = scale_recursion(h, k, m_0, m_1)
h = [0;h;0];
step = 2^(k-1);
result = zeros([(length(h)-1) * step, 1]);
for i = 0 : step - 1
    temp = h(1:end-1);
    rest = i;
    for j = 1 : k - 1
        if mod(rest, 2) == 1
            temp = sqrt(2) * m_1 * temp;
            %temp = 2 * m_1 * temp;
        else
            temp = sqrt(2) * m_0 * temp;
            %temp = 2 * m_0 * temp;
        end
        rest = fix(rest / 2);
    end
    result(i+1:step:end) = temp;
end
result = result / sqrt(2);
result = result(step+1:end-step+1);
end

function result = wavelet_coef(h, k, m_0, m_1, n_0, n_1)
h = [0;h;0];
step = 2^(k-1);
result = zeros([(length(h)-1) * step, 1]);
for i = 0 : step - 1
    temp = h(1:end-1);
    rest = i;
    for j = 1 : k - 2
        if mod(rest, 2) == 1
            temp = sqrt(2) * m_1 * temp;
            %temp = 2 * m_1 * temp;
        else
            temp = sqrt(2) * m_0 * temp;%
            %temp = 2 * m_0 * temp;
        end
        rest = fix(rest / 2);
    end
    if mod(rest, 2) == 1
        temp = sqrt(2) * n_1 * temp;%
        %temp = 2 * m_1 * temp;
    else
        temp = sqrt(2) * n_0 * temp;%
        %temp = 2 * m_0 * temp;
    end
    result(i+1:step:end) = temp;
end
result = result / sqrt(2);
result = result(3*2^(k-2)+1:end-3*2^(k-2)+1);
end



function [m_0, m_1] = getScalingMat(h)
h = [0; h; 0];
len_h = length(h) - 1;
m_0 = zeros([len_h, len_h]);
m_1 = zeros([len_h, len_h]);
for i = 1 : len_h
    for j = 1 : len_h
        k = 2 * (i - 1);
        idx = k - j + 2;
        if idx >= 1 && idx <= length(h)
            m_0(i, j) = h(idx);
        end
    end
end

for i = 1 : len_h
    for j = 1 : len_h
        k = 2 * (i - 1) + 1;
        idx = k - j + 2;
        if idx >= 1 && idx <= length(h)
            m_1(i, j) = h(idx);
        end
    end
end

end

function [n_0, n_1] = getWaveletMat(h, g)
len_h = length(h) + 1;
g = padarray(g, [(length(h) - length(g)) / 2 + 1, 0], 0);
n_0 = zeros([len_h, len_h]);
n_1 = zeros([len_h, len_h]);
for i = 1 : len_h
    for j = 1 : len_h
        k = 2 * (i - 1);
        idx = k - j + 2;
        if idx >= 1 && idx <= length(g)
            n_0(i, j) = g(idx);
        end
    end
end

for i = 1 : len_h
    for j = 1 : len_h
        k = 2 * (i - 1) + 1;
        idx = k - j + 2;
        if idx >= 1 && idx <= length(g)
            n_1(i, j) = g(idx);
        end
    end
end

end