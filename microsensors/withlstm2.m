clear; clc; close all;

grid_size = 32;
time_steps = 32;
channels = 1;
num_samples = 250;
validation_ratio = 0.2;

% 生成数据（同之前）
X_data = zeros(grid_size, grid_size, channels, time_steps, num_samples);
Y_data = zeros(num_samples, 2);

for i = 1:num_samples
    porosity_var = 0.75 + 0.05*(2*rand-1);
    ratio_var = 0.15 + 0.015*(2*rand-1);
    [~, vox] = generate_simple_heterojunction(grid_size, porosity_var, ratio_var);
    [cond_val, sa_val] = calculate_performance(vox, 0.1e-6);
    for t = 1:time_steps
        slice_idx = t;
        X_data(:, :, 1, t, i) = vox(:, :, slice_idx);
    end
    Y_data(i, :) = [cond_val, sa_val];
end

num_val = floor(validation_ratio * num_samples);
val_idx = randperm(num_samples, num_val);
train_idx = setdiff(1:num_samples, val_idx);

XTrain = X_data(:, :, :, :, train_idx);
YTrain = Y_data(train_idx, :);
XValidation = X_data(:, :, :, :, val_idx);
YValidation = Y_data(val_idx, :);

% 转换数据格式，展平每帧图像，形成序列输入
XTrainSeq = cell(length(train_idx), 1);
for i = 1:length(train_idx)
    temp_seq = zeros(grid_size*grid_size, time_steps);
    for t = 1:time_steps
        img = XTrain(:, :, 1, t, i);
        temp_seq(:, t) = img(:);
    end
    XTrainSeq{i} = temp_seq;
end

XValSeq = cell(length(val_idx), 1);
for i = 1:length(val_idx)
    temp_seq = zeros(grid_size*grid_size, time_steps);
    for t = 1:time_steps
        img = XValidation(:, :, 1, t, i);
        temp_seq(:, t) = img(:);
    end
    XValSeq{i} = temp_seq;
end

layers = [
    sequenceInputLayer(grid_size*grid_size, 'Name', 'input')
    lstmLayer(128, 'OutputMode', 'last', 'Name', 'lstm')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(2, 'Name', 'output')
    regressionLayer('Name', 'regression')];

options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 8, ...
    'ValidationData', {XValSeq, YValidation}, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true);

fprintf('开始训练模型...\n');
net = trainNetwork(XTrainSeq, YTrain, layers, options);

% --- 预测测试样本 ---
fprintf('验证集预测与误差：\n');
num_test = 5;
% === 生成性能对比图 ===
num_plot = length(XValSeq);  % 验证集大小

YPred_all = zeros(num_plot, 2);
for i = 1:num_plot
    YPred_all(i, :) = predict(net, XValSeq{i});
end

figure;
subplot(1,2,1);
scatter(YValidation(:,1), YPred_all(:,1), 'filled');
hold on;
plot([min(YValidation(:,1)) max(YValidation(:,1))], [min(YValidation(:,1)) max(YValidation(:,1))], 'r--', 'LineWidth',1.5);
xlabel('真实电导率');
ylabel('预测电导率');
title('电导率预测 vs 真实');
grid on;
axis equal;

subplot(1,2,2);
scatter(YValidation(:,2), YPred_all(:,2), 'filled');
hold on;
plot([min(YValidation(:,2)) max(YValidation(:,2))], [min(YValidation(:,2)) max(YValidation(:,2))], 'r--', 'LineWidth',1.5);
xlabel('真实表面积');
ylabel('预测表面积');
title('表面积预测 vs 真实');
grid on;
axis equal;

for i = 1:num_test
    idx = i;  % 这里你可以随机取，也可以按顺序
    xTest = XValSeq{idx};
    yTrue = YValidation(idx, :);
    yPred = predict(net, xTest);
    
    err_cond = abs(yPred(1) - yTrue(1)) / yTrue(1) * 100;
    err_sa = abs(yPred(2) - yTrue(2)) / yTrue(2) * 100;
    
    fprintf('样本 %d - 电导率预测: %.3f (实际: %.3f, 误差: %.2f%%)，表面积预测: %.3f (实际: %.3f, 误差: %.2f%%)\n', ...
        idx, yPred(1), yTrue(1), err_cond, yPred(2), yTrue(2), err_sa);
end

% ======================= 子函数 ===========================
function [lig_matrix, vox_matrix] = generate_simple_heterojunction(grid_size, porosity, vox_ratio)
    lig_matrix = rand(grid_size, grid_size, grid_size) > porosity;
    vox_matrix = lig_matrix & (rand(size(lig_matrix)) < vox_ratio);
end

function [conductivity, surface_area] = calculate_performance(vox, res)
    surface_ratio = calculate_surface_ratio(vox);
    filler_ratio = mean(vox(:));
    conductivity = 8e3 * filler_ratio * (1 - (1 - surface_ratio)^2);
    surface_area = 150 * surface_ratio;
end

function surface_ratio = calculate_surface_ratio(vox)
    eroded = imerode(vox, strel('sphere', 1));
    surface_voxels = vox & ~eroded;
    surface_ratio = nnz(surface_voxels) / nnz(vox);
end
