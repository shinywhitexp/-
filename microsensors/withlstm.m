clear; clc; close all;

%% ========== 参数设置 ==========
grid_size = 32;         % 网格尺寸（32x32）
voxel_size = 0.1e-6;    % 单元尺寸0.1μm
base_porosity = 0.75;   % 基础孔隙率
vox_ratio = 0.15;       % VOX掺杂比例

num_samples = 100;      % 样本数量
num_epochs = 10;        % 训练轮次
validation_ratio = 0.2; % 验证集比例

fprintf('==== 基于LSTM的VOX/LIG性能预测 ====\n');

%% 1. 生成数据集
X_train = zeros(grid_size, grid_size, 1, num_samples);
Y_train = zeros(num_samples, 2);  % 两个目标值：电导率 & 表面积

for i = 1:num_samples
    % 随机扰动参数 ±5%
    porosity_var = base_porosity + 0.05*(2*rand-1);
    ratio_var = vox_ratio + 0.015*(2*rand-1);
    
    % 生成结构
    [~, vox] = generate_simple_heterojunction(grid_size, porosity_var, ratio_var);
    
    % 计算性能指标
    [cond_val, sa_val] = calculate_performance(vox, voxel_size);
    
    % 取中间层作为输入
    mid_layer = vox(:, :, round(grid_size/2));
    X_train(:, :, 1, i) = mid_layer;
    
    Y_train(i, :) = [cond_val, sa_val];
end

%% 2. 数据格式转换为LSTM输入格式 (cell数组，序列长度=grid_size, 特征维度=grid_size)
% 这里每个样本的输入是 grid_size 时间步，每步有 grid_size 特征
% 即：输入是 size [features x timeSteps] = [32 x 32] 的矩阵
XTrain = cell(num_samples, 1);
for i = 1:num_samples
    XTrain{i} = squeeze(X_train(:, :, 1, i))';  % 转置为 [features x timeSteps]
end
YTrain = Y_train;  % size [num_samples x 2]

% 划分训练与验证集
num_val = floor(validation_ratio * num_samples);
val_indices = randperm(num_samples, num_val);
train_indices = setdiff(1:num_samples, val_indices);

XTrain_train = XTrain(train_indices);
YTrain_train = YTrain(train_indices, :);
XValidation = XTrain(val_indices);
YValidation = YTrain(val_indices, :);

%% 3. 构建LSTM模型
inputSize = grid_size;   % 每时间步输入特征数
numHiddenUnits = 128;    % 隐藏单元数
numResponses = 2;        % 输出维度（电导率 & 表面积）

layers = [ ...
    sequenceInputLayer(inputSize, 'Name', 'input')
    lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'lstm')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu')
    fullyConnectedLayer(numResponses, 'Name', 'output')
    regressionLayer('Name', 'regression')];

options = trainingOptions('adam', ...
    'MaxEpochs', num_epochs, ...
    'MiniBatchSize', min(8, length(train_indices)), ...
    'ValidationData', {XValidation, YValidation}, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'InitialLearnRate', 0.001);

%% 4. 训练模型
fprintf('开始训练LSTM模型...\n');
lstm_model = trainNetwork(XTrain_train, YTrain_train, layers, options);

%% 5. 模型可视化
figure;
plot(layerGraph(lstm_model.Layers));
title('LSTM网络结构');

%% 6. 性能预测与误差计算
fprintf('预测测试样本性能...\n');
% 随机选取5个测试样本做预测和对比
num_test = 5;
test_indices = val_indices(1:num_test);

for i = 1:num_test
    xTest = XTrain{test_indices(i)};
    yTrue = YTrain(test_indices(i), :);
    
    yPred = predict(lstm_model, xTest);
    
    err_cond = abs(yPred(1) - yTrue(1)) / yTrue(1) * 100;
    err_sa = abs(yPred(2) - yTrue(2)) / yTrue(2) * 100;
    
    fprintf('样本 %d - 电导率预测: %.3f (实际: %.3f, 误差: %.2f%%)，表面积预测: %.3f (实际: %.3f, 误差: %.2f%%)\n', ...
        test_indices(i), yPred(1), yTrue(1), err_cond, yPred(2), yTrue(2), err_sa);
end

%% ======================= 简化子函数 ========================
function [lig_matrix, vox_matrix] = generate_simple_heterojunction(grid_size, porosity, vox_ratio)
    lig_matrix = rand(grid_size, grid_size, grid_size) > porosity;
    vox_matrix = lig_matrix & (rand(size(lig_matrix)) < vox_ratio);
end

function [conductivity, surface_area] = calculate_performance(vox, res)
    surface_ratio = calculate_surface_ratio(vox);
    filler_ratio = mean(vox(:));
    conductivity = 8e3 * filler_ratio * (1 - (1 - surface_ratio)^2);
    surface_area = 150 * surface_ratio; % m²/g简化估计
end

function surface_ratio = calculate_surface_ratio(vox)
    eroded = imerode(vox, strel('sphere', 1));
    surface_voxels = vox & ~eroded;
    surface_ratio = nnz(surface_voxels) / nnz(vox);
end
