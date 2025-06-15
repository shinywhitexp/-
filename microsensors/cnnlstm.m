clear; clc; close all;

%% ========== 参数设置 ==========
grid_size = 32;           % 网格尺寸（32x32）
voxel_size = 0.1e-6;      % 单元尺寸0.1μm
base_porosity = 0.75;     % 基础孔隙率
vox_ratio = 0.15;         % VOX掺杂比例
num_samples = 200;        % 样本数量
validation_ratio = 0.2;   % 验证集比例

%% ========== 1. 生成数据 ==========
X_data = zeros(grid_size, grid_size, 1, num_samples);
Y_data = zeros(num_samples, 2);  % [电导率, 表面积]

for i = 1:num_samples
    porosity = base_porosity + 0.05 * (2*rand - 1);
    ratio = vox_ratio + 0.02 * (2*rand - 1);
    [~, vox] = generate_simple_heterojunction(grid_size, porosity, ratio);
    [cond, sa] = calculate_performance(vox, voxel_size);
    X_data(:, :, 1, i) = vox(:, :, round(grid_size/2));
    Y_data(i, :) = [cond, sa];
end

%% ========== 2. 格式转换 ==========
XTrain = cell(num_samples, 1);
for i = 1:num_samples
    XTrain{i} = reshape(X_data(:, :, :, i), [grid_size, grid_size, 1]);
end
YTrain = Y_data;

% 拆分训练与验证集
num_val = round(validation_ratio * num_samples);
val_idx = randperm(num_samples, num_val);
train_idx = setdiff(1:num_samples, val_idx);

XTrain_data = XTrain(train_idx);
YTrain_data = YTrain(train_idx, :);
XVal = XTrain(val_idx);
YVal = YTrain(val_idx, :);

%% ========== 3. 构建 CNN + LSTM 网络 (已修复) ==========
inputSize = [grid_size, grid_size, 1];
feature_dim = 64;

% 创建层图
lgraph = layerGraph();

% 输入层
input = sequenceInputLayer(inputSize, 'Name', 'input');
lgraph = addLayers(lgraph, input);

% 序列折叠层
fold = sequenceFoldingLayer('Name', 'fold');
lgraph = addLayers(lgraph, fold);

% CNN 特征提取部分
cnnLayers = [
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')  % 添加额外池化层减少尺寸
    fullyConnectedLayer(feature_dim, 'Name', 'fc_cnn')
];
lgraph = addLayers(lgraph, cnnLayers);

% 序列展开层
unfold = sequenceUnfoldingLayer('Name', 'unfold');
lgraph = addLayers(lgraph, unfold);

% 添加展平层处理CNN输出
flatten = flattenLayer('Name', 'flatten');

% LSTM 回归部分
postLayers = [
    flatten                                    % 添加的展平层
    lstmLayer(64, 'OutputMode', 'last', 'Name', 'lstm')
    fullyConnectedLayer(32, 'Name','fc1')
    reluLayer('Name','relu_final')
    fullyConnectedLayer(2, 'Name','fc_out')
    regressionLayer('Name','regressionoutput')
];
lgraph = addLayers(lgraph, postLayers);

% 连接层（确保所有连接正确）
lgraph = connectLayers(lgraph, 'input', 'fold/in');
lgraph = connectLayers(lgraph, 'fold/out', 'conv1');
lgraph = connectLayers(lgraph, 'fc_cnn', 'unfold/in');
lgraph = connectLayers(lgraph, 'fold/miniBatchSize', 'unfold/miniBatchSize');
lgraph = connectLayers(lgraph, 'unfold/out', 'flatten');  % 连接到展平层

% 网络结构图
figure;
plot(lgraph);
title('CNN + LSTM 网络结构');

%% ========== 4. 训练网络 ==========
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 8, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XVal, YVal}, ...
    'Plots','training-progress', ...
    'Verbose',false);

fprintf("== 开始训练 ==\n");
net = trainNetwork(XTrain_data, YTrain_data, lgraph, options);

%% ========== 5. 测试与可视化 ==========
fprintf("\n== 预测结果展示 ==\n");
num_test = 5;
for i = 1:num_test
    xTest = XTrain{val_idx(i)};
    yTrue = YTrain(val_idx(i), :);
    yPred = predict(net, {xTest});
    fprintf('样本 %d - 电导率预测: %.2f (实际: %.2f), 表面积预测: %.2f (实际: %.2f)\n', ...
        i, yPred(1), yTrue(1), yPred(2), yTrue(2));
end

%% ========== 6. 绘图：预测性能散点图 ==========
YPred_all = zeros(num_val, 2);
for i = 1:num_val
    x = XTrain{val_idx(i)};
    YPred_all(i,:) = predict(net, {x});
end

figure;
subplot(1,2,1);
scatter(YVal(:,1), YPred_all(:,1), 30, 'filled');
xlabel('真实电导率'); ylabel('预测电导率');
title('电导率预测性能'); grid on; axis equal; hold on;
plot(xlim, xlim, 'r--');

subplot(1,2,2);
scatter(YVal(:,2), YPred_all(:,2), 30, 'filled');
xlabel('真实表面积'); ylabel('预测表面积');
title('表面积预测性能'); grid on; axis equal; hold on;
plot(xlim, xlim, 'r--');

%% ========== 子函数 ==========
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