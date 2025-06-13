
% 适用于：16GB内存/核显/MATLAB2021a
clear; clc; close all;

%% ========== 参数设置 ==========
grid_size = 32;         % 优化后的网格尺寸（32x32）
voxel_size = 0.1e-6;   % 单元尺寸保持0.1μm
base_porosity = 0.75;   % 目标孔隙率
vox_ratio = 0.15;      % VOX掺杂比例

% 机器学习参数
num_samples = 100;       % 样本数量（较低要求）
num_epochs = 10;        % 训练轮次（保持较低）
validation_ratio = 0.2; % 验证集比例

%% ========== 主程序开始 ==========
fprintf('==== 轻量级VOX/LIG-CNN建模系统启动 ====\n');

%% 1. 生成基础结构
[lig_base, vox_base] = generate_simple_heterojunction(grid_size, base_porosity, vox_ratio);

% 可视化基础结构
plot_single_layer(vox_base(:,:,round(grid_size/2)), '基础结构');

%% 2. 创建用于CNN的训练数据集
fprintf('生成%d个训练样本...\n', num_samples);

% 预分配数据
X_train = zeros(grid_size, grid_size, 1, num_samples); % 使用单层结构作为输入
Y_train = zeros(num_samples, 2); % [电导率, 表面积]

progress = 0;
fprintf('进度: 00%%');

for i = 1:num_samples
    % 参数变化（±10%）
    porosity_var = base_porosity + 0.05*(2*rand-1);
    ratio_var = vox_ratio + 0.015*(2*rand-1);
    
    % 生成简化结构
    [~, vox] = generate_simple_heterojunction(grid_size, porosity_var, ratio_var);
    
    % 计算性能（简化模型）
    [cond_val, sa_val] = calculate_performance(vox, voxel_size);
    
    % 存储数据 - 使用中间层作为输入（减少3D计算需求）
    mid_layer = vox(:,:,round(grid_size/2));
    X_train(:, :, 1, i) = mid_layer;
    Y_train(i, :) = [cond_val, sa_val];
    
    % 进度显示
    if mod(i, ceil(num_samples/10)) == 0
        progress = round(100*i/num_samples);
        fprintf('\b\b\b%02d%%', progress);
    end
end
fprintf('\b\b\b完成!\n');

%% 3. 构建轻量级CNN模型
fprintf('构建轻量级CNN模型...\n');

% 输入尺寸 (32x32像素单通道)
input_size = [grid_size, grid_size, 1];

% 创建CNN架构
layers = [
    imageInputLayer(input_size, 'Name', 'input', 'Normalization', 'none')
    
    % 特征提取
    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    
    % 分类/回归
    fullyConnectedLayer(32, 'Name', 'fc1')
    reluLayer('Name', 'relu3')
    
    fullyConnectedLayer(2, 'Name', 'output')  % 两个输出节点
    regressionLayer('Name', 'regression')
];

% 分割训练/验证集
num_val = floor(validation_ratio * num_samples);
val_indices = randperm(num_samples, num_val);
train_indices = setdiff(1:num_samples, val_indices);

% 训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', num_epochs, ...
    'MiniBatchSize', min(8, num_samples - num_val), ...
    'ValidationData', {X_train(:,:,:,val_indices), Y_train(val_indices, :)}, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'InitialLearnRate', 0.001);

%% 4. 训练轻量级CNN
fprintf('训练CNN模型 (%d epochs)...\n', num_epochs);
cnn_model = trainNetwork(X_train(:,:,:,train_indices), Y_train(train_indices, :), layers, options);

% 保存模型（精简版本）
save('cnn_model.mat', 'cnn_model', '-v7.3');
fprintf('模型训练完成并保存为cnn_model.mat\n');

%% 5. CNN性能预测
fprintf('使用CNN进行性能预测...\n');

% 生成测试结构
[~, vox_test] = generate_simple_heterojunction(grid_size, base_porosity, vox_ratio);
mid_test = vox_test(:,:,round(grid_size/2));
mid_test = reshape(mid_test, [grid_size, grid_size, 1]);

% CNN预测
predicted = predict(cnn_model, mid_test);

% 实际性能
[actual_cond, actual_sa] = calculate_performance(vox_test, voxel_size);

% 输出结果
fprintf('\n=== CNN预测性能 ===\n');
fprintf('电导率预测: %.3f S/m (实际: %.3f S/m, 误差: %.1f%%)\n',...
        predicted(1), actual_cond, abs(predicted(1)/actual_cond-1)*100);
fprintf('表面积预测: %.3f m²/g (实际: %.3f m²/g, 误差: %.1f%%)\n',...
        predicted(2), actual_sa, abs(predicted(2)/actual_sa-1)*100);

%% 6. 性能优化分析（参数扫描）
fprintf('\n=== 参数优化分析 ===\n');
porosity_vals = linspace(0.65, 0.85, 5);
results = zeros(length(porosity_vals), 2);

for i = 1:length(porosity_vals)
    [~, vox] = generate_simple_heterojunction(grid_size, porosity_vals(i), vox_ratio);
    mid = vox(:,:,round(grid_size/2));
    mid = reshape(mid, [grid_size, grid_size, 1]);
    
    % 使用CNN预测性能
    results(i, :) = predict(cnn_model, mid);
    fprintf('孔隙率=%.2f: 电导率=%.3f S/m, 表面积=%.3f m²/g\n',...
            porosity_vals(i), results(i,1), results(i,2));
end

% 可视化优化结果
figure('Position', [100,100,800,300]);
subplot(1,2,1);
plot(porosity_vals, results(:,1), 'o-');
title('孔隙率对电导率的影响');
xlabel('孔隙率'); ylabel('电导率 (S/m)');
grid on;

subplot(1,2,2);
plot(porosity_vals, results(:,2), 'o-');
title('孔隙率对表面积的影响');
xlabel('孔隙率'); ylabel('表面积 (m²/g)');
grid on;
saveas(gcf, 'porosity_optimization_cnn.png');

%% ========== 简化子函数 ==========
function [lig_matrix, vox_matrix] = generate_simple_heterojunction(grid_size, porosity, vox_ratio)
    % 简化的三维结构生成器
    lig_matrix = rand(grid_size, grid_size, grid_size) > porosity;
    vox_matrix = lig_matrix & (rand(size(lig_matrix)) < vox_ratio);
end

function [conductivity, surface_area] = calculate_performance(vox, res)
    % 简化的性能计算（考虑孔隙连通性）
    surface_ratio = calculate_surface_ratio(vox);
    
    % 电导率模型：考虑体积分数和连通性
    filler_ratio = mean(vox(:));
    conductivity = 8e3 * filler_ratio * (1 - (1 - surface_ratio)^2);
    
    % 表面积模型（物理单位）
    surface_area = 150 * surface_ratio; % m²/g简化估计
end

function surface_ratio = calculate_surface_ratio(vox)
    % 计算表面积占比（代替完整计算）
    % 通过腐蚀操作检测表面体素
    eroded = imerode(vox, strel('sphere', 1));
    surface_voxels = vox & ~eroded;
    surface_ratio = nnz(surface_voxels) / nnz(vox);
end

function plot_single_layer(layer_data, plot_title)
    % 可视化单层结构
    figure('Position', [100,100,400,300]);
    imagesc(layer_data);
    colormap([0.8,0.8,0.8; 0.2,0.6,0.9]); % LIG灰色，VOX蓝色
    title(plot_title);
    axis equal;
    axis off;
end