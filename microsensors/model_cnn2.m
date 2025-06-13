
% 彻底解决预测误差问题
clear; clc; close all; rng(42);

%% ========== 参数设置 ==========
grid_size = 36;          % 网格尺寸
voxel_size = 0.1e-6;    % 单元尺寸 (0.1μm)
base_porosity = 0.75;    % 目标孔隙率
vox_ratio = 0.15;       % VOX掺杂比例

% 机器学习参数
num_samples = 200;      % 增加样本数量
num_epochs = 30;        % 增加训练轮次
validation_ratio = 0.2; % 验证集比例

% 物理模型参数
density = 6.7e3;        % 材料密度 (kg/m³)

%% ========== 主程序开始 ==========
fprintf('==== VO/LIG高精度建模系统 ====\n');
fprintf('网格尺寸: %dx%dx%d | 样本数: %d\n', grid_size, grid_size, grid_size, num_samples);

%% 1. 精确结构生成
fprintf('生成精确结构...\n');
[lig, vox] = generate_accurate_structure(grid_size, base_porosity, vox_ratio);

% 计算结构参数
filler_ratio = sum(vox(:)) / numel(vox);
fprintf('材料参数: 孔隙率=%.3f, VOX比例=%.3f, 填充比例=%.3f\n',...
        1 - filler_ratio, vox_ratio, filler_ratio);

% 可视化结构
plot_accurate_structure(vox);

%% 2. 精确性能计算
fprintf('计算精确性能...\n');
[cond_real, sa_real] = calculate_accurate_performance(vox, voxel_size, density);
fprintf('实际性能: 电导率=%.2f S/m, 表面积=%.2f m²/g\n', cond_real, sa_real);

%% 3. 创建高精度数据集
fprintf('生成%d个高精度样本...\n', num_samples);

% 预分配数据
X_train = zeros(grid_size, grid_size, 6, num_samples); % 3通道输入
Y_train = zeros(num_samples, 2); % [电导率, 表面积]

% 记录数据范围用于归一化
cond_min = Inf; cond_max = -Inf;
sa_min = Inf; sa_max = -Inf;

progress_bar = waitbar(0, '生成样本中...');

for i = 1:num_samples
    % 参数变化 - 更广泛的参数空间
    porosity_var = max(0.60, min(0.90, base_porosity + 0.08*randn()));
    ratio_var = max(0.10, min(0.25, vox_ratio + 0.04*randn()));
    
    % 生成结构
    [~, vox_sample] = generate_accurate_structure(grid_size, porosity_var, ratio_var);
    
    % 计算精确性能
    [cond_val, sa_val] = calculate_accurate_performance(vox_sample, voxel_size, density);
    
    % 记录数据范围
    cond_min = min(cond_min, cond_val);
    cond_max = max(cond_max, cond_val);
    sa_min = min(sa_min, sa_val);
    sa_max = max(sa_max, sa_val);
    
    % 提取高级特征
    X_train(:,:,:,i) = extract_advanced_features(vox_sample);
    
    % 保存性能参数
    Y_train(i,:) = [cond_val, sa_val];
    
    % 进度更新
    progress = i/num_samples;
    waitbar(progress, progress_bar, sprintf('生成样本 %d/%d (%.0f%%)', i, num_samples, progress*100));
    
    % 定期显示样本信息
    if mod(i, 20) == 0
        fprintf('样本 %d: 电导率=%.2f, 表面积=%.2f\n', i, cond_val, sa_val);
    end
end
close(progress_bar);
fprintf('数据集创建完成! 范围: 电导率[%.2f, %.2f] S/m, 表面积[%.2f, %.2f] m²/g\n',...
        cond_min, cond_max, sa_min, sa_max);

% 归一化目标值 - 解决数量级差异问题
Y_train_normalized = normalize_performance(Y_train, cond_min, cond_max, sa_min, sa_max);

%% 4. 构建高级CNN模型
fprintf('构建高级CNN模型...\n');

% 输入尺寸
input_size = [grid_size, grid_size, 6];

% 更深的CNN架构
layers = [
    imageInputLayer(input_size, 'Name', 'input', 'Normalization', 'rescale-zero-one')
    
    % 特征提取模块
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    
    % 深度特征提取
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv5')
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv6')
    batchNormalizationLayer('Name', 'bn6')
    reluLayer('Name', 'relu6')
    
    % 回归头
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu7')
    dropoutLayer(0.3, 'Name', 'dropout1')
    
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu8')
    
    fullyConnectedLayer(32, 'Name', 'fc3')
    reluLayer('Name', 'relu9')
    
    fullyConnectedLayer(2, 'Name', 'output')
    regressionLayer('Name', 'regression')
];

% 训练选项 - 添加学习率调度
options = trainingOptions('adam', ...
    'MaxEpochs', num_epochs, ...
    'MiniBatchSize', 16, ...
    'ValidationData', {X_train(:,:,:,1:floor(num_samples*validation_ratio)), ...
                       Y_train_normalized(1:floor(num_samples*validation_ratio), :)}, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.7, ...
    'LearnRateDropPeriod', 7, ...
    'Verbose', true, ...
    'L2Regularization', 1e-4);

%% 5. 训练高级CNN模型
fprintf('训练CNN模型 (%d epochs)...\n', num_epochs);
t0 = tic;
cnn_model = trainNetwork(X_train, Y_train_normalized, layers, options);
training_time = toc(t0);
fprintf('模型训练完成! 耗时: %.1f秒\n', training_time);

% 保存模型
save('accurate_cnn_model.mat', 'cnn_model', '-v7.3');

%% 6. 高精度预测
fprintf('执行高精度预测...\n');

% 生成测试结构
[~, vox_test] = generate_accurate_structure(grid_size, base_porosity, vox_ratio);
features_test = extract_advanced_features(vox_test);

% CNN预测 (归一化空间)
predicted_normalized = predict(cnn_model, features_test);

% 反归一化到真实物理量
predicted_cond = denormalize_value(predicted_normalized(1), cond_min, cond_max);
predicted_sa = denormalize_value(predicted_normalized(2), sa_min, sa_max);

% 实际性能
[actual_cond, actual_sa] = calculate_accurate_performance(vox_test, voxel_size, density);

% 输出结果
fprintf('\n=== 高精度预测 ===\n');
fprintf('材料属性: 孔隙率=%.3f, VOX比例=%.3f\n', base_porosity, vox_ratio);
fprintf('电导率: 预测=%.2f S/m, 实际=%.2f S/m, 误差=%.2f%%\n',...
        predicted_cond, actual_cond, abs(predicted_cond/actual_cond-1)*100);
fprintf('表面积: 预测=%.2f m²/g, 实际=%.2f m²/g, 误差=%.2f%%\n',...
        predicted_sa, actual_sa, abs(predicted_sa/actual_sa-1)*100);

% 可视化预测精度
visualize_prediction_accuracy(Y_train, predict(cnn_model, X_train),...
                               cond_min, cond_max, sa_min, sa_max,...
                               [predicted_cond, predicted_sa],...
                               [actual_cond, actual_sa]);

fprintf('\n==== 系统运行完成 ====\n');

%% ========== 关键子函数 ==========
function [lig, vox] = generate_accurate_structure(grid_size, porosity, ratio)
    % 高精度结构生成器
    
    % 1. 创建LIG骨架
    lig = rand(grid_size, grid_size, grid_size) > porosity;
    
    % 2. 添加VOX颗粒 - 使用聚集模型
    % 确定VOX簇的数量和大小
    num_clusters = max(1, round(ratio * grid_size^3 / 10));
    vox = false(size(lig));
    
    for i = 1:num_clusters
        % 随机簇中心
        center_x = randi(grid_size);
        center_y = randi(grid_size);
        center_z = randi(grid_size);
        
        % 随机簇半径 (1-3体素)
        radius = 1 + 2*rand();
        
        % 创建球形簇
        [X, Y, Z] = meshgrid(1:grid_size, 1:grid_size, 1:grid_size);
        dist = sqrt((X-center_x).^2 + (Y-center_y).^2 + (Z-center_z).^2);
        cluster = dist <= radius;
        
        vox = vox | cluster;
    end
    
    % 确保只保留连接在LIG骨架上的VOX颗粒
    dilated_lig = imdilate(lig, strel('sphere', 2));
    vox = vox & dilated_lig;
    
    % 组合结构
    full_structure = lig | vox;
    
    % 只保留主连通簇
    CC = bwconncomp(full_structure, 6);
    if CC.NumObjects > 0
        cluster_sizes = cellfun(@numel, CC.PixelIdxList);
        [~, idx] = max(cluster_sizes);
        
        mask = false(size(full_structure));
        mask(CC.PixelIdxList{idx}) = true;
        
        lig = mask & lig;
        vox = mask & vox;
    end
end

function [conductivity, surface_area] = calculate_accurate_performance(vox, res, density)
    % 高精度性能计算
    
    % 1. 基本参数
    total_voxels = sum(vox(:));
    grid_volume = size(vox,1) * size(vox,2) * size(vox,3);
    filler_ratio = total_voxels / grid_volume;
    
    % 2. 电导率模型 (基于导电通路)
    CC = bwconncomp(vox, 6);
    if CC.NumObjects == 0
        conductivity = 0;
    else
        % 最大簇比例
        max_cluster = max(cellfun(@numel, CC.PixelIdxList));
        connectivity = max_cluster / grid_volume;
        
        % 简化渗流模型
        pc = 0.15; % 渗流阈值
        mu = 1.5;  % 临界指数
        
        if connectivity < pc
            conductivity = 1e-3; % 极小值
        else
            % 导电率模型: σ = k * (ϕ - ϕc)^μ
            conductivity = 1e4 * filler_ratio * ((connectivity - pc)/ (1 - pc))^mu;
        end
    end
    
    % 3. 精确表面积计算
    % 识别表面体素 (暴露面)
    surface_voxels = false(size(vox));
    
    % 检查所有方向
    for z = 1:size(vox,3)
        for y = 1:size(vox,2)
            for x = 1:size(vox,1)
                if ~vox(x,y,z), continue; end
                
                % 6邻域检查
                neighbors = 0;
                if x == 1 || ~vox(x-1,y,z), neighbors = neighbors + 1; end
                if x == size(vox,1) || ~vox(x+1,y,z), neighbors = neighbors + 1; end
                if y == 1 || ~vox(x,y-1,z), neighbors = neighbors + 1; end
                if y == size(vox,2) || ~vox(x,y+1,z), neighbors = neighbors + 1; end
                if z == 1 || ~vox(x,y,z-1), neighbors = neighbors + 1; end
                if z == size(vox,3) || ~vox(x,y,z+1), neighbors = neighbors + 1; end
                
                if neighbors > 0
                    surface_voxels(x,y,z) = true;
                end
            end
        end
    end
    
    % 正确计算表面积
    surface_voxel_count = sum(surface_voxels(:));
    voxel_area = res^2; % 单个体素的表面积
    total_surface = surface_voxel_count * voxel_area; % 总表面积 (m²)
    
    % 计算比表面积 (m²/g)
    filled_volume = total_voxels * (res)^3; % 填充体积 (m³)
    mass = filled_volume * density;        % 填充部分的质量 (kg)
    surface_area = total_surface / (mass / 1000); % 比表面积 (m²/g)
end

function features = extract_advanced_features(vox)
    % 高级特征提取
    
    % 三视图
    xy_view = max(vox, [], 3);  % XY投影
    xz_view = squeeze(max(vox, [], 2)); % XZ投影
    yz_view = squeeze(max(vox, [], 1)); % YZ投影
    
    % 标准化尺寸
    target_size = [size(vox,1), size(vox,2)];
    xz_view = imresize(xz_view, target_size);
    yz_view = imresize(yz_view, target_size);
    
    % 形态学特征
    perim_xy = bwperim(xy_view);
    perim_xz = bwperim(xz_view);
    perim_yz = bwperim(yz_view);
    
    % 创建特征立方体 (6通道)
    features = cat(3, ...
        xy_view, perim_xy, ...
        xz_view, perim_xz, ...
        yz_view, perim_yz);
end

function plot_accurate_structure(vox)
    % 高精度结构可视化
    figure('Position', [100,100,1000,400]);
    
    % XY平面
    subplot(1,4,1);
    imagesc(vox(:,:,round(size(vox,3)/2)));
    title('XY平面');
    axis equal; axis off;
    colormap(jet);
    
    % XZ平面
    subplot(1,4,2);
    imagesc(squeeze(vox(round(size(vox,1)/2),:,:))');
    title('XZ平面');
    axis equal; axis off;
    
    % YZ平面
    subplot(1,4,3);
    imagesc(squeeze(vox(:,round(size(vox,2)/2),:))');
    title('YZ平面');
    axis equal; axis off;
    
    % 3D可视化
    subplot(1,4,4);
    vox = double(vox);
    p = patch(isosurface(vox,0.5));
    p.FaceColor = 'red';
    p.EdgeColor = 'none';
    daspect([1 1 1]);
    view(3);
    axis tight;
    camlight;
    lighting gouraud;
    title('3D结构');
    
    saveas(gcf, 'accurate_structure.png');
end

function Y_norm = normalize_performance(Y, cond_min, cond_max, sa_min, sa_max)
    % 归一化性能参数
    % 使用min-max归一化到[0,1]范围
    
    % 电导率归一化
    cond_norm = (Y(:,1) - cond_min) / (cond_max - cond_min);
    
    % 表面积归一化
    sa_norm = (Y(:,2) - sa_min) / (sa_max - sa_min);
    
    Y_norm = [cond_norm, sa_norm];
end

function value = denormalize_value(norm_val, min_val, max_val)
    % 反归一化
    value = norm_val * (max_val - min_val) + min_val;
end

function visualize_prediction_accuracy(Y_train, Y_pred_norm, cond_min, cond_max, sa_min, sa_max, predicted, actual)
    % 可视化预测精度
    
    % 反归一化训练集预测值
    Y_pred_cond = denormalize_value(Y_pred_norm(:,1), cond_min, cond_max);
    Y_pred_sa = denormalize_value(Y_pred_norm(:,2), sa_min, sa_max);
    
    figure('Position', [100,100,1000,400]);
    
    % 电导率预测精度
    subplot(1,2,1);
    scatter(Y_train(:,1), Y_pred_cond, 20, 'filled');
    hold on;
    plot(actual(1), predicted(1), 'ro', 'MarkerSize', 12, 'LineWidth', 2);
    plot([min(Y_train(:,1)) max(Y_train(:,1))], [min(Y_train(:,1)) max(Y_train(:,1))], 'r--');
    title('电导率预测精度');
    xlabel('真实值 (S/m)');
    ylabel('预测值 (S/m)');
    grid on;
    legend('训练样本', '测试点', '理想线', 'Location', 'northwest');
    
    % 表面积预测精度
    subplot(1,2,2);
    scatter(Y_train(:,2), Y_pred_sa, 20, 'filled');
    hold on;
    plot(actual(2), predicted(2), 'ro', 'MarkerSize', 12, 'LineWidth', 2);
    plot([min(Y_train(:,2)) max(Y_train(:,2))], [min(Y_train(:,2)) max(Y_train(:,2))], 'r--');
    title('表面积预测精度');
    xlabel('真实值 (m²/g)');
    ylabel('预测值 (m²/g)');
    grid on;
    legend('训练样本', '测试点', '理想线', 'Location', 'northwest');
    
    saveas(gcf, 'prediction_accuracy.png');
end