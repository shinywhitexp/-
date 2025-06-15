clear; clc; close all;

%% ========== 参数设置 ==========
grid_size = 32;         % 网格尺寸
voxel_size = 0.1e-6;    % 单元尺寸 = 0.1μm
base_porosity = 0.75;   % 基础孔隙率
vox_ratio = 0.15;       % VOX掺杂比例

num_samples = 100;      % 样本数量
num_epochs = 10;        % 训练轮数
validation_ratio = 0.2; % 验证集比例

%% ========== 数据生成 ==========
[X_train, Y_train] = generate_training_data(grid_size, base_porosity, vox_ratio, voxel_size, num_samples);

% 重塑输入为序列（每个图像拉直为1024维向量）
X_seq = reshape(X_train, [grid_size*grid_size, num_samples])';

%% ========== Transformer模型定义 ==========
inputSize = grid_size * grid_size;
numHiddenUnits = 64;
numHeads = 4;
numResponses = 2;

layers = [
    sequenceInputLayer(inputSize, 'Name', 'input')
    transformerLayer(numHeads, numHiddenUnits, 'Name', 'transformer')
    fullyConnectedLayer(32, 'Name', 'fc1')
    reluLayer('Name', 'relu')
    fullyConnectedLayer(numResponses, 'Name', 'fc2')
    regressionLayer('Name', 'regression')
];

% 数据分割
num_val = floor(validation_ratio * num_samples);
val_indices = randperm(num_samples, num_val);
train_indices = setdiff(1:num_samples, val_indices);

XTrain = X_seq(train_indices, :)';
YTrain = Y_train(train_indices, :);
XVal = X_seq(val_indices, :)';
YVal = Y_train(val_indices, :);

% 训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', num_epochs, ...
    'MiniBatchSize', min(8, length(train_indices)), ...
    'ValidationData', {XVal, YVal}, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'InitialLearnRate', 1e-3);

%% ========== 训练模型 ==========
transformer_net = trainNetwork(XTrain, YTrain, layers, options);
save('transformer_model.mat', 'transformer_net', '-v7.3');

%% ========== 测试预测 ==========
[~, vox_test] = generate_simple_heterojunction(grid_size, base_porosity, vox_ratio);
test_input = reshape(vox_test(:,:,round(grid_size/2)), 1, grid_size*grid_size);

predicted = predict(transformer_net, test_input');
[actual_cond, actual_sa] = calculate_performance(vox_test, voxel_size);

fprintf('\n=== Transformer预测性能 ===\n');
fprintf('电导率预测: %.3f S/m (实际: %.3f S/m)\n', predicted(1), actual_cond);
fprintf('表面积预测: %.3f m^2/g (实际: %.3f m^2/g)\n', predicted(2), actual_sa);

%% ========== 子函数 ==========
function [X_train, Y_train] = generate_training_data(grid_size, base_porosity, vox_ratio, voxel_size, num_samples)
    X_train = zeros(grid_size, grid_size, 1, num_samples);
    Y_train = zeros(num_samples, 2);
    for i = 1:num_samples
        porosity_var = base_porosity + 0.05*(2*rand - 1);
        ratio_var = vox_ratio + 0.015*(2*rand - 1);
        [~, vox] = generate_simple_heterojunction(grid_size, porosity_var, ratio_var);
        mid_layer = vox(:,:,round(grid_size/2));
        X_train(:,:,1,i) = mid_layer;
        [cond_val, sa_val] = calculate_performance(vox, voxel_size);
        Y_train(i,:) = [cond_val, sa_val];
    end
end

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
