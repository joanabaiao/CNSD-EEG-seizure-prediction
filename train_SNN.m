% Computacao Neuronal e Sistemas Difusos 2020/21 - Trabalho 2
% Andre Bernardes (2017248159) & Joana Baiao (2017260526) - MIEB

% train_SNN: criar a rede neuronal shallow e treina-la

function trained_net = train_SNN(T, P, nn_type, hidden_layers, neurons, training_style, training_option, activation_function, delays, optimization)

% CRIAR REDES
if isequal(nn_type, "FF") % Feedforward  
    
    if isequal(training_style, "IL") % Incremental Learning
        net = feedforwardnet(neurons, 'trainc');
        
    elseif isequal(training_style, "BT") % Batch Training
        net = feedforwardnet(neurons, training_option);
    end
    
elseif isequal(nn_type, "LR") % Layer recurrent
    layer_delay = 1:delays;
    
    if isequal(training_style, "IL")
        net = layrecnet(layer_delay, neurons, 'trainc');
        
    elseif isequal(training_style, "BT")
        net = layrecnet(layer_delay, neurons, training_option); % training option = training function
    end    
end
net.numLayers = hidden_layers + 1;

% TRAINING STYLES
if isequal(training_style, "IL")
    for i = 1:length(hidden_layers)
        net.layers{i}.transferFcn = activation_function;
        net.inputWeights{i,:}.learnFcn = training_option; % training option = learning function
    end

elseif isequal(training_style, "BT")
    for i = 1:length(hidden_layers)
        net.layers{i}.transferFcn = activation_function;
    end
end


% PARAMETROS DA REDE
net.performParam.lr = 0.01;      % learning rate
net.trainParam.epochs = 1000;    % maximum epochs
net.trainParam.show = 35;        % show
net.trainParam.goal = 1e-6;      % objective
net.trainParam.min_grad = 10e-6; % minimum performance gradient
net.trainParam.max_fail = 100;   % maximum validation failures
net.performFcn = 'mse';          % criterion

net.divideFcn = 'dividerand';      % random division
net.divideMode = 'sample';
net.divideParam.trainRatio = 0.85; % training ratio
net.divideParam.valRatio = 0.15;   % validation ratio


% DEFINIR OS PESOS DE CADA CLASSE: dar mais importancia à ictal e preictal
ind_interictal = find(T(1,:) == 1);
ind_preictal = find(T(2,:) == 1);
ind_ictal = find(T(3,:) == 1);

total_weight = length(ind_interictal) + length(ind_preictal) + length(ind_ictal);
weight_interictal = total_weight/ length(ind_interictal);
weight_preictal = total_weight/ length(ind_preictal);
weight_ictal = total_weight/ length(ind_ictal);

error_weight = all(T == [1 0 0]') .* weight_interictal + ...
    all(T == [0 1 0]') .* weight_preictal + ...
    all(T == [0 0 1]') .* weight_ictal;
    

% TREINAR REDE
if optimization == 0 % Nenhum
    [trained_net, tr] = train(net, P, T, [], [], error_weight);
    
elseif optimization == 1 % Parallel
    [trained_net, tr] = train(net, P, T, [], [], error_weight, 'UseParallel', 'yes');
    
elseif optimization == 2 % GPU
    [trained_net, tr] = train(net, P, T, [], [], error_weight, 'UseGPU', 'yes');  
    
elseif optimization == 3 % GPU + Parallel
    [trained_net, tr] = train(net, P, T, [], [], error_weight, 'UseParallel', 'yes', 'UseGPU', 'yes');  
end

plotperform(tr);


train_results = sim(trained_net, P);
[~, T_train] = max(T); % vetor com a classificação
[~, train_output] = max(train_results); 

% PLOT DAS MATRIZES DE CONFUSÃO DE TREINO
postprocessing_PP(T_train, train_output);
postprocessing_FT(T_train, train_output);

end



