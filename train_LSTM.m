% Computacao Neuronal e Sistemas Difusos 2020/21 - Trabalho 2
% Andre Bernardes (2017248159) & Joana Baiao (2017260526) - MIEB

% train_LSTM: criar a rede neuronal do tipo LSTM e treinar

function trained_net = train_LSTM(T, P, hidden_units, n_features, epochs, solver)

% DEFINIR AS LAYERS DA REDE
layers = [ ...
    sequenceInputLayer(n_features)
    lstmLayer(hidden_units,'OutputMode','last') % camada intermédia
    fullyConnectedLayer(3) % numero de classes
    softmaxLayer
    classificationLayer];

% ESPECIFICAR AS OPCOES DE TREINO
options = trainingOptions(solver, ... 
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs', epochs,...
    'GradientThreshold', 1, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

% TREINAR A REDE
trained_net = trainNetwork(P, T, layers, options);

% RESULTADOS TREINO
train_results = classify(trained_net, P); 
T_train = grp2idx(T); % converter vetor categorico para vetor numerico de classificação
train_output = grp2idx(train_results);  

% PLOT DAS MATRIZES DE CONFUSÃO DE TREINO
postprocessing_PP(T_train, train_output);
postprocessing_FT(T_train, train_output);


end






 