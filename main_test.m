% Computacao Neuronal e Sistemas Difusos 2020/21 - Trabalho 2
% Andre Bernardes (2017248159) & Joana Baiao (2017260526) - MIEB

% main_test: funcao principal que é chamada na interface quando é
%            escolhida a opcao de teste 


function [sens_pred_PP, spec_pred_PP, sens_det_PP, spec_det_PP, sens_pred_FT, spec_pred_FT, sens_det_FT, spec_det_FT,...
    nn_type, seizures_detected, seizures_predicted, n_seizures] = main_test(filename, path)

step = 29;

% CARREGAR REDE
load(fullfile(path, filename));

fprintf('TEST:\n* Pacient ID = %s \n* NN type = %s \n* Features = %d \n', patientID, nn_type, n_features)

% CARREGAR FICHEIRO
if n_features < 29
    load(fullfile(pwd,'Datasets', strcat(patientID,'_',num2str(n_features),'.mat')), 'FeatVectSel', 'Trg');
elseif n_features == 29
    load(fullfile(pwd,'Datasets', strcat(patientID,'.mat')), 'FeatVectSel', 'Trg');
end

% PRÉ-PROCESSAMENTO DE TESTE
[T_test, P_test] = preprocessing(Trg, FeatVectSel, 'test', nn_type, step);


% TESTE
if isequal(nn_type, "FF") || isequal(nn_type, "LR") % Shallow networks
    test_results = sim(trained_net, P_test); 
elseif isequal(nn_type, "CNN") || isequal(nn_type, "LSTM") % Deep networks
    test_results = classify(trained_net, P_test);    
end


% PÓS-PROCESSAMENTO
if isequal(nn_type, "FF") || isequal(nn_type, "LR")
    [~, T] = max(T_test); % vetor com a classificação
    [~, output] = max(test_results); 
elseif isequal(nn_type, "CNN") || isequal(nn_type, "LSTM") % Deep networks       
    T = grp2idx(T_test); % converter vetor categorico para vetor numerico de classificação
    output = grp2idx(test_results);  
end


% MÉTODOS E CALCULO DA SENSITIVIDADE E ESPECIFICIDADE
[sens_pred_PP, spec_pred_PP, sens_det_PP, spec_det_PP] = postprocessing_PP(T, output); % Ponto a ponto
if isequal(nn_type, "FF") || isequal(nn_type, "LR") || isequal(nn_type, "LSTM")
    [sens_pred_FT, spec_pred_FT, sens_det_FT, spec_det_FT] = postprocessing_FT(T, output); % 5 em 10
    [seizures_detected,seizures_predicted, n_seizures]=postprocessing_SD(T, output); % determinar seizures detetadas e previstas
    
elseif isequal(nn_type, "CNN")
    sens_pred_FT = 0;
    spec_pred_FT = 0;
    sens_det_FT = 0;
    spec_det_FT = 0;
    seizures_detected = 0;
    seizures_predicted = 0;
    n_seizures = 0;
end

end