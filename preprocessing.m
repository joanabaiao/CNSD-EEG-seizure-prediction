% Computacao Neuronal e Sistemas Difusos 2020/21 - Trabalho 2
% Andre Bernardes (2017248159) & Joana Baiao (2017260526) - MIEB

% preprocessing: nesta funcao é feito o processamento inicial dos dados: 
%                - alteracao do vetor de targets considerando as 3 classes; 
%                - divisao dos dados: grupo de teste e treino;
%                - balanceamento dos dados no grupo de treino;
%                - se aplicavél, é chamada a funcao de processamento das DNN

function [processed_T, processed_P] = preprocessing(Trg, FeatVectSel, choice, nn_type, step)

% ALTERAR O TARGET (3 CLASSES: 1- interictal; 2- preictal; 3- ictal)
target = ones(length(Trg), 1);
seizures = [];
for i = 1:length(Trg)
    if Trg(i) == 1
        target(i) = 3; % ictal (seizure)         
        if Trg(i-1) == 0 % posição onde se inicia a seizure
            target(i-900:i-1) = 2; % preictal
            ictal_start = i;
        elseif Trg(i+1) == 0 && i<length(Trg)
            ictal_end = i;
            seizures = [seizures [ictal_start; ictal_end]]; 
            % guardar as posicoes das seizures (linha 1: inicio; linha 2: final)  
        end
    end
end

% CRIAR MATRIZ T
T = [];
for i = 1 : length(target)
    if target(i) == 1
        T(:,i) = [1 0 0]';
    elseif target(i) == 2
        T(:,i) = [0 1 0]';
    elseif target(i) == 3
        T(:,i) = [0 0 1]';
    end
end

% CRIAR MATRIZ 
P = FeatVectSel'; % Cada coluna será um vetor com as features


% DIVIDIR O DATASET
% 80% (treino + validacao) + 20% (teste)
n_seizures = length(seizures); % numero de seizures
seizures_train = round(n_seizures* 0.8, 0);
%seizures_test = n_seizures - seizures_train;


% TESTE
if isequal(choice, 'test') % No conjunto de teste nao se faz o equilibrio das classes

    test_P = P(:, seizures(2, seizures_train)+1:end);
    test_T = T(:, seizures(2, seizures_train)+1:end);
    
    processed_T = test_T;
    processed_P = test_P;
    
% TREINO
elseif isequal(choice, 'train')
    
    train_P = P(:, 1:seizures(2, seizures_train));
    train_T = T(:, 1:seizures(2, seizures_train));        
       
    % Guardar as posicoes correspondentes a cada classe
    ind_interictal = find(train_T(1,:) == 1);
    ind_preictal = find(train_T(2,:) == 1);
    ind_ictal = find(train_T(3,:) == 1);

    % Equilibrar o numero de pontos das classes no conjunto de treino (undersampling)
    interictal_rand = ind_interictal(randperm(length(ind_interictal)));
    ind_interictal = interictal_rand(1: (length(ind_preictal) + length(ind_ictal))); 
    % o numero de pontos da interictal deve ser igual à soma dos pontos nas outras classes
    
    indexes = [ind_interictal ind_preictal ind_ictal]';
    train_P = train_P(:, indexes);
    train_T = train_T(:, indexes); 
    
    processed_T = train_T;
    processed_P = train_P;
    
end


% PRÉ-PROCESSAMENTO DAS DEEP NEURAL NETWORKS
if isequal(nn_type, "CNN")
    [processed_T, processed_P] = preprocessing_CNN(processed_T, processed_P, choice, step);

elseif isequal(nn_type, "LSTM")
    [processed_T, processed_P]= preprocessing_LSTM(processed_T, processed_P);   
end

end