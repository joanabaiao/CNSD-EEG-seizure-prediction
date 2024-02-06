% Computacao Neuronal e Sistemas Difusos 2020/21 - Trabalho 2
% Andre Bernardes (2017248159) & Joana Baiao (2017260526) - MIEB

% main_train: funcao principal que é chamada na interface quando é
%             escolhida a opcao de treino

function main_train(patientID, nn_type, n_features, optimization, delay,...
                training_style,training_option, activation_function, ...
                solver, epochs, layers, pooling, neurons, conv_nfilters, conv_filtersize, conv_stride, ...
                pooling_size, pooling_stride, step)

fprintf('TRAIN:\n* Pacient ID = %s \n* NN type = %s \n* Features = %d \n', patientID, nn_type, n_features)
            
% CARREGAR FICHEIRO
if n_features == 29
    load(fullfile(pwd,'Datasets', strcat(patientID,'.mat')), 'FeatVectSel', 'Trg');
else
    try % Fazer o load do dataset com as features já reduzidas
        load(fullfile(pwd,'Datasets', strcat(patientID,'_',num2str(n_features),'.mat')), 'FeatVectSel', 'Trg');
    
    catch % Reduzir o número de features com recurso a autoencoders e guardar
        load(fullfile(pwd,'Datasets', strcat(patientID,'.mat')), 'FeatVectSel', 'Trg');     

        autoencoder = trainAutoencoder(FeatVectSel', n_features);
        reduced_features = encode(autoencoder, FeatVectSel');
        FeatVectSel = reduced_features';
        save(fullfile(pwd,'Datasets', strcat(patientID,'_',num2str(n_features),'.mat')), 'FeatVectSel', 'Trg');       
    end
end
      

% PRÉ-PROCESSAMENTO DE TREINO
[T, P] = preprocessing(Trg, FeatVectSel, 'train', nn_type, step);


% TREINO
if isequal(nn_type, "LR")
    trained_net = train_SNN(T, P, nn_type, layers, neurons, training_style, training_option, activation_function, delay, optimization);
    filename = strcat('SNN_LR_', patientID, '_', activation_function, '_', training_option, '_LD', num2str(delay), '_F', num2str(n_features), '_N', mat2str(neurons), '.mat');
    nn_type = "SNN";

elseif isequal(nn_type, "FF")
    trained_net = train_SNN(T, P, nn_type, layers, neurons, training_style, training_option, activation_function, delay, optimization);
    filename = strcat('SNN_FF_', patientID, '_', activation_function, '_', training_option, '_F', num2str(n_features),'_N', mat2str(neurons), '.mat');
    nn_type = "SNN";

elseif isequal(nn_type, "LSTM")
    trained_net = train_LSTM(T, P, neurons, n_features, epochs, solver);
    filename = strcat('DNN_LSTM_',patientID, '_e', num2str(epochs), '_', solver,'_F', num2str(n_features),'_N', mat2str(neurons), '.mat');  
    
elseif isequal(nn_type, "CNN")
    trained_net = train_CNN(T, P, pooling, solver, epochs, layers, conv_nfilters, conv_filtersize, conv_stride, pooling_size, pooling_stride);
    filename = strcat('DNN_CNN_',patientID, '_e', num2str(epochs), '_', solver, '_Conv_NF', mat2str(conv_nfilters), '_FSz', mat2str(conv_filtersize),...
        '_S', mat2str(conv_stride), '_Pool_Sz', mat2str(pooling_size), '_S', mat2str(pooling_stride), '_step', num2str(step), '.mat'); 
end


% GUARDAR REDES TREINADAS
if isequal(nn_type, "CNN")
    save(fullfile(pwd,'Trained Networks', nn_type, patientID, filename), 'trained_net', 'patientID', 'nn_type', 'n_features', 'step');
else
    save(fullfile(pwd,'Trained Networks', nn_type, patientID, filename), 'trained_net', 'patientID', 'nn_type', 'n_features');
end

end