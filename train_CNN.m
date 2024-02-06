% Computacao Neuronal e Sistemas Difusos 2020/21 - Trabalho 2
% Andre Bernardes (2017248159) & Joana Baiao (2017260526) - MIEB

% train_CNN: criar a rede neuronal do tipo CNN e treinar

function trained_net = train_CNN(T, P, pooling, solver, epochs, n_layers, ...
    conv_nfilters, conv_filtersize, conv_stride, ...
    pooling_size, pooling_stride)

conv_stride


% ESPECIFICAR AS OPCOES DE TREINO
options = trainingOptions(solver, ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs', epochs, ...
    'Shuffle', 'every-epoch', ... 
    'ValidationFrequency', 30, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

% DEFINIR AS LAYERS DA REDE
if n_layers == 1
    
    if isequal(pooling,'avg')
        pooling_layer = averagePooling2dLayer(pooling_size, 'Stride', pooling_stride);
    
    elseif isequal(pooling,'max')
        pooling_layer = maxPooling2dLayer(pooling_size, 'Stride', pooling_stride);
    end
    
    layers = [...
        imageInputLayer([29 29 1])
        convolution2dLayer(conv_nfilters(1), conv_filtersize(1),'Stride', conv_stride(1), 'Padding','same')
        batchNormalizationLayer
        reluLayer
        pooling_layer 
        
        fullyConnectedLayer(3) 
        softmaxLayer
        classificationLayer];
    
elseif n_layers == 2
    
    if isequal(pooling,'avg')
        pooling_layer1 = averagePooling2dLayer(pooling_size(1), 'Stride', pooling_stride(1));
        pooling_layer2 = averagePooling2dLayer(pooling_size(2), 'Stride', pooling_stride(2));
        
    elseif isequal(pooling,'max')
        pooling_layer1 = maxPooling2dLayer(pooling_size(1), 'Stride', pooling_stride(1));
        pooling_layer2 = maxPooling2dLayer(pooling_size(2), 'Stride', pooling_stride(2));
    end
    
    layers = [...
        imageInputLayer([29 29 1])  
        
        convolution2dLayer(conv_nfilters(1), conv_filtersize(1),'Stride', conv_stride(1), 'Padding','same')
        batchNormalizationLayer
        reluLayer
        pooling_layer1 
        
        convolution2dLayer(conv_nfilters(2), conv_filtersize(2),'Stride', conv_stride(2), 'Padding','same')
        batchNormalizationLayer
        reluLayer
        pooling_layer2
        
        fullyConnectedLayer(3)
        softmaxLayer
        classificationLayer];

elseif n_layers == 3
    
    if isequal(pooling,'avg')
        pooling_layer1 = averagePooling2dLayer(pooling_size(1), 'Stride', pooling_stride(1));
        pooling_layer2 = averagePooling2dLayer(pooling_size(2), 'Stride', pooling_stride(2));
        pooling_layer3 = averagePooling2dLayer(pooling_size(3), 'Stride', pooling_stride(3));
        
    elseif isequal(pooling,'max')
        pooling_layer1 = maxPooling2dLayer(pooling_size(1), 'Stride', pooling_stride(1));
        pooling_layer2 = maxPooling2dLayer(pooling_size(2), 'Stride', pooling_stride(2));
        pooling_layer3 = maxPooling2dLayer(pooling_size(3), 'Stride', pooling_stride(3));
    end
    
    layers = [...
        imageInputLayer([29 29 1])  
        convolution2dLayer(conv_nfilters(1), conv_filtersize(1),'Stride', conv_stride(1), 'Padding','same')
        batchNormalizationLayer
        reluLayer
        pooling_layer1

        convolution2dLayer(conv_nfilters(2), conv_filtersize(2),'Stride', conv_stride(2), 'Padding','same')
        batchNormalizationLayer
        reluLayer
        pooling_layer2

        convolution2dLayer(conv_nfilters(3), conv_filtersize(3),'Stride', conv_stride(3), 'Padding','same')
        batchNormalizationLayer
        reluLayer
        pooling_layer3

        fullyConnectedLayer(3)
        softmaxLayer
        classificationLayer];
end

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





