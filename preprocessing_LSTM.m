% Computacao Neuronal e Sistemas Difusos 2020/21 - Trabalho 2
% Andre Bernardes (2017248159) & Joana Baiao (2017260526) - MIEB

% preprocessing_LSTM: transformar matriz de features num cell-array e o 
%                     target num vetor categorical

function [processed_T, processed_P] = preprocessing_LSTM(T, P)

% TARGET
[~, ind_T] = max(T); % maior indice de cada coluna
processed_T = categorical(ind_T');  % conversão para um vetor categórico 

% FEATURES
processed_P = {};
for i = 1:size(P,2) % 1 até ao comprimento da coluna da matriz
     processed_P{end+1, 1} = P(:,i); % guarda as colunas 
end

end