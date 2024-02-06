% Computacao Neuronal e Sistemas Difusos 2020/21 - Trabalho 2
% Andre Bernardes (2017248159) & Joana Baiao (2017260526) - MIEB

% postprocessing_PP: comparacao do output de teste da rede com o target 
%                    esperado para cada amostra (ponto a ponto) e cálculo 
%                    da sensibilidade e especificidadeda da previsão e deteção

function [sens_predict, spec_predict, sens_detect, spec_detect] = postprocessing_PP(T, output)

% PLOT DA MATRIZ DE CONFUSÃO
figure
confusionchart(T,output, 'Title','Confusion matrix - Point by point')

TP_predict = 0; TP_detect = 0;
TN_predict = 0; TN_detect = 0;
FP_predict = 0; FP_detect = 0;
FN_predict = 0; FN_detect = 0;

for i = 1 : length(output) 
    
    % PREVISÃO
    if output(i) == 2 && T(i) == 2 % Verdadeiros positivos (TP)
        TP_predict = TP_predict + 1;
    
    elseif output(i) ~= 2 && T(i) ~= 2 % Verdadeiros negativos (TN)
        TN_predict = TN_predict + 1;
    
    elseif output(i) == 2 && T(i) ~= 2  % Falsos positivos (FP)
        FP_predict = FP_predict + 1;
    
    elseif output(i) ~= 2 && T(i) == 2  % Falsos negativos (FN)
        FN_predict = FN_predict + 1;
    end 
    
    
    % DETEÇÃO  
    if output(i) == 3 && T(i) == 3 % Verdadeiros positivos (TP)
        TP_detect = TP_detect + 1;
    
    elseif output(i) ~= 3 && T(i) ~= 3  % Verdadeiros negativos (TN)
        TN_detect = TN_detect + 1;
    
    elseif output(i) == 3 && T(i) ~= 3  % Falsos positivos (FP)
        FP_detect = FP_detect + 1;
    
     elseif output(i) ~= 3 && T(i) == 3 % Falsos negativos (FN)
        FN_detect = FN_detect + 1;
    end    
end 

% CALCULO DA SENSIBILIDADE E ESPECIFICADE
sens_predict = TP_predict/(TP_predict + FN_predict) * 100;
spec_predict = TN_predict/(TN_predict + FP_predict) * 100;

sens_detect = TP_detect/(TP_detect + FN_detect) * 100;
spec_detect = TN_detect/(TN_detect + FP_detect) * 100;

end


    

