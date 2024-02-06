% Computacao Neuronal e Sistemas Difusos 2020/21 - Trabalho 2
% Andre Bernardes (2017248159) & Joana Baiao (2017260526) - MIEB

% postprocessing_FT: processamento em bloco com pelo menos 5 amostras em 10
%                    (five in ten - FT), isto é, numa janela de 10 amostras,
%                    se pelo menos 5 forem classificadas na mesma classe,
%                    essa classe é atribuida a toda a amostra.

function [sens_predict, spec_predict, sens_detect, spec_detect] = postprocessing_FT(T, output)


% PROCESSAMENTO EM JANELAS DE 10
processed_output = output;
for i = 1:(length(output)-10)

    if i < (length(output)-10) 
            
        output_interictal = find(output(i+1:i+10) == 1); 
        output_preictal = find(output(i+1:i+10) == 2);
        output_ictal = find(output(i+1:i+10) == 3);

        [m, ind] = max([length(output_interictal), length(output_preictal), length(output_ictal)]);

        if m >= 5
            processed_output(i) = ind;
        end
    end
end


% PLOT DA MATRIZ DE CONFUSÃO
figure
confusionchart(T,processed_output, 'Title','Confusion matrix - Five in ten');

TP_predict = 0; TP_detect = 0;
TN_predict = 0; TN_detect = 0;
FP_predict = 0; FP_detect = 0;
FN_predict = 0; FN_detect = 0;

for i = 1 : length(processed_output) 
    
    % PREVISÃO
    if processed_output(i) == 2 && T(i) == 2 % Verdadeiros positivos (TP)
        TP_predict = TP_predict + 1;
    
    elseif processed_output(i) ~= 2 && T(i) ~= 2 % Verdadeiros negativos (TN)
        TN_predict = TN_predict + 1;
    
    elseif processed_output(i) == 2 && T(i) ~= 2  % Falsos positivos (FP)
        FP_predict = FP_predict + 1;
    
    elseif processed_output(i) ~= 2 && T(i) == 2  % Falsos negativos (FN)
        FN_predict = FN_predict + 1;
    end 
    
    
    % DETEÇÃO  
    if processed_output(i) == 3 && T(i) == 3 % Verdadeiros positivos (TP)
        TP_detect = TP_detect + 1;
    
    elseif processed_output(i) ~= 3 && T(i) ~= 3  % Verdadeiros negativos (TN)
        TN_detect = TN_detect + 1;
    
    elseif processed_output(i) == 3 && T(i) ~= 3  % Falsos positivos (FP)
        FP_detect = FP_detect + 1;
    
     elseif processed_output(i) ~= 3 && T(i) == 3 % Falsos negativos (FN)
        FN_detect = FN_detect + 1;
    end    
end 

% CALCULO DA SENSIBILIDADE E ESPECIFICADE
sens_predict = TP_predict/(TP_predict + FN_predict) * 100;
spec_predict = TN_predict/(TN_predict + FP_predict) * 100;

sens_detect = TP_detect/(TP_detect + FN_detect) * 100;
spec_detect = TN_detect/(TN_detect + FP_detect) * 100;

end
