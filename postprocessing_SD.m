% Computacao Neuronal e Sistemas Difusos 2020/21 - Trabalho 2
% Andre Bernardes (2017248159) & Joana Baiao (2017260526) - MIEB

% postprocessing_SD: determinar o número de seizures detetadas e previtas

function [seizures_detected,seizures_predicted, n_seizures]=postprocessing_SD(T, output)

% DETETAR SEIZURES
seizures = [];
crise = false; % Fica verdadeiro caso estejamos a percorrer os pontos pertencentes a uma crise
for i = 2:length(T)
    if crise == false
        if T(i) == 3
            ictal_start = i;
            crise = true;
        end
    else
        if  T(i+1) ~= 3
            ictal_end = i;
            crise = false;
            % guardar as posicoes das seizures (linha 1: inicio; linha 2: final)
            seizures = [seizures [ictal_start; ictal_end]];
        end
    end        
end
n_seizures=length(seizures); %nr de seizures real

% SEIZURES DETETADAS
n_ictal=0;
seizures_detected=0;
for i = 1:length(seizures)

    for j=seizures(1,i):seizures(2,i)
        if output(j)==3
            n_ictal=n_ictal+1;
        end
    end

    n=seizures(2,i)-seizures(1,i);
    
    if n_ictal>=(n*0.80)
        seizures_detected=seizures_detected+1;
    end

    n_ictal=0;
end

% SEIZURES PREVISTAS
n_preictal=0;
seizures_predicted=0;
for i = 1:length(seizures)
    
    for j=seizures(1,i)-900:seizures(1,i)
         if output(j)==2
             n_preictal=n_preictal+1;
         end
    end

    if n_preictal>=(900*0.3)
        seizures_predicted=seizures_predicted+1;
    end

    n_preictal=0;
end

end
 