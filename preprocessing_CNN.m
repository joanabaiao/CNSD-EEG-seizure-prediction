% Computacao Neuronal e Sistemas Difusos 2020/21 - Trabalho 2
% Andre Bernardes (2017248159) & Joana Baiao (2017260526) - MIEB

% preprocessing_CNN: converter o dataset para imagens 2D (matriz 29x29) e
%                    construir o array 4D (29x29x1xnumero_imagens)

% O step pode ser de 29 ou de 15, consoante decidimos se fazemos o dataset 
% com ou sem sobreposição (29: sem sobreposição; 15: sobreposição de 15s)

function [processed_T, processed_P]= preprocessing_CNN(T, P, choice, step)

cell_array = {};
processed_T = [];

if isequal(choice, 'test') %para o teste
    
    %fazer grupos
    for i = 1:step:length(P)
        if i < length(P) - 28
            cell_array{end+1,1} =P(:,i:i+28);
        end
    end
    
    %target para cada um desses grupos
    for i = 1:step:length(T)
        if i < length(T) - 28
            n_interictal = nnz(find(all(T(:,i:i+28) == [1 0 0]')));
            n_preictal = nnz(find(all(T(:,i:i+28) == [0 1 0]')));
            n_ictal = nnz(find(all(T(:,i:i+28) == [0 0 1]')));
            L = [n_interictal n_preictal n_ictal];
            [~,ind] = max(L);
        
            processed_T = [processed_T ind];  
        end 
    end
    processed_T = categorical(processed_T');
    processed_P = cat(4, cell_array{:});
    
elseif isequal(choice, 'train') %para o treino
    
    % PRÉ-ICTAL
    ind_preictal = find(T(2,:) == 1);
    preictal = P(:, ind_preictal);

    n_preictal = 0;
    for i = 1:step:length(ind_preictal)
        if i < length(ind_preictal) - 28
            cell_array{end+1,1} = preictal(:,i:i+28);
            n_preictal = n_preictal + 1;
        end
    end
    for i = 1:n_preictal
        processed_T = [processed_T 2];
    end

    % ICTAL
    ind_ictal = find(T(3,:) == 1);
    ictal = P(:, ind_ictal);

    n_ictal = 0;
    for i = 1:step:length(ind_ictal)
        if i < length(ind_ictal) - 28
            cell_array{end+1,1} = ictal(:,i:i+28);
            n_ictal = n_ictal + 1;
        end
    end
    for i = 1:n_ictal
        processed_T = [processed_T 3];
    end


    % INTERICTAL
    ind_interictal = find(T(1,:) == 1);
    interictal = P(:, ind_interictal);

    n_interictal = 0;
    for i = 1:step:length(ind_interictal)
        if i < length(ind_interictal) - 28 && n_interictal <= (n_preictal + n_ictal)
            cell_array{end+1,1} = interictal(:,i:i+28);
            n_interictal = n_interictal + 1;
        end
    end
    for i = 1:n_interictal
        processed_T = [processed_T 1];
    end


    % NOVAS MATRIZES (TARGET E FEATURES) APÓS PROCESSAMENTO
    processed_T = categorical(processed_T');  % conversão para um vetor categórico 
    processed_P = cat(4, cell_array{:});
end

end