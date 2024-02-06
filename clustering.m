% Computacao Neuronal e Sistemas Difusos 2020/21 - Trabalho 2
% Andre Bernardes (2017248159) & Joana Baiao (2017260526) - MIEB

% clustering: realizar o clustering para conjuntos de dados com 3 features
%             atraves do método K-means ou DBSCAN. O plot 3D é feito na
%             interface

function [P, ind_interictal, ind_preictal, ind_ictal, ind_1, ind_2, ind_3] = clustering(patientID, clustering_algorithm, min_points, epsilon)

% CARREGAR FICHEIRO
try % Fazer o load do dataset com as features já reduzidas
    load(fullfile(pwd,'Datasets', strcat(patientID,'_3.mat')), 'FeatVectSel', 'Trg');

catch  % Reduzir o número de features com recurso a autoencoders e guardar
    load(fullfile(pwd,'Datasets', strcat(patientID,'.mat')), 'FeatVectSel', 'Trg');     
    
    autoencoder = trainAutoencoder(FeatVectSel', 3);
    reduced_features = encode(autoencoder, FeatVectSel');
    FeatVectSel = reduced_features';
    save(fullfile(pwd,'Datasets', strcat(patientID,'_3.mat')), 'FeatVectSel', 'Trg');       
end

P = FeatVectSel;


% DADOS REDUZIDOS COM 3 FEATURES
target = ones(length(Trg), 1);
T = [];
for i = 1:length(Trg)
    if Trg(i) == 1
        target(i) = 3; % ictal (seizure)         
        if Trg(i-1) == 0 % posição onde se inicia a seizure
            target(i-900:i-1) = 2; % preictal
        end
    end
end

for i = 1 : length(target)
    if target(i) == 1
        T(:,i) = [1 0 0]';
    elseif target(i) == 2
        T(:,i) = [0 1 0]';
    elseif target(i) == 3
        T(:,i) = [0 0 1]';
    end
end

ind_interictal = find(T(1,:) == 1);
ind_preictal = find(T(2,:) == 1);
ind_ictal = find(T(3,:) == 1);


% CLUSTERING DBSCAN
if isequal(clustering_algorithm, 'DBSCAN')
    
    try
        load(fullfile(pwd,'Clustering', strcat(patientID,'_', clustering_algorithm,'_minPts', num2str(min_points),'_eps', num2str(epsilon), '.mat')), 'ind_dbscan');
    catch
        ind_dbscan = dbscan(P, epsilon, min_points);
        save(fullfile(pwd,'Clustering', strcat(patientID,'_', clustering_algorithm,'_minPts', num2str(min_points),'_eps', num2str(epsilon), '.mat')), 'ind_dbscan');
    end
    
    ind_1 = find(ind_dbscan == 1);
    ind_2 = find(ind_dbscan == 2);
    ind_3 = find(ind_dbscan == 3);
 
% CLUSTERING K-MEANS
elseif isequal(clustering_algorithm, 'K-means')
    
    [ind_kmeans, ~] = kmeans(P, 3);
    ind_1 = find(ind_kmeans == 1);
    ind_2 = find(ind_kmeans == 2);
    ind_3 = find(ind_kmeans == 3);
    
end

end
