%IS LD5 Dainius Varna EKSfm20
%%
function LD5()
    clc;
    clear all;
    close all;
    %%
    %%
    pavadinimas = 'train_data.png'; %Mokymo dataset
    a = 8;  % eiluciu skaicius train_data.png
    b = 10; % simboliu skaicius
    %%
    %%Tinklas 1 Radial basis network
    %%
    pozymiai_tinklo_mokymui = pozymiai_skaiciams_atpazinti(pavadinimas, a);
    [P,tinklas1,tinklas2] = tinklu_mokymas(pozymiai_tinklo_mokymui,b);
    tinklo_patikra(P,tinklas1); %Mokymo rezultatu atvaizdavimas
    %%
    t1_pavadinimas = 'test_data1.png';  %Tinklo tikrinimui dataset1
    pozymiai_patikrai1 = pozymiai_skaiciams_atpazinti(t1_pavadinimas, 1);
    simboliu_atpazinimas(pozymiai_patikrai1, tinklas1,8); 
    %%
    t2_pavadinimas = 'test_data_2.png'; %Tinklo tikrinimui dataset2
    pozymiai_patikrai2 = pozymiai_skaiciams_atpazinti(t2_pavadinimas, 1);
    simboliu_atpazinimas(pozymiai_patikrai2, tinklas1,9);
    %
    %%
    fprintf('Pirmas tinklas atvaizduotas \n');
    pause;
    fprintf('Antra tinkla atvaizduojam \n');
    close all
    pause(2);
    %%Tinklas 2 feed-forward backpropagation network
    %%
    tinklo_patikra(P,tinklas2); %Mokymo rezultatu atvaizdavimas
    %%
    simboliu_atpazinimas(pozymiai_patikrai1, tinklas2,8); 
    %%
    simboliu_atpazinimas(pozymiai_patikrai2, tinklas2,9);
    %
    %%
end
%%
function [P,tinklas1,tinklas2] = tinklu_mokymas(pozymiai_tinklo_mokymui,b)
    % take the features from cell-type variable and save into a matrix-type variable
    P = cell2mat(pozymiai_tinklo_mokymui);
    % create the matrices of correct answers for each line (number of matrices = number of symbol lines)
    T = [eye(b), eye(b), eye(b), eye(b), eye(b), eye(b), eye(b), eye(b)];
    % create an RBF network for classification with 13 neurons, and sigma = 1
    tinklas1 = newrb(P,T,0,1,13); %Nuo 9 neuronu pradeda atpazinti test_data_2
    %tinklas2 = newff(P,T,100); %feed-forward backpropagation network
    net = newff(P,T,[]);
    tinklas2 = train(net,P,T);
    %net = newff(P,T,S,TF,BTF,BLF,PF,IPF,OPF,DDF)
end
%%
function simboliu_atpazinimas(pozymiai_patikrai, tinklas,c)
    %% Perform letter/symbol recognition
    % features from cell-variable are stored to matrix-variable
    P2 = cell2mat(pozymiai_patikrai);
    % estimating neuran network output for newly estimated features
    Y2 = sim(tinklas, P2);
    % searching which output gives maximum value
    [a2, b2] = max(Y2);
    %% Rezultato atvaizdavimas | Visualization of result
    % calculating number of symbols - number of columns
    skaiciu_sk = size(P2,2);
    % rezultatà saugosime kintamajame 'atsakymas'
    atsakymas = [];
    for k = 1:skaiciu_sk
        switch b2(k)
            case 1
                atsakymas = [atsakymas, '1'];
            case 2
                atsakymas = [atsakymas, '2'];
            case 3
                atsakymas = [atsakymas, '3'];
            case 4
                atsakymas = [atsakymas, '4'];
            case 5
                atsakymas = [atsakymas, '5'];
            case 6
                atsakymas = [atsakymas, '6'];
            case 7
                atsakymas = [atsakymas, '7'];
            case 8
                atsakymas = [atsakymas, '8'];
            case 9
                atsakymas = [atsakymas, '9'];
            case 10
                atsakymas = [atsakymas, '0'];
        end
    end
    % pateikime rezultatà komandiniame lange
    disp(atsakymas)
    figure(c), text(0.1,0.5,atsakymas,'FontSize',35), axis off
end
%%
function tinklo_patikra(P,tinklas)
    P2 = P(:,12:22);
    Y2 = sim(tinklas, P2);
    % find which neural network output gives maximum value
    [a2, b2] = max(Y2);
    %% Rezultato atvaizdavimas
    %% Visualize result
    % calculate the total number of symbols in the row
    skaiciu_sk = size(P2,2);
    % we will save the result in variable 'atsakymas'
    atsakymas = [];
    for k = 1:skaiciu_sk
        switch b2(k)
            case 1
                atsakymas = [atsakymas, '1'];
            case 2
                atsakymas = [atsakymas, '2'];
            case 3
                atsakymas = [atsakymas, '3'];
            case 4
                atsakymas = [atsakymas, '4'];
            case 5
                atsakymas = [atsakymas, '5'];
            case 6
                atsakymas = [atsakymas, '6'];
            case 7
                atsakymas = [atsakymas, '7'];
            case 8
                atsakymas = [atsakymas, '8'];
            case 9
                atsakymas = [atsakymas, '9'];
            case 10
                atsakymas = [atsakymas, '0'];
        end
    end
    % show the result in command window
    disp(atsakymas)
end
%%
function pozymiai = pozymiai_skaiciams_atpazinti(pavadinimas, pvz_eiluciu_sk)
    % Vaizdo su pavyzdþiais nuskaitymas | Read image with written symbols
    V = imread(pavadinimas);
    figure(12), imshow(V)
    %% Perform segmentation of the symbols and write into cell variable 
    % RGB image is converted to grayscale
    V_pustonis = rgb2gray(V);
    % a threshold value is calculated for binary image conversion
    slenkstis = graythresh(V_pustonis);
    % pustonio vaizdo keitimas dvejetainiu
    % a grayscale image is converte to binary image
    V_dvejetainis = im2bw(V_pustonis,slenkstis);
    % rezultato atvaizdavimas
    % show the resulting image
    figure(1), imshow(V_dvejetainis)
    % search for the contour of each object
    V_konturais = edge(uint8(V_dvejetainis));
    % rezultato atvaizdavimas
    % show the resulting image
    figure(2),imshow(V_konturais)
    % fill the contours
    se = strel('square',7); % strukturinis elementas uzpildymui
    V_uzpildyti = imdilate(V_konturais, se); 
    % rezultato atvaizdavimas
    % show the result
    figure(3),imshow(V_uzpildyti)
    % fill the holes
    V_vientisi= imfill(V_uzpildyti,'holes');
    % rezultato atvaizdavimas
    % show the result
    figure(4),imshow(V_vientisi)
    % set labels to binary image objects
    [O_suzymeti Skaicius] = bwlabel(V_vientisi);
    % calculate features for each symbol
    O_pozymiai = regionprops(O_suzymeti);
    % find/read the bounding box of the symbol
    O_ribos = [O_pozymiai.BoundingBox];
    % change the sequence of values, describing the bounding box
    O_ribos = reshape(O_ribos,[4 Skaicius]); % Skaicius - objektu skaicius
    % reag the mass center coordinate
    O_centras = [O_pozymiai.Centroid];
    % group center coordinate values
    O_centras = reshape(O_centras,[2 Skaicius]);
    O_centras = O_centras';
    % set the label/number for each object in the image
    O_centras(:,3) = 1:Skaicius;
    % arrange objects according to the column number
    O_centras = sortrows(O_centras,2);
    % sort accordign to the number of rows and number of symbols in the row
    raidziu_sk = Skaicius/pvz_eiluciu_sk;
    for k = 1:pvz_eiluciu_sk
        O_centras((k-1)*raidziu_sk+1:k*raidziu_sk,:) = ...
            sortrows(O_centras((k-1)*raidziu_sk+1:k*raidziu_sk,:),3);
    end
    % cut the symbol from initial image according to the bounding box estimated in binary image
    for k = 1:Skaicius
        objektai{k} = imcrop(V_dvejetainis,O_ribos(:,O_centras(k,3)));
    end
    % vieno ið vaizdo fragmentø atvaizdavimas
    % show one of the symbol's image
    figure(5),
    for k = 1:Skaicius
       subplot(pvz_eiluciu_sk,raidziu_sk,k), imshow(objektai{k})
    end
    % image segments are cutt off
    for k = 1:Skaicius % Skaicius = 88, jei yra 88 raidës
        V_fragmentas = objektai{k};
        % nustatomas kiekvieno vaizdo fragmento dydis
        % estimate the size of each segment
        [aukstis, plotis] = size(V_fragmentas);
        % eliminate white spaces
        % apskaiciuojame kiekvieno stulpelio sk
        stulpeliu_sumos = sum(V_fragmentas,1);
        % naikiname tuos stulpelius, kur suma lygi auksciau
        V_fragmentas(:,stulpeliu_sumos == aukstis) = [];
        % perskaiciuojamas objekto dydis
        [aukstis, plotis] = size(V_fragmentas);
        % 2. Baltø eiluèiø naikinimas
        % apskaiciuojame kiekvienos eilutes suma
        eiluciu_sumos = sum(V_fragmentas,2);
        % naikiname tas eilutes, kur suma lygi plociui
        V_fragmentas(eiluciu_sumos == plotis,:) = [];
        objektai{k}=V_fragmentas;% �ia rasome vietoje neapkarpyto
    end
    % show the segment
    figure(6),
    for k = 1:Skaicius
       subplot(pvz_eiluciu_sk,raidziu_sk,k), imshow(objektai{k})
    end
    %%
    %% Make all segments of the same size 50x70
    for k=1:Skaicius
        V_fragmentas=objektai{k};
        V_fragmentas_7050=imresize(V_fragmentas,[70,50]);
        % padalinkime vaizdo fragmentà �? 10x10 dydþio dalis
        % divide each image into 10x10 size segments
        for m=1:7
            for n=1:5
                % apskaièiuokime kiekvienos dalies vidutin�? ðviesumà 
                % calculate an average intensity for each 10x10 segment
                Vid_sviesumas_eilutese=sum(V_fragmentas_7050((m*10-9:m*10),(n*10-9:n*10)));
                Vid_sviesumas((m-1)*5+n)=sum(Vid_sviesumas_eilutese);
            end
        end
        % 10x10 dydzio dalyje maksimali sviesumo galima reikme yra 100
        % normuokime sviesumo reiksmes intervale [0, 1]
        % perform normalization
        Vid_sviesumas = ((100-Vid_sviesumas)/100);
        % transform features into column-vector
        Vid_sviesumas = Vid_sviesumas(:);
        % save all fratures into single variable
        pozymiai{k} = Vid_sviesumas;
    end
end