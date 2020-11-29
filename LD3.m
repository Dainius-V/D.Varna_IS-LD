%LD3 IS
%1. Sukurkite spindulio tipo baziniø funkcijø (SBF) tinklo parametrams apskaièiuoti skirtà programà. SBF turi atlikti aproksimatoriaus funkcijà. SBF struktûra:
%- vienas áëjimas (áëjime paduodamas 20 skaièiø vektorius X, su reikðmëmis intervale nuo 0 iki 1, pvz., x = 0.1:1/22:1; ).
%- vienas iðëjimas (pvz., iðëjime tikimasi tokio norimo atsako, kurá galima bûtø apskaièiuoti pagal formulæ: y = (1 + 0.6\*sin(2\*pi\*x/0.7)) + 0.3\*sin(2\*pi\*x))/2; - kuriamas neuronø tinklas turëtø "modeliuoti/imituoti ðios formulës elgesá" naudodamas visiðkai kitokià matematinæ iðraiðkà nei ði);
%- dvi spindulio tipo funkcijos (naudokite Gauso funkcijas: F = exp(-(x-c)^2/(2*r^2)));
%- centrø reikðmes c1 ir c2, spinduliø reikðmes r1 ir r2 parinkite rankiniu bûdu;
%- tiesine aktyvavimo funkcija iðëjimo neurone;
%- mokymo algoritmas skirtas iðëjimo parametrams w1, w2 ir w0(b) apskaièiuoti - toks pat kaip ir perceptrono mokyme (1 LD).
% Papildoma uþduotis (papildomi 2 balai)
%Apmokykite SBF tinklà kitu algoritmu, kuris taip pat atnaujina/parenka automatiðkai ir centrø c1, c2 bei spinduliø r1, r2 reikðmes.

function LD3()
    clear;
    clc;
    
    x = linspace(0,1,20);
    y_n = (1 + 0.6 * sin(2 * pi * x /0.7)) + (0.3 *sin(2 *pi *x)/2);%imituoti
    %Dvi spindulio tipo funkcijos:
    %F = exp(-(x-c)^2/(2*r^2)); %Gauso funkcijas
    %centru reiksmes c1 ir c2, spindulio r1 ir r2 (rankiniu budu)
    c1 = 1; c2 = 0.5; r1 = 0.2; r2 = 0.5;
    %tiesine aktyvavimo funkcija isejimo neurone:
    [x1, x2, w1, w2, b] = mokymoAlgoritmas(x,y_n);
    g1 = F(x1,c1,r1); g2 = F(x2,c2,r2);
    y_g = outY(g1,w1,g2,w2,b);
    fprintf('%f',y_g);
    
end
%Gauso funkcija dviem spindulio tipo funkcijoms:
function y = F(x,c,r)
    y = exp(-1*(x-c).^2/(2*r^2));
end

%Tiesine aktyvavimo funkcija:
%function y = linear(x)
%    y = x;
%end

%Isejimo funkcija
function y = outY(g1,w1,g2,w2,b)
    y=g1*w1+g2*w2+b;
end

function y = atsTikrinimas(a)
    y = 0;
    if (a > y)
        y = 1;
    elseif(a <= 0)
        y = -1;
    end
end
function [x1, x2, w1, w2, b] = mokymoAlgoritmas(x1, y_n)
%Paruosiam kintamuosius:
    w1 = randn();
    w2 = randn();
    b = randn();
    n = length(y_n);
    
    v = zeros(1,n);
    y = zeros(1,n);
    e = zeros(1,n);
    etotal = 1;

%kartojam cikla kol etotal nebus 0
%while etotal ~= 0
    eta = 0.1; %mokymo zingsnis
    x2 = (-1*(w1/w2))*x1-(b/w2);

    for i=1:n    
        %Atliekame skaiciavimus
        v(n) = x1(n) * w1 + x2(n) * w2 + b;
        y(n) = atsTikrinimas(v(n));
        e(n) = y_n(n) -  y(n);

        %Parametru naujinimas
        w1 = w1 + eta * e(n) * x1(n);
        w2 = w2 + eta * e(n) * x2(n);
        b = b + eta * e(n);
        ce_t = check_update(w1, w2, b, x1, x2, y_n);
       
    end   
    ce_t
    %etotal = sum(e);
%end
end

function ce = check_update(w1,w2,b,x1,x2,y_n)
    cy=x1*w1+x2*w2+b;
    ce = zeros(1,length(x1));
    for i=1:length(x1)
        ce(1) = atsTikrinimas(cy);
    end        
    ce = y_n-ce;
end
    