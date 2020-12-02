%LD3 IS
%1. Sukurkite spindulio tipo bazini� funkcij� (SBF) tinklo parametrams apskai�iuoti skirt� program�. SBF turi atlikti aproksimatoriaus funkcij�. SBF strukt�ra:
%- vienas ��jimas (��jime paduodamas 20 skai�i� vektorius X, su reik�m�mis intervale nuo 0 iki 1, pvz., x = 0.1:1/22:1; ).
%- vienas i��jimas (pvz., i��jime tikimasi tokio norimo atsako, kur� galima b�t� apskai�iuoti pagal formul�: y = (1 + 0.6\*sin(2\*pi\*x/0.7)) + 0.3\*sin(2\*pi\*x))/2; - kuriamas neuron� tinklas tur�t� "modeliuoti/imituoti �ios formul�s elges�" naudodamas visi�kai kitoki� matematin� i�rai�k� nei �i);
%- dvi spindulio tipo funkcijos (naudokite Gauso funkcijas: F = exp(-(x-c)^2/(2*r^2)));
%- centr� reik�mes c1 ir c2, spinduli� reik�mes r1 ir r2 parinkite rankiniu b�du;
%- tiesine aktyvavimo funkcija i��jimo neurone;
%- mokymo algoritmas skirtas i��jimo parametrams w1, w2 ir w0(b) apskai�iuoti - toks pat kaip ir perceptrono mokyme (1 LD).
% Papildoma u�duotis (papildomi 2 balai)
%Apmokykite SBF tinkl� kitu algoritmu, kuris taip pat atnaujina/parenka automati�kai ir centr� c1, c2 bei spinduli� r1, r2 reik�mes.
function IS_LD3()
    clear all;
    clc;
    close all;
    
    x = linspace(0,1,20);
    y_n = (1 + 0.6 * sin(2 * pi * x /0.7)) + (0.3 *sin(2 *pi *x)/2);%imituoti
    c1 = 0.15; c2 = 0.89; r1 = 0.25; r2 = 0.27; iteracijos = 100;
    [w1, w2, b] = mokymoAlgoritmas(x, y_n, c1, c2, r1, r2, iteracijos);
    y_g = tinklo_pabandymas(w1,w2,b,x,c1,c2,r1,r2);
    plot(x,y_g,x,y_n);grid on;
    legend('Aproksimuotas','Realus');
    ylabel('y');xlabel('x');

end
%%
%Tinklo pabandymas su apmokytais duomenims
function y = tinklo_pabandymas(w1,w2,b,x,c1,c2,r1,r2)
   for i=1:length(x)
       g1(i) = F(x(i),c1,r1); 
       g2(i) = F(x(i),c2,r2);
       y(i) = outY(g1(i),w1,g2(i),w2,b);
   end
end
%%
function [w1, w2, b] = mokymoAlgoritmas(x, y_n, c1, c2, r1, r2,iteracijos)
    w1 = rand(1);
    w2 = rand(1);
    b = rand(1);
    eta = 0.2; %mokymo zingsnis
    for k = 1:iteracijos
        for i=1:length(y_n)
            g1 = F(x(i),c1,r1); g2 = F(x(i),c2,r2);
            v = outY(g1,w1,g2,w2,b);
            error = y_n(i) - v;
            w1 = w1 + eta*error*g1;
            w2 = w2 + eta*error*g2;
            b = b + eta*error;
        end
    end
end
%%
%Gauso funkcija dviem spindulio tipo funkcijoms:
function y = F(x,c,r)
    y = exp(-1*((x-c).^2/(2*r.^2)));
end
%%
%Isejimo funkcija
function y = outY(g1,w1,g2,w2,b)
    y=g1*w1+g2*w2+b;
end
%%