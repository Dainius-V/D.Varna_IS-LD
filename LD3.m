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
%%1234567890
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
    %[x1, x2, w1, w2, b] = mokymoAlgoritmas(x,y_n);
    mokymoAlgoritmas(x,y_n);
    %g1 = F(x1,c1,r1); g2 = F(x2,c2,r2);
    %y_g = outY(g1,w1,g2,w2,b);
%     fprintf('y_g: ');fprintf('%f ',y_g);
%     fprintf('\n');
%     fprintf('y_n: ');fprintf('%f ',y_n);
%     fprintf('\n ');
    
end
%%
%Gauso funkcija dviem spindulio tipo funkcijoms:
function y = F(x,c,r)
    y = exp(-1*(x-c).^2/(2*r^2));
end
%%
%Tiesine aktyvavimo funkcija:
%function y = linear(x)
%    y = x;
%end
%%
%Isejimo funkcija
function y = outY(g1,w1,g2,w2,b)
    y=g1*w1+g2*w2+b;
end
%%
function [x1, x2, w1, w2, b] = mokymoAlgoritmas(x1, y_n)
%Paruosiam kintamuosius:
    w1 = randn();
    w2 = randn();
    b = randn();
    eta = 0.1; %mokymo zingsnis
    x2 = (-1*(w1/w2))*x1-(b/w2);
    c1 = .2; c2 = 0.5; r1 = .7; r2 = 0.2;
%Paruosiam matricas:
    n = length(y_n);
    g1 = zeros(1,n);
    g2 = zeros(1,n);
    v = zeros(1,n);
    y = zeros(1,n);
    y_g = zeros(1,n);
    e = zeros(1,n);
      
%%
 
        %Atliekame skaiciavimus
        %% Apskaiciuojame vertes su sugeneruotais iverciais w1,w2,b
        for j=1:n 
            v(j) = x1(j) * w1 + x2(j) * w2 + b;
            g1(j) = F(x1(n),c1,r1); g2(j) = F(x2(j),c2,r2);
            y_g(j) = outY(g1(j),w1,g2(j),w2,b);
            e(j) = y_n(j) -  y_g(j); %Patikriname klaida
        end
        fprintf('\nsum(e)=%f\n ',sum(e));
        %%
        e_r=0;
        errorN = 0.05; %norima paklaida
        while e_r ~= 1
            %lets go
            fprintf('\ny_n=');
            fprintf('%f \n',y_n);
            fprintf('\ny_g=');
            fprintf('%f \n',y_g);
            for n=1:length(x1)
                if (n==length(x1))
                %Jei masyvo pozicija paskutine
                    if (abs(y_n(n)-y_g(n)) > errorN) %Ar ats norimas
                        v(n) = x1(n) * w1 + x2(n) * w2 + b;
                        g1(n) = F(x1(n),c1,r1); g2(n) = F(x2(n),c2,r2);
                        y_g(n) = outY(g1(n),w1,g2(n),w2,b);
                        e(n)=y_n(n)-y_g(n);
                    end
                else
                    %Jei masyvo pozicija ne max
                    if(abs(y_n(n)-y_g(n)) <= errorN)
                    %tikriname ar turim norima ats
                        if(abs(y_n(n+1)-y_g(n+1)) <= errorN)
                            %tikriname ar turim sekanti norima ats
                        else
                            %Jei sekancio masyvo reiksme neturi norimo ats
                            %atnaujinima parametrus
                            w1 = w1 + eta * e(n) * x1(n);
                            w2 = w2 + eta * e(n) * x2(n);
                            b = b + eta * e(n);
                            %Patikriname ar gucci po atnaujinimo
                            for h=1:n
                                if (check_update(w1,w2,b,x1(h),x2(h),y_n(h),c1,c2,r1,r2) > errorN)
                                    break;
                                end
                            end
                        end
                    else %Jei nera norimo ats, atliekame skaiciavimus
                        v(n) = x1(n) * w1 + x2(n) * w2 + b;
                        g1(n) = F(x1(n),c1,r1); g2(n) = F(x2(n),c2,r2);
                        y_g(n) = outY(g1(n),w1,g2(n),w2,b);
                        e(n)=y_n(n)-y_g(n);
                        if(abs(y_n(n+1)-y_g(n+1)) > errorN)
                            %jei blogas sekantis ats atnaujiname parametrus
                            w1 = w1 + eta * e(n) * x1(n);
                            w2 = w2 + eta * e(n) * x2(n);
                            b = b + eta * e(n);
                            %Patikriname ar gucci po atnaujinimo
                            for h=1:n
                                if (check_update(w1,w2,b,x1(h),x2(h),y_n(h),c1,c2,r1,r2) > errorN)
                                    break;
                                end
                            end
                        end  
                    end
                end
            end
            e_t = (abs(sum(e)))/length(x1);
            if(e_t <= 0.05)
                e_r=1;
            end
        end
        %%
%         %%
%         %Jeigu klaidu suma didesne nei 1 - naujiname kintamuosius
%         if (abs(sum(e)) > 1)
%             ce_r=0;
%             while ce_r ~= 1
%                 fprintf('naujinam\n');
%                 for l=1:length(x1)
%                     w1 = w1 + eta * e(l) * x1(l);
%                     w2 = w2 + eta * e(l) * x2(l);
%                     b = b + eta * e(l);
%                 end
%                 ce_t = check_update(w1, w2, b, x1, x2, y_n, c1, c2, r1, r2); %Patikrinimame
%                 if ce_t <= 1 %Jeigu klaidu suma mazesne nei 1 breakinam
%                     for i=1:length(x1)
%                         v(i) = x1(i) * w1 + x2(i) * w2 + b;
%                         g1(i) = F(x1(n),c1,r1); g2(i) = F(x2(i),c2,r2);
%                         y_g(i) = outY(g1(i),w1,g2(i),w2,b);
%                     end
%                     ce_r = 1;
%                     break;
%                 end
%             end
%         else
%             for i=1:length(x1)
%                 v(i) = x1(i) * w1 + x2(i) * w2 + b;
%                 g1(i) = F(x1(n),c1,r1); g2(i) = F(x2(i),c2,r2);
%                 y_g(i) = outY(g1(i),w1,g2(i),w2,b);
%             end
%             ce_r = 1;
%         end
plot(x1,y_g,x1,y_n);
fprintf('y_g: ');fprintf('%f ',round(y_g,2));
fprintf('\nsum=%f\n ',sum(y_g));
fprintf('y_n: ');fprintf('%f ',round(y_n,2));
fprintf('\nsum=%f\n ',sum(y_n));
end
%%
function ce = check_update(w1,w2,b,x1,x2,y_n,c1,c2,r1,r2)
    v = x1 * w1 + x2 * w2 + b;
    g1 = F(x1,c1,r1); g2 = F(x2,c2,r2);
    y_g = outY(g1,w1,g2,w2,b);
    ce = abs(y_n-y_g);
%     fprintf('\n ce=%f\n ',ce);
end
% function ct = check_update(w1,w2,b,x1,x2,y_n,c1,c2,r1,r2)
%     cy=x1*w1+x2*w2+b;
%     ce = zeros(1,length(x1));
%     g1 = zeros(1,length(x1));
%     g2 = zeros(1,length(x1));
%     y_g = zeros(1,length(x1));
%     for i=1:length(x1)
%         v(i) = x1(i) * w1 + x2(i) * w2 + b;
%         g1(i) = F(x1(i),c1,r1); g2(i) = F(x2(i),c2,r2);
%         y_g(i) = outY(g1(i),w1,g2(i),w2,b);
%         ce(i) = y_n(i)-y_g(i);
%     end        
%     ct = abs(sum(y_n)-sum(ce));
%     fprintf('\n ct=%f\n ',ct);
% end
%%   