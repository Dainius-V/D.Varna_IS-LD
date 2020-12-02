%%IS LD2 Dainius Varna EKSfm20
function LD2()
clear all;
close all;
clc;

x1_input = 0.1:1/22:1; %pirmas iejimas
y_n = (1 + 0.6*sin(2*pi*x1_input/0.7)) + 0.3*sin(2*pi*x1_input)/2; %norimas ats
n = 0.3; %mokymo zingsnis
y = backpropogation(x1_input,y_n,n); %isejimas su backpropogation

plot(x1_input,y_n,x1_input,y); grid on;
legend('Norimas ats','Backpropogation');


end
%%
%Sigmoidine aktyvavimo funkcija
function y = outS(x,w,b)
    y = 1./(1+exp(-x*w+b));
end
%%
function y = backpropogation(x,y_n,n)
    %Paruosiam parametrus:
    w_11 = [randn(1),randn(1),randn(1),randn(1),randn(1)];
    b_11 = [randn(1),randn(1),randn(1),randn(1),randn(1)];
    w_21 = [randn(1),randn(1),randn(1),randn(1),randn(1)];
    b_21 = randn(1);

    %prealocatinam del greicio
    y=zeros(1,length(x));
    e1=zeros(1,length(x));
    
    %Backpropogation
    for i=1:length(x)
        %%
        %hidden neuronu isvedimu paskaiciavimas su sigmoidine funkcija
        y1_1 = outS(x(i),w_11(1),b_11(1));
        y1_2 = outS(x(i),w_11(2),b_11(2));
        y1_3 = outS(x(i),w_11(3),b_11(3));
        y1_4 = outS(x(i),w_11(4),b_11(4));
        y1_5 = outS(x(i),w_11(5),b_11(5));
        y(i) = y1_1 * w_21(1) + y1_2 * w_21(2) + y1_3 * w_21(3) + y1_4 * w_21(4)+ y1_5 * w_21(5)+b_21;
        %%
        %apskaiciuojame  klaida:
        e1(i) = y_n(i)-y(i);
        %%
        %parametru atnaujinimas:
        w_21(1) = w_21(1) + n * e1(i) * y1_1;
        w_21(2) = w_21(2) + n * e1(i) * y1_2;
        w_21(3) = w_21(3) + n * e1(i) * y1_3;
        w_21(4) = w_21(4) + n * e1(i) * y1_4;
        w_21(5) = w_21(5) + n * e1(i) * y1_5;
        %%
        %hidden layer update:
        delta11 = (1/(1+exp(-x(i)*w_11(1)-b_11(1))))*(1/(1+exp(-x(i)*w_11(1)-b_11(1))))*e1(i)*w_21(1);
        w_11(1) = w_11(1) + n * delta11 * x(i);
        delta12 = (1/(1+exp(-x(i)*w_11(2)-b_11(2))))*(1/(1+exp(-x(i)*w_11(2)-b_11(2))))*e1(i)*w_21(2);
        w_11(2) = w_11(2) + n * delta12 * x(i);
        delta13 = (1/(1+exp(-x(i)*w_11(3)-b_11(2))))*(1/(1+exp(-x(i)*w_11(3)-b_11(3))))*e1(i)*w_21(3);
        w_11(3) = w_11(3) + n * delta13 * x(i);
        delta14 = (1/(1+exp(-x(i)*w_11(4)-b_11(4))))*(1/(1+exp(-x(i)*w_11(4)-b_11(4))))*e1(i)*w_21(4);
        w_11(4) = w_11(4) + n * delta14 * x(i);
        delta15 = (1/(1+exp(-x(i)*w_11(5)-b_11(5))))*(1/(1+exp(-x(i)*w_11(5)-b_11(5))))*e1(i)*w_21(5);
        w_11(5) = w_11(5) + n * delta15 * x(i);
        b_11(1) = b_11(1) + n * delta11;
        b_11(2) = b_11(2) + n * delta12;
        b_11(3) = b_11(3) + n * delta13;
        b_11(4) = b_11(4) + n * delta14;
        b_11(5) = b_11(5) + n * delta15;
    end
    
    
end