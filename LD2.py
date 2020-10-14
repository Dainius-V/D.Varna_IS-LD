# IS laboratorinis darbas nr. 2. Dainius Varna EKSfm20 202000333

import math as m
import numpy as np
import random


# Sugeneruojame neurono isejime tikimasi atsakyma
def n_input(k):
    a = [0] * k
    n = 0
    m = 1 / k

    while n != len(a):
        if a[n] == 1:
            break
        else:
            a[n] = a[n-1] + m
            n += 1
    return a


# Sugeneruojame neurono isejima
def n_output(a):
    return (1 + 0.6 * m.sin(2 * m.pi * a / 0.7)) + 0.3 * (m.sin(2 * m.pi * a)) / 2


# Sigmoidinė  funkcija
def sigmoid(h):
    return 1 / 1 + np.exp(-h)


# Norimi atsakymai
def n_ats(xx):
    y_norimi = [0] * len(xx)
    a = 0
    while a != len(xx):
        y_norimi[a] = n_output(xx[a])
        a += 1

    return y_norimi


# Sigmoidine aktyvavimo funkcija
def n_aktyvimas(xx):
    y_1 = [0] * len(xx)
    a = 0

    while a != len(xx):
        y_1[a] = sigmoid(xx[a])
        a += 1

    return y_1


# Funkcija apskaiciuoti klaidas - 1/2 * (e^2)
def klaidu_tikrinimas(d, y):
    e = d - y
    return e
# m.sqrt(e[0] ** e[0])


# Funkcija w parametrams sugeneruoti pagal nurodyta kieki
def w_parametru_generavimas(n_h):
    w1 = [0] * n_h
    n = 0
    while n != len(w1):
        w1[n] = random.random()
        n += 1
    return w1


# Funkcija b parametrams sugeneruoti pagal nurodyta kieki
def b_parametru_generavimas(n_h):
    b1 = [0] * n_h
    n = 0
    while n != len(b1):
        b1[n] = random.random()
        n += 1
    return b1


# Funkcija y isejimui paskaiciuoti pagal gautus ivercius
def y1_funkcija(w_1, b_1, x_1):
    y1 = [0] * len(w_1)
    a = 0

    while a != len(w_1):
        y1[a] = x_1 * w_1[a] + b_1[a]
        a += 1

    return y1


def y_funkcija(y1, w2, b2):
    n = 0
    a = 0
    while n != len(y1):
        a = 0 + y1[n] * w2[n]
        n += 1

    return a + b2


# Funkcija tikslumui paskaiciuoti
def tikslumas(n1, n2):
    return n1 - n2


# Funkcija palyginti sugeneruota y isejima pagal formule ir algoritma
def x_mtip_compare(wt1, bt1, wt2, bt2):
    xt_n = 20  # neurono iejimu kiekis
    xtc = n_input(xt_n)
    y_1 = [0] * len(xtc)
    v_1 = [0] * len(xtc)
    y = [0] * len(xtc)  # masyvas algoritmui
    yr = [0] * len(xtc)  # masyvas atsakymam su formule (output)
    n = 0
    while n != len(xtc):
        yr[n] = n_output(xtc[n])
        n += 1
    n = 0
    while n != len(xtc):
        y_1[n] = y1_funkcija(wt1, bt1, xtc[n])
        v_1[n] = n_aktyvimas(y_1[n])
        y[n] = y_funkcija(v_1[n], wt2, bt2)
        n += 1
    print("Palyginti ats:")
    print("yr(output) =", yr)
    print("y(algoritmo) =", y)


# Funkcija isbandyti algortimo gautus ivercius
def mtip_backpropogation_test(bp_x1, bp_w1, bp_b1, bp_w2, bp_b2):
    y_1 = y1_funkcija(bp_w1, bp_b1, bp_x1)  # Issiunciame parametrus ir apskaiciuojame lygtis
    v_1 = n_aktyvimas(y_1)  # issiunciame lygties atsakymus i aktyvavimo funkcija
    y = y_funkcija(v_1, bp_w2, bp_b2)
    klaidu_tikrinimas(n_output(bp_x1), y)
    x_mtip_compare(bp_w1, bp_b1, bp_w2, bp_b2)

    return y


# Parametru atnaujinimo funkcija
def o_parametru_atnaujinimas(l_r, w2, b2, e, v1, y1, b1, w1, x_i, y):
    a = len(w2)
    v1_inv = [0] * len(w2)
    delta = [0] * len(w2)
    delta_1 = [0] * len(w2)
    n = 0
    while n != a:
        w2[n] = w2[n] + l_r * e * y
        n += 1
    b2 = b2 + l_r * e * 1

    n = 0
    while n != a:
        v1_inv[n] = v1[n] * (1 - v1[n])
        n += 1
    n = 0

    while n != a:
        delta[n] = v1_inv[n] * e * w2[n]
        n += 1
    n = 0

    while n != a-1:

        w2[n+1] = w2[n] + l_r * delta[n] * y1[n]
        w1[n+1] = w1[n] + l_r * delta[n] * x_i
        n += 1
    n = 0
    while n != a:
        delta_1[n] = (1 / 1 + m.exp((-1 * x_i) * w1[n] - b1[n]))*(1-1/(1 + m.exp((-1*x_i)*w1[n]-b1[n])))*e*w2[n]
        n += 1
    n = 0

    while n != a:
        w1[n] = w1[n] + l_r * delta_1[n] * x_i
        b1[n] = b1[n] + l_r * delta_1[n]
        n += 1
    n3 = mtip_backpropogation_test(x_i, w1, b1, w2, b2)
    # x_mtip_compare(w1, b1, w2, b2)
    # print(y_gen)
    return n3  # , w1, b1, w2, b2


# Tinklo sukurimo funkcija
def mlp_sukurimas(x1, hidden, rate):

    w1_1 = w_parametru_generavimas(hidden)  # Sugeneruojame w1 parametra pagal norima neuronu skaiciu
    b1_1 = b_parametru_generavimas(hidden)  # Sugeneruojame b1 parametra pagal norima neuronu skaiciu
    y_1 = y1_funkcija(w1_1, b1_1, x1)  # Issiunciame parametrus ir apskaiciuojame lygtis
    v_1 = n_aktyvimas(y_1)  # issiunciame lygties atsakymus i aktyvavimo funkcija
    w2_1 = w_parametru_generavimas(hidden)  # Sugeneruojame w1 parametra pagal norima neuronu skaiciu
    b2_1 = b_parametru_generavimas(1)  # Sugeneruojame b1 parametra pagal norima neuronu skaiciu
    y = y_funkcija(v_1, w2_1, b2_1)
    n1 = n_output(x1)
    e = klaidu_tikrinimas(n1, y)
    n2 = o_parametru_atnaujinimas(rate, w2_1, b2_1, e, v_1, y_1, b1_1, w1_1, x1, y)
    n3 = tikslumas(n1, n2)
    rez = 0

    # Kol skirtumas ne mazesnis jei nustatytas kartosim (0.1 / -0.1)
    while 0.1 < n3 > -0.1:
        n2 = o_parametru_atnaujinimas(rate, w2_1, b2_1, e, v_1, y_1, b1_1, w1_1, x1, y)
        n3 = tikslumas(n1, n2)
        if n3 > 2:
            rez = 1
            break
        if n3 < -2:
            rez = 1
            break
    if rez == 0:
        print("n_output:", n1)
        print("backpropogation:", n2)
        print("Skirtumas", n3)
    return rez


# Main
x_n = 20  # neurono iejimu kiekis
x = n_input(x_n)

n_target = 1  # X pavyzdzio pasirinkimas
n_hidden = 4  # pasleptu neuronu kiekis
l_rate = 0.3  # learning rate

# Kol negausime norimo atsakymo kartosime
a = 1
while a != 0:
    a = mlp_sukurimas(x[n_target], n_hidden, l_rate)

