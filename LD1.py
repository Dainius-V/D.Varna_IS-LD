# https://en.wikipedia.org/wiki/Naive_Bayes_classifier
# Pridedame bibliotekas random, numpy
import random
import numpy as np


def lygtis(a, c):  # Funkcijos lygtis a=x1, c=x2
    # Funkcija apskaiciuoti lygciai x1 * w1 + c * w2 + b
    w1 = random.random()
    w2 = random.random()
    b = random.random()
    return a * w1 + c * w2 + b


def atsakymo_tikrinimas(g):
    # Funkcija tikrinti lygties gauta rezultata
    y = 0
    if g > 0:
        y = 1
    elif g <= 0:
        y = -1
    return y


def klaidos_skaiciavimas(y, t):
    # Funkcija gauto klaidos paskaiciavimui su gautu ir norimu rezultatu
    e = t - y
    return e


def klasifikatorius():
    print("Paleidziame Klasifikatoriu")
    x1, x2, T = np.loadtxt('Data.txt', delimiter=",", unpack=True)  # Sutalpinam Data.txt duomenis i masyvus
    # Apskaiciuojame v1..v13 lygtis su x1 ir x2 duomenimis is data.txt
    # Apskaiciuojame e1..e13 klaidas patikrine gautus atsakymus v1..v13
    v1 = lygtis(x1[0], x2[0])
    e1 = klaidos_skaiciavimas(atsakymo_tikrinimas(v1), T[0])

    v2 = lygtis(x1[1], x2[1])
    e2 = klaidos_skaiciavimas(atsakymo_tikrinimas(v2), T[1])

    v3 = lygtis(x1[2], x2[2])
    e3 = klaidos_skaiciavimas(atsakymo_tikrinimas(v3), T[2])

    v4 = lygtis(x1[3], x2[3])
    e4 = klaidos_skaiciavimas(atsakymo_tikrinimas(v4), T[3])

    v5 = lygtis(x1[4], x2[4])
    e5 = klaidos_skaiciavimas(atsakymo_tikrinimas(v5), T[4])

    v6 = lygtis(x1[5], x2[5])
    e6 = klaidos_skaiciavimas(atsakymo_tikrinimas(v6), T[5])

    v7 = lygtis(x1[6], x2[6])
    e7 = klaidos_skaiciavimas(atsakymo_tikrinimas(v7), T[6])

    v8 = lygtis(x1[7], x2[7])
    e8 = klaidos_skaiciavimas(atsakymo_tikrinimas(v8), T[7])

    v9 = lygtis(x1[8], x2[8])
    e9 = klaidos_skaiciavimas(atsakymo_tikrinimas(v9), T[8])

    v10 = lygtis(x1[9], x2[9])
    e10 = klaidos_skaiciavimas(atsakymo_tikrinimas(v10), T[9])

    v11 = lygtis(x1[10], x2[10])
    e11 = klaidos_skaiciavimas(atsakymo_tikrinimas(v11), T[10])

    v12 = lygtis(x1[11], x2[11])
    e12 = klaidos_skaiciavimas(atsakymo_tikrinimas(v12), T[11])

    v13 = lygtis(x1[12], x2[12])
    e13 = klaidos_skaiciavimas(atsakymo_tikrinimas(v13), T[12])

    # Susumuojame gautas klaidas ir isspausdiname rezultata
    e = abs(e1) + abs(e2) + abs(e3) + abs(e4) + abs(e5) + abs(e6) + abs(e7) + abs(e8) + abs(e9) + abs(e10) + abs(
        e11) + abs(e12) + abs(e13)
    print("e = [", e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, "]")
    print("e = ", e)


def algoritmas():
    # Funkcija mokymo algortimui klasifikatoriaus parametrams apskaiciuoti
    print("Paleidziam algoritma")
    # Ikeliame kintamuosius is data.txt
    xx1, xx2, xxT = np.loadtxt('data.txt', delimiter=",", unpack=True)
    # Paskelbiame reikalingus kintamuosiuos
    etotal = 1
    eta = 0
    xT = [1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1]  # Norimi atsakymai
    w1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    w2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    e = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Paleidziam kol klaidu suma (e) nera lygi 0
    while etotal != 0:

        m = 0
        n = 0
        # random.uniform(-1,1)
        # random.random()
        # Ciklo pradzioje duodame pradinius kintamuosius
        w1[0] = random.uniform(-1, 1)
        eta = random.uniform(0, 1)
        # 0 < eta < 1
        w2[0] = random.uniform(-1, 1)
        b[0] = random.uniform(-1, 1)

        # Paleidziame cikla gauti norimus rezultatus pagal gautus parametrus
        while m != 13:

            if n == 12:
                # Jeigu masyvo pozicija 12
                if y[n] == xT[n]:
                    # Jei masyvo 12 pozicija turi norima atsakyma
                    # sustabdo cikla
                    break
                else:
                    # Jei ne atliekamas lygties paskaiciavimas, atsakymo patikrinimas ir klaidos paskaiciavimas
                    v[n] = xx1[n] * w1[n] + xx2[n] * w2[n] + b[n]
                    y[n] = atsakymo_tikrinimas(v[n])
                    e[n] = klaidos_skaiciavimas(xT[n], y[n])
                    # Stabdome cikla
                    break
            else:
                # Jei masyvo pozicija ne 12
                if y[n] == xT[n]:
                    # Tikrina ar turimas lygties atsakymas tinka norimam rezultatui
                    if y[n + 1] == xT[n + 1]:
                        # Jei taip, tikrina sekancio masyvo reiksme su norimu rezultatu
                        # Padidiname kintamuosius 1
                        n += 1
                        m += 1
                    else:
                        # Jei sekancio masyvo reiksme neturi tinkamo atsakymo atliekame parametru naujima
                        w1[n + 1] = w1[n] + eta * e[n] * xx1[n]
                        w2[n + 1] = w2[n] + eta * e[n] * xx2[n]
                        b[n + 1] = b[n] + eta * e[n]
                        # Padidiname kintamuosius 1
                        n += 1
                        m += 1

                else:
                    # Jei turimas lygties neatitinka norimo rezultato atliekame skaiciavimus
                    v[n] = xx1[n] * w1[n] + xx2[n] * w2[n] + b[n]
                    y[n] = atsakymo_tikrinimas(v[n])
                    e[n] = klaidos_skaiciavimas(xT[n], y[n])

                    if y[n + 1] == xT[n + 1]:
                        # Tikriname sekancio masyvo turima rezultata
                        n += 1
                        m += 1
                    else:
                        # Jei neatitinka norimo rezultato, atnaujiname parametrus
                        w1[n + 1] = w1[n] + eta * e[n] * xx1[n]
                        w2[n + 1] = w2[n] + eta * e[n] * xx2[n]
                        b[n + 1] = b[n] + eta * e[n]
                        # Papildom kintamuosius
                        n += 1
                        m += 1
        # Apskaiciavus visu masyvu veiksmus sudedame turimas klaidas
        # Jei etotal nelygi 0, kartojame cikla, jei lygi 0 stabdome
        etotal = sum(e)
    # Gavus norima etotal isvedame parametrus su kuriais gavome atsakyma
    print("Done")
    print("w1 = ", w1)
    print("w2 = ", w2)
    print("b = ", b)
    print("eta = ", eta)
    print("T = ", xT)
    print("y = ", xT)
    print("e = ", e)
    print("etotal = ", etotal)


# ######################~~~~~~MAIN~~~~~~########################################
# Norimi atsakymai T [ 1,1,1,1,1,1,1,1,1,-1,-1,-1,-1]
klasifikatorius()
algoritmas()

