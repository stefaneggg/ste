import numpy as np
import math
import matplotlib.pyplot as plt

#Definir  intervalo
a = 0
b = 4

#Passo
n= 100
h=(b-a)/n


def f(t,y): #definicao da derivada
    return 4*np.exp(0.8*t)-0.5*y

def analitica(t):
    return 4/1.3*(np.exp(0.8*t)- np.exp(-0.5*t)) + 2*np.exp(-0.5*t)

#Pré-alocação de varáves
t = np.linspace(a,b,n)
y = np.zeros(n)

Ept = np.zeros(n)

#Definição da Condição Inicial
y[0] = 2

#Método de RK23
for i in range (0,n-1):
    k1 = h * f(t[i], y[i])
    k2 = h * f(t[i] + (h/2), y[i] + (0.5 * k1*h))
    k3 = h * f(t[i] + (0.5 * h), y[i] + (0.5 * k2)
    k4 = h * f(t[i] + h, y[i] + k3*h)
    y[i + 1] = y[i] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    Ept[i + 1] = abs((analitica(t[i + 1]) - y[i + 1]) / analitica(t[i + 1])) * 100.0



print(Ept)

# Gráfico 1 – soluções
plt.plot(t, y, 'or', label='Método de Euler')          # <- sem aspas em t e y
plt.plot(t, analitica(t), '--ob', label='Analítica')   # <- sem aspas em t
plt.legend()
plt.xlabel('t'); plt.ylabel('y')
plt.grid(True)
plt.show()

#Grafico

plt.plot(t,y,'or',label='Método de Runge Kutta')
plt.plot(t, analitica(t),'--ob',label='Analitica')
plt.legend()
plt.show()

#Grafico 2
plt.plot(t, Ept,'sg',label='Ept')
plt.show()


plt.plot(f(t,y))
plt.show()

plt.plot(analitica(t))
plt.show()