import numpy as np
import math
import matplotlib.pyplot as plt

#Definir  intervalo
a = 0
b = 4

#Passo
n= 4
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

#Método de RK23 (exatidão de segunda, erro de terceira ordem)
for i in range (0,n-1):
    k_1=h*f(t[i],y[i])
    k_2=h*f(t[i]+(h/2), y[i]+((k_1*h)/2))
    k_3 = h * f(t[i] + (h / 2), y[i] + ((k_2 * h) / 2))
    k_4 = h * f(t[i] + h , y[i] + (k_3 * h))
    y[i+1]= y[i] + (1/6)*(k_1+2*k_2+2*k_3+k_4)
    Ept[i+1] = abs((analitica(t[i+1]) - y[i+1])/analitica(t[i+1]))*100


print(Ept)

#Grafico

plt.plot(t,y,'or',label='Método de RK23')
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


