# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Program ftdemo"
#Demostracion de la Transformada de Fourier discreta usando el metodo de la 
#suma directa de la Transformada de Fourier rapida

import numpy as np
import matplotlib.pyplot as plt

#Inicializar la onda sinusoidal a ser transformada
N     = input('Ingrese el numero de puntos: ')
freq  = input('Ingrese la frecuencia de la onda: ')
phase = input('Ingrese la phase de la onda: ')

tau = 1.                        #Incremento del tiempo
t   = np.arange(N)*tau          #t = [0, tau, 2*tau, ...] 
y   = np.empty(N)
for i in range(N):
    y[i] = np.sin(2*np.pi*t[i]*freq + phase)
f = np.arange(N)/(N*tau)        #f = [0, 1/(N*tau), ...]

#Calculamos la transformada usado el metodo correspondiente
yt = np.zeros(N,dtype=complex)
Method = input('Calcular transformada por 1.- Suma directa; 2.- FFT: ')

import time
startTime = time.time()
if Method == 1:
    twoPiN = -2* np.pi * (1j) /N   #(1j) = sqrt(-1)
    for k in range(N):
        for j in range(N):
            expTerm = np.exp(twoPiN*j*k)
            yt[k] += y[j]*expTerm
            
else:
    yt = np.fft.fft(y)
stopTime = time.time()

print('Tiempo transcurrido = ',stopTime - startTime,' segundos')

#Graficar la transformada
plt.subplot(1,2,1)  #Grafico de la izquierda
plt.plot(t,y)
plt.title('Serie de tiempo original')
plt.xlabel('Tiempo')

plt.subplot(1,2,2)  #Grafico de la derecha
plt.plot(f,np.real(yt), '-', f, np.imag(yt), '--')
plt.legend(['Real','Imaginario'])
plt.title('Transformada de Fourier')
plt.xlabel('Frecuencia')

plt.show()

#Graficar el espectro de potencia de la serie de tiempo
powspec = np.empty(N)
for i in range(N):
    powspec[i] = abs(yt[i])**2
plt.semilogy(f,powspec, '-')
plt.title('Espectro de potencias (no normalizado)')
plt.xlabel('Frecuencia')
plt.ylabel('Potencia')
plt.show()

