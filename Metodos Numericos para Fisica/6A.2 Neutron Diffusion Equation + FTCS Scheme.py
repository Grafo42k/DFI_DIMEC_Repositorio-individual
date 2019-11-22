# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Program neutrn"
#Solves the neutron diffusion equation using the FTSC scheme

"Probar el codigo con los valores N = 61, tau = 5e-4, L = 2 y nStep = 12000"

import numpy as np
import matplotlib.pyplot as plt

#Inicializamos los parametros
tau = input('Ingrese paso del tiempo: ')
N   = input('Ingrese numero de puntos en la grid: ')
L   = input('Ingrese el largo del sistema: ')

#El sistema se extenderá desde x = -L/2 a x = L/2
h = L/float(N-1)  #Tamaño de la grid
D = 1.            #Coeficiente de difusion
C = 1.            #Tasa de generacion

coeff  = D*tau/h**2
coeff2 = C*tau

if coeff < 0.5:
    print('Se espera solucion estable')
else:
    print('WARNING: Se espera solucion numericamente inestable')
    
#configuramos las condiciones iniciales y de contorno
nn     = np.zeros(N)  #Inicializa densidad cero en todos los puntos
nn_new = np.zeros(N)  #Inicializa arreglo temporal usado en FTCS
nn[int(N/2.)] = 1/h   #Condicion inicial es una funcion delta en el centro

"Condiciones de borde son nn[0] = nn[N-1] = 0"

xplot     = np.arange(N)*h - L/2
iplot     = 0             #Contador de graficos ploteados
nStep     = input('Ingrese el numero de pasos de tiempo: ')
nplots    = 50            #Numero de plots instantaneos a tomar
plot_step = nStep/nplots  #Numero de pasos de tiempo entre plots

nnplot = np.empty((N,nplots))
tplot  = np.empty(nplots)
nAve   = np.empty(nplots)

for i in range(nStep):
    nn[1:(N-1)] = (nn[1:(N-1)] + coeff*(nn[2:N] + nn[0:(N-2)] - 2*nn[1:(N-1)])
                   + coeff2*nn[1:(N-1)])
    
    if (i+1)% plot_step < 1:
        nnplot[:,iplot] = np.copy(nn)
        tplot[iplot]    = (i+1)*tau
        nAve[iplot]     = np.mean(nn)
        iplot += 1
        print('Finished ',i,' of ',nStep,' steaps')
        
#Ploteamos la densidad versus x y tt como un grafido 3D
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection = '3d')
Tp, Xp = np.meshgrid(tplot, xplot)
ax.plot_surface(Tp, Xp, nnplot, rstride=2, cstride=2, cmap=cm.gray)
ax.set_xlabel('Tiempo')
ax.set_ylabel('x')
ax.set_zlabel('n(x,t)')
ax.set_title('Neutron Difussion')
plt.show()

#Plot la densidad promedio de neutrones versus el tiempo
plt.plot(tplot,nAve,'*')
plt.xlabel('Tiempo')
plt.ylabel('Densidad promedio')
plt.title(r'$L" = %g, ($L_c = \pi$)' % L)
plt.show()
