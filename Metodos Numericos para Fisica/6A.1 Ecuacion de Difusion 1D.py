# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Program dftcs"
#Se resuelve la ecuacion de difusion usando el esquema FTCS

import numpy as np
import matplotlib.pyplot as plt

#Inicializamos los parametros iniciales (time step, grid spacing, etc...)
tau   = input('Ingrese el paso de tiempo: ')
N     = input('Ingrese el numero de puntos en la grid: ')
L     = 1.      #El sistema se extiende desde x = -L/2 a x = L/2
h     = L/(N-1) #Tamaño de la grid
kappa = 1.      #Coeficiente de difusion

coeff = kappa*tau/h**2
if coeff < 0.5:
    print('Se espera solucion estable')
else:
    print('WARNING: Se espera solucion inestable')
    
#Ajustar las condiciones iniciales y de contorno
tt = np.zeros(N)       #Inicializamos la temperatura en 0 en todos los puntos.
tt[int(N/2)] = 1./h    #Colocamos una delta en el centro

"Las condiciones de borde son tt[0] = tt[N-1] = 0"

#Se configuran las variables de plot
xplot     = np.arange(N)*h - L/2   #Guardamos los x para plotear
iplot     = 0                      #Contador para cuantificar los plot
nStep     = 300                    #Maximo numero de iteraciones 
nplots    = 50                     #Numero de snapshots (plots) a tomar
plot_step = nStep/nplots           #Numero de pasos de tiempo entre plots

ttplot = np.empty((N,nplots))
tplot  = np.empty(nplots)

for i in range(nStep):
    tt[1:(N-1)] = (tt[1:(N-1)] + coeff*(tt[2:N] + tt[0:(N-2)] - 2*tt[1:(N-1)]))
    
    if (i+1)% plot_step < 1:
        ttplot[:,iplot] = np.copy(tt)
        tplot[iplot]    = (i+1)*tau
        iplot += 1
        
#Ploteamos la temperatura versus x y t en nuestro wire-mesh plot.

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection = '3d')
Tp, Xp = np.meshgrid(tplot, xplot)
ax.plot_surface(Tp,Xp, ttplot, rstride=2, cstride=2, cmap=cm.gray)
ax.set_xlabel('Tiempo')
ax.set_ylabel('x')
ax.set_zlabel('T(x,t)')
ax.set_title('Diffusion of a delta spike')
plt.show()

#Ploteamos la temperatura versus x y t como un plot de lineas de contorno
levels = np.linspace(0, 10, num = 21)
ct = plt.contour(tplot, xplot, ttplot, levels)
plt.clabel(ct, fmt = '%1.2f')
plt.xlabel('Tiempo')
plt.ylabel('x')
plt.title('Temperature contour plot')
plt.show

"Probar utilizando t = 1e-4 y N = 61 (Solucion numericamente estable)"


