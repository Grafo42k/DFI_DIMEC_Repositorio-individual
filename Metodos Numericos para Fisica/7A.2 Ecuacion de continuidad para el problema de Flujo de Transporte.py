# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Program traffic"
#Resuelve la ecuacion de continuidad para el problema de transporte de flujo

import numpy as np
import matplotlib.pyplot as plt

#Seleccionar los parametros numericos (tims step, grid spacing, etc...)
method = input('Metodo numerico; (1. FTCS, 2. Lax, 3. Lax-Wendroff): ')
N      = input('Ingrese el numero de puntos en la Grid: ')
L      = 400.   #Tamaño del sistema
h      = L/N    #Grid spacing para condiciones de borde periodicas
v_max  = 25.    #Maxima velocidad del auto (m/s)

print('Suggested timstep is ', h/v_max)
tau = input(' Ingrese el paso del tiempo (tau): ')
print('Last car starts moving after ',(L/4)/(v_max*tau),' steps')

nstep   = input('Ingrese el numero de pasos: ')
coeff   = tau/(2*h)         #Coeficiente usado por todos los esquemas
coefflw = tau**2 / (2*h**2) #coeficiente usado por Lax-Wendroff

#Configurar las condiciones iniciales y de contorno
rho_max  = 1.0                 #Densidad maxima
Flow_max = 0.25*rho_max*v_max  #Flujo maximo

Flow = np.empty(N)
cp   = np.empty(N); cm = np.empty(N)

#Condicion inicial es un pulso cuadrado desde x = -L/4 a x = 0
rho = np.zeros(N)
for i in range(int(N/4), int(N/2)):
    rho[i] = rho_max       #Maxima densidad en el pulso cuadrado
    
rho[int(N/2)] = rho_max/2  #Tratar de correr sin esta linea

#Establecer las condiciones de borde periodico
ip      = np.arange(N) + 1
ip[N-1] = 0                  #ip = i+1 con condiciones periodicas
im      = np.arange(N) - 1 
im[0]   = N-1                #im = i-1 con condiciones periodicas

#Inicializamos las condiciones de ploteo.
iplot      = 1
xplot      = (np.arange(N)-1/2.)*h - L/2
rplot      = np.empty((N,nstep+1))
tplot      = np.empty(nstep+1)
rplot[:,0] = np.copy(rho)
tplot[0]   = 0

for i in range(nstep):
    #Calculamos el flujo = densidad*velocidad
    Flow[:] = rho[:]*(v_max*(1-rho[:]/rho_max))
    
    #Calcular los nuevos valores de la densidad
    if method == 1:
        rho[:] = rho[:] - coeff*(Flow[ip] - Flow[im])
    elif method == 2:
        rho[:] = 0.5*(rho[ip] + rho[im]) - coeff*(Flow[ip] - Flow[im])
    else:
        cp[:]  = v_max*(1 - (rho[ip] + rho[:])/rho_max);
        cm[:]  = v_max*(1 - (rho[:] + rho[im])/rho_max);
        rho[:] = rho[:] - coeff*(Flow[ip] - Flow[im]) 
        + coefflw*(cp[:]*(Flow[ip] - Flow[:]) - cm[:]*(Flow[:] - Flow[im]))
    
    #Guardar la densidad para plotear
    rplot[:,iplot] = np.copy(rho)
    tplot[iplot]   = tau*(i+1)
    iplot += 1
    
#Grafico de la densidad versus posicion y tiempo
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax  = fig.gca(projection = '3d')
Tp, Xp = np.meshgrid(tplot[0:iplot],xplot)
ax.plot_surface(Tp, Xp, rplot[:,0:iplot], rstride=1, cstride=1, cmap=cm.gray)
ax.view_init(elev = 30., azim = 10.)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('rho')
ax.set_title('Densidad versus posicion y tiempo')
plt.show()

#Graficar lineas de contorno
levels = np.linspace(0.,1.,num=11)
ct = plt.contour(xplot,tplot,np.flipud(np.rot90(rplot)),levels)
plt.clabel(ct,fmt='%1.2f')
plt.xlabel('x')
plt.ylabel('Tiempo')
plt.title('Contornos de densidad')
plt.show()

    
    




