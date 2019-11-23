# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Program advect"
#Resuelve la ecuacion de adveccion usando diferentes esquemas numericos.

import numpy as np
import matplotlib.pyplot as plt

#Seleccionar los parametros numericos (time step, grid spacing, etc...)
method = input('Elija un metodo numerico, 1. FTCS, 2. Lax, 3. Lax-Wendroff ')
N      = input('Escoger el numero de puntos en la Grid: ')
L      = 1.    #Tamaño del sistema
h      = L/N   #Espaciado de la Grid
c      = 1.    #Velocidad de la onda

print('Time for wave to move one grid spacing is ',h/c)

tau     = input('Ingrese el paso del tiempo: ')
coeff   = -c*tau/(2.*h) #Coeficiente usado por todos los esquemas
coefflw = 2*coeff**2    #Coeficiente usado por el esquema L-W

print('Wave circles system in ',L/(c*tau),' steps')
nStep = input('Ingrese el numero de pasos: ')

#Configuramos las condiciones iniciales y de contorno
sigma  = 0.1                   #Ancho del pulso Gausiano
k_wave = np.pi/sigma           #Numero de onda
x      = np.arange(N)*h - L/2  #Puntos de la grid

#La condicion inicial es un pulso Gaussian-cosine
a = np.empty(N)
for i in range(N):
    a[i] = np.cos(k_wave*x[i])*np.exp(-x[i]**2 / (2*sigma**2))
    
#Usamos condiciones de borde periodicas
ip      = np.arange(N) + 1
ip[N-1] = 0                 #ip = i+1 con condicion de borde periodica
im      = np.arange(N) - 1
im[0]   = N-1               #im = i-1 con condicion de borde periodica

#Inicializamos las variables de ploteo
iplot  = 1                     #Contador de plots
nplots = 50                    #Numero deseado de plots
aplot  = np.empty((N,nplots))

tplot      = np.empty(nplots)
aplot[:,0] = np.copy(a)        #Guarda el estado inicial
tplot[0]   = 0                 #Guarda el tiempo inicial (t = 0)
plotStep   = nStep/nplots + 1  #Numero de pasos entre plots

for i in range(nStep):
    if method == 1:
        a[:] = a[:] + coeff*(a[ip] - a[im])
    elif method == 2:
        a[:] = 0.5*(a[ip] + a[im]) + coeff*(a[ip] - a[im])
    else:
        a[:] = (a[:] + coeff*(a[ip] - a[im]) + coefflw*(a[ip] + a[im] -2*a[:]))
        
    if (i+1)%plotStep < 1:
        aplot[:,iplot] = np.copy(a)
        tplot[iplot]   = tau*(i+1)
        iplot += 1
        
        print(i,' out of ',nStep,' steps completed')
        
#Plotear el estado inicial y final
plt.plot(x,aplot[:,0],'-',x,a,'--')
plt.legend(['Initial ','Final'])
plt.xlabel('x')
plt.ylabel('a(x,t)')
plt.show()

#Plot la amplitud de la onda versus la posicion y el tiempo
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection = '3d')
Tp, Xp = np.meshgrid(tplot[0:iplot],x)
ax.plot_surface(Tp,Xp,aplot[:,0:iplot], rstride=1, cstride=1, cmap=cm.gray)
ax.view_init(elev = 30, azim = 190)
ax.set_ylabel('Posicion')
ax.set_xlabel('Tiempo')
ax.set_zlabel('Amplitud')
plt.show()





