# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Program fftpoi"
#Resuelve la ecuacion de Poisson usando el metodo de la TF multiple

import numpy as np
import matplotlib.pyplot as plt

#Inicializamos los parametros del problema (system size, grid spacing, etc.)
eps0 = 8.8542e-12                #Permitividad(C²/(Nm²))
N    = 50                        #Numero de puntos en la grid
L    = 1.                         #Tamaño del sistema
h    = L/N                       #Espaciado de la grid
x    = (np.arange(N) + 1/2)*h   #Coordenadas de la grid
y    = np.copy(x)                #Grid cuadrada

print('Sistema es una placa cuadrada de largo ',L)

#Configuramos la densidad de carga rho(i,j)
rho = np.zeros((N,N))    #Inicializa la densidad de carga en cero
M   = input('Enter the number of line charges: ')
for i in range(M):
    print(' For charge #',i)
    r  = input('Enter position [x, y]: ')
    ii = int(r[0]/h + 0.5)
    jj = int(r[1]/h + 0.5)
    q  = input('Enter charge density: ')
    rho[ii,jj] += q/h**2
          
#Calculamos la matriz P
cx = np.cos((2*np.pi/N)*np.arange(N))
cy = np.copy(cx)

numerator  = -h**2/(2.*eps0)
tinyNumber = 1e-20           #Evitar divisiones por cero

P = np.empty((N,N))
for i in range(N):
    for j in range(N):
        P[i,j] = numerator/(cx[i]+cy[j] -2.+tinyNumber) #Podria ser *

#Calculamos el potencial usando MFT
rhoT = np.fft.fft2(rho)   #Transforma rho al dominio de frecuencias
phiT = rhoT*P             #Calcula phi en el dominio de frecuencias
phi  = np.fft.ifft2(phiT) #Transformada inversa de phi
phi  = np.real(phi)

#Calculamos el campo electrico como E = - grad phi
[Ex, Ey] = np.gradient(np.flipud(np.rot90(phi)))
for i in range(N):
    for j in range(N):
        magnitude = np.sqrt(Ex[i,j]**2 + Ey[i,j]**2)
        Ex[i,j] /= -magnitude
        Ey[i,j] /= -magnitude
        
#Plot potencial
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection = '3d')
Xp, Yp = np.meshgrid(x,y)
ax.contour(Xp,Yp,np.flipud(np.rot90(phi)),35)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel(r'$\Phi(x,y)$')
plt.show()

#Graficamos el campo electrico
plt.quiver(Xp,Yp,Ey,Ex)
plt.title('E Field (Direction)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('square')
plt.axis([0.,L,0.,L])
plt.show()
