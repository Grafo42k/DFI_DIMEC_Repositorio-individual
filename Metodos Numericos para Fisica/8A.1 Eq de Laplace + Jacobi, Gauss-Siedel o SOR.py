# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Program relax"
#Resuelvel la ecuacion de laplace usando Jacobi, Gaus-Seidel o SOR method.

import numpy as np
import matplotlib.pyplot as plt

#Inicializamos los parametros del sistema (system size, spacing grid, etc...)
method = input('Elegir metodo (1.Jacobi 2.Gauss-Siedel 3.SOR): ')
N      = input('Ingrese el numero de puntos en la grid: ')
L      = 1.
h      = L/(N-1)
x      = np.arange(N)*h
y      = np.arange(N)*h

#Seleccionar el factor de sobre relajacion (SOR solamente)
if method == 3:
    omega0pt = 2./(1+ np.sin(np.pi/N))  #Optimo teorico
    print('Theorical optimum omega = ',omega0pt)
    omega = input('Ingrese el omega deseado: ')
    
phi0 = 1      #Potencial en y = L.
phi = np.empty((N,N))
for i in range(N):
    for j in range(N):
        phi[i,j] = phi0*4/(np.pi*np.sinh(np.pi)
        )*np.sin(np.pi*x[i]/L)*np.sinh(np.pi*y[j]/L)
        
#Seteamos las condiciones de borde
phi[0,:]  = 0
phi[-1,:] = 0
phi[:,0]  = 0
phi[:,-1] = phi0*np.ones(N)

print('Potential at y = L equals ', phi0)
print('Potential is zero on all other boundaries')

newphi = np.copy(phi)      #Copiamos la solucion (Se usara en Jacobi)
iterMax = N**2             #Seteamos un maximo de iteraciones
change = np.empty(iterMax)
changeDesired = 1e-4       #Para cuando el cambio esta limitado por este valor
print('Desired fractional change = ',changeDesired)

for k in range(iterMax):
    changeSum = 0
    
    if method == 1:   #Metodo de Jacobi
        for i in range(1,N-1):
            for j in range(1,N-1):
                newphi[i,j] = 0.25*(phi[i+1,j] + phi[i-1,j] + phi[i,j-1] 
                + phi[i,j+1])
                changeSum += abs(1 - phi[i,j]/newphi[i,j])
                
        phi = np.copy(newphi)
    
    elif method == 2: #Metodo G-S
        for i in range(1,N-1):
            for j in range(1,N-1):
                temp = 0.25*(phi[i+1,j] + phi[i-1,j] + phi[i,j-1] 
                + phi[i,j+1])
                changeSum += abs(1 - phi[i,j]/temp)
                phi[i,j] = temp
                
    else:             #Metodo SOR
        for i in range(1,N-1):
            for j in range(1,N-1):
                temp = 0.25*(phi[i+1,j] + phi[i-1,j] + phi[i,j-1] 
                + phi[i,j+1]) + (1-omega)*phi[i,j]
                changeSum += abs(1 - phi[i,j]/temp)
                phi[i,j] = temp
                
    #Chequeamos si el cambio es suficientemente pequeño al iterar
    change[k] = changeSum/(N-2)**2
    if (k+1)%10 < 1:
        print('After ',k+1,' iterations, fractional change = ',change[k])
    if change[k] < changeDesired:
        print('Desired accuracy achieved after ',k+1,' iterations')
        print('Breaking out of main loop')
        break
    
#Ploteamos el potencial final estimado con lineas de contorno
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

levels = np.linspace(0.,1., num = 11)
ct = plt.contour(x,y,np.flipud(np.rot90(phi)),levels)
plt.clabel(ct, fmt ='%1.2f')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Potential after %g iterations' % k)
plt.show()

#Plot el estado final del potencial con contornos y superficies
fig = plt.figure()
ax = fig.gca(projection = '3d')
Xp, Yp = np.meshgrid(x,y)
ax.plot_surface(Xp,Yp,np.flipud(np.rot90(phi)),rstride=1,cstride=1,cmap=cm.gray)
ax.view_init(elev=30., azim=210)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel(r'$\Phi(x,y)$')
plt.show()







        
                
                
                
                
                