# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Program newtn"
#Encuentra el estado estable del modelo de Lorenz usando el metodo de newton.

import numpy as np
import matplotlib.pyplot as plt


#Definimos nuestra funcion correspondiente al metodo de newton
def fnewt(x,a):
    """Funcion usada por el metodo de Newton con N-variables
    
    Valores de entrada -
    x = Vector de estado [x y z]
    a = Parametros [r sigma b]
    
    Valores de salida - 
    f = Lado derecho del modelo de Lorenz [dx/dt dy/dt dz/dt]
    D = Matriz Jacobiana, D(i,j) = df(j)/dx(i)
    """
    
    #Evaluar f(i)
    f    = np.empty(3)
    f[0] = a[1]*(x[1] - x[0])
    f[1] = a[0]*x[0] - x[1] - x[0]*x[2]
    f[2] = x[0]*x[1] - a[2]*x[2]
    
    #Evaluar D(i,j)
    D      = np.empty((3,3))
    D[0,0] = -a[1]           #df(0)/dx(0)
    D[0,1] = a[0] - x[2]     #df(1)/dx(0) ...
    D[0,2] = x[1]
    D[1,0] = a[1]
    D[1,1] = -1.
    D[1,2] = x[0]
    D[2,0] = 0.
    D[2,1] = -x[0]
    D[2,2] = -a[2]
    
    return [f, D]

#Configuramos la adivinanza inicial y los parametros
x0 = np.array(input('Ingrese una adivinanza inicial (vector fila): '))
x  = np.copy(x0)
a  = np.array(input('Ingrese el parametro a: '))

#Configuraciones de las iteraciones
nStep = 10
xp = np.empty((len(x),(nStep)))
xp[:,0] = np.copy(x[:])        #Se guarda la adivinanza inicial para graficar

for i in range(nStep):
    #Evaluar la funcion f y su Jacobiano
    [f, D] = fnewt(x,a)
    
    #Encuentra dx usando eliminacion Gaussiana
    dx = np.linalg.solve(np.transpose(D),f)
    
    #Actualizar la raiz estimada
    x = x - dx               #Iteracion de Newton para encontrar un nuevo x
    xp[:,i] = np.copy(x[:])  #Guardar el actual valor estimado
    
#Imprimir el valor final estimado para la raiz
print('After ',nStep,' iterations the root is')
print(x)

#Plot las iteraciones desde la adivinanza inicial hasta la estimacion final
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax  = fig.gca(projection = '3d')
ax.plot(xp[0,:], xp[1,:], xp[2,:], '*-')
ax.plot([x[0]],[x[1]], [x[2]], 'ro')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Estado estable del modelo de Lorenz')
plt.show()
    
    
    
    
    
    