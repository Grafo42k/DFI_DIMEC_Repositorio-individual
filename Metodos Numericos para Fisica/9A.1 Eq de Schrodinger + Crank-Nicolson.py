# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Program schro"
#Movimiento de un paquete Gaussiano resolviendo la Eq de Schrodinger por medio
#del metodo de Crank-Nicolson

import numpy as np
import matplotlib.pyplot as plt

#Inicializamos los parametros (grid spacing, time step, etc...)
i_imag = 1j                     #Imaginario i
N      = input('Ingrese el numero de puntos en la Grid: ')
L      = 100.      #Sistema se extiende desde -L/2 a L/2
h      = L/(N-1)                #Grid size
x      = np.arange(N)*h - L/2   #Coordenadas de los puntos de la grid
h_bar  = 1.                     #Unidades naturales
mass   = 1.                     #Unidades naturales
tau    = input('Ingrese el paso del tiempo: ')

#Configuramos el operador hamiltoniano
ham   = np.zeros((N,N))         #Todos los elementos en cero
coeff = -h_bar**2/(2*mass*h**2)

for i in range(1,N-1):
    ham[i,i-1] = coeff
    ham[i,1]   = -2*coeff       #Set interior row
    ham[i,i+1] = coeff
    
#Primera y ultima fila con condiciones de borde periodicas
ham[0,-1]  = coeff; ham[0,0]   = -2*coeff; ham[0,1]  = coeff
ham[-1,-2] = coeff; ham[-1,-1] = -2*coeff; ham[-1,0] = coeff

#Calculamos la matriz usando Crank-Nicolson
dCN = np.dot(np.linalg.inv(np.identity(N) + 0.5*i_imag*tau/h_bar*ham),
             (np.identity(N) - 0.5*i_imag*tau/h_bar*ham))

#Inicializamos la funcion de onda
x0       = 0.                    #Localizacion del centro del paquete de ondas
velocity = 0.5                   #Velocidad promedio del paquete de ondas

k0       = mass*velocity/h_bar   #Numero de onda promedio
sigma0   = L=10.                 #Desviacion estandar de la funcion de onda
Norm_psi = 1./(np.sqrt(sigma0*np.sqrt(np.pi)))  #Normalizacion

psi = np.empty(N,dtype=complex)
for i in range(N):
    psi[i] = Norm_psi*np.exp(i_imag*k0*x[i])*np.exp(-(x[i]-x0)**2/(2*sigma0**2))
    
#Ploteamos la funcion de onda inicial
plt.plot(x,np.real(psi),'-',x,np.imag(psi),'--')
plt.xlabel('x')
plt.ylabel(r'$\psi(x)$')
plt.legend(('Real ','Imag '))
plt.title('Initial wave function')
plt.show()

#Iniciamos el loop y variables de ploteo
max_iter    = int(L/(velocity*tau) + 0.5)
plot_iter   = max_iter/8
p_plot      = np.empty((N,max_iter + 1))
p_plot[:,0] = np.absolute(psi[:])**2 
iplot       = 0
axisV       = [-L/2., L/2., 0., max(p_plot[:,0])] #Fijar max y min de los ejes

for i in range(max_iter):
    psi = np.dot(dCN,psi)
    
    if (i+1)%plot_iter < 1:
        iplot += 1
        p_plot[:,iplot] = np.absolute(psi[:])**2
        plt.plot(x,p_plot[:,iplot]);  #Display snap-shot of P(x)
        plt.xlabel('x')
        plt.ylabel('P(x,t)')
        plt.title('Finished %d of %d iterations' % (i,max_iter))
        plt.axis(axisV)
        plt.show()
        
#Plot la probabilidad versus posicion y tiempo
pFinal = np.empty(N)
pFinal = np.absolute(psi[:])**2
for i in range(iplot+1):
    plt.plot(x,p_plot[:,i])
plt.xlabel('x')
plt.ylabel('P(x,t)')
plt.title('Probability density at various times')
plt.show()




