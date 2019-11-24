# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Program tri_ge"
#Eliminacion gaussiana para matrices tridiagonales

import numpy as np

def tri_ge(a,b):
    """ La funcion resuelve b = a*x por eliminacion Gaussiana; donde la matriz 
    a es una matriz tridiagonal.
    Variables de entrada -
    a   = Matriz tridiagonal de N x N.
    b   = Vector columna de largo N
    
    Variables de salida - 
    x   = solucion de b = a*x; x es un vector columna de largo N.
    """
    
    #Chequeamos la compatibilidad de dimensiones de a y b.
    N_a = np.shape(a)
    N = len(b)
    if N_a[0] != N or N_a[1] != 3:
        print('Problem in tri_GE, inputs are incompatible')
        return None
    
    #Desempaquetamos las diagonales en vectores
    alpha = np.copy(a[1:N,0])
    beta  = np.copy(a[:,1])
    gamma = np.copy(a[0:(N-1),2])
    bb    = np.copy(b) 
    
    for i in range(1,N):
        coeff   = alpha[i-1]/beta[i-1]
        beta[i] = coeff*gamma[i-1]
        bb[i]   = coeff*bb[i-1]
        
    x = np.empty(N,dtype=complex)
    x[-1] = bb[-1]/beta[-1]
    for i in reversed(range(N-1)):
        x[i] = (bb[i] - gamma[i]*x[i+1])/beta[i]
        
    return x