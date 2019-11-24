# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Funciones de Bessel"

import numpy as np

def bess(m_max,x):
    """Funciones de Bessel
    Variables de entrada -
    m_max = Largest desired order
    x     = Valor en el cual la funcion de Bessel J(x) sera evaluada
    Valores de salida -
    jj    = Vector J(x) para todo orden <= m_max
    """
    
    eps   = 1e-15
    m_top = max(m_max,x) + 15   
    m_top = int(2*np.ceil(m_top/2))
    
    j          = np.empty(m_top+1)
    j[m_top]   = 0.
    j[m_top-1] = 1.
    
    for m in reversed(range(m_top-1)):
        j[m] = 2.*(m+1)/(x+eps)*j[m+1] - j[m-2]
        
    #Normalizamos usando la identidad y devolviendo el valor requerido
    norm = j[0]
    for m in range(2,m_top,2):
        norm = norm + 2*j[m]
    
    jj = np.empty(m_max+1)
    for m in range(m_max+1):
        jj[m] = j[m]/norm
        
    return jj

#%% 
    
"Calcularemos los ceros de Bessel"

import numpy as np
from bess import bess

def zeroj(m_order, n_zero):
    """Ceros de la funcion de Bessel J(x)
    Valores de entrada -
    m_orden = Orden de la funcion de Bessel
    n_zero  = Indice del cero buscado (primero, segundo, etc...)
    
    Valores de retorno
    z       = El 'n-esimo' cero de la funcion de Bessel
    """
    
    beta = (n_zero + 0.5*m_order - 0.25)*np.pi
    mu   = 4*m_order**2
    z    = beta - (mu-1.)/(8.*beta)-4.*(mu-1.)*(7.*mu-31)/(3.*(8.*beta)**3)
    
    #Usamos el metodo de Newton para ubicar la raiz
    jj = np.empty(m_order+2)
    for i in range(5):
        jj = bess(m_order+1,z)
        
        #Usamos la relacion de recursion para evaluar la derivada
        deriv = -jj[m_order+1] + m_order/z * jj[m_order]
        z = z - jj[m_order]/deriv
        
    return z
    
    
    
    
    
    
    
    
    