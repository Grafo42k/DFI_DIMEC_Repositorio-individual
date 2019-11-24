# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Polinomios de Legendre"

import numpy as np

def legndr(n,x):
    """Funcion para generar los polinomios de Legendre
    Variables de entrada - 
    n   = Orden del polinomio devuelto por la funcion
    x   = Valor en el cual será evaluado dicho polinomio
    Valores de retorno -
    P = Vector que contiene P(x) para orden 0, 1, ..., n
    """
    
    #Realizamos una recursion
    p = np.empty(n+1)
    p[0] = 1.         #P(x) para n = 0
    
    if n == 0:
        return p
    
    p[1] = x          #P(x) para n = 1
    if n == 1:
        return p
    
    #Usamos una recursion para obtener el resto de n's
    for i in range(1,n):
        p[i+1] =((2.*i+1)*x*p[i] - i*p[i-1])/(i+1)
        
    return p

