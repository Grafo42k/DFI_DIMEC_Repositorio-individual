# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Funcion rombf"
#Calcula integrales usando el metodo de integracion de Romberg

import numpy as np

def rombf(a,b,N,func,param):
    """Funcion calcula la integral usando el algoritmo de Romberg
    Valores de entrada -
    a, b  = Limites inferior y superior de la integral
    N     = Tabla de Romberg de N x N
    func  = Funcion a integrar
    param = Set de parametros que se le entregan a la funcion
    
    Valores de salida -
    R     = Tabla de Romberg, con la mejor estimacion del valor de la integral
    """
    
    #Calculamos los primeros terminos R(1,1)
    h       = b - a     #Tamaño mas grueso del panel
    npanels = 1         #Actual numero de paneles
    
    R      = np.zeros((N+1,N+1))
    R[1,1] = h/2. * (func(a,param) + func(b,param))
    
    #Generamos un loop sobre las filas i = 2, 3, ..., N
    for i in range(2,N+1):
        #Calculamos la suma usando la regla del trapecio
        h = h/2.
        npanels *= 2
        sumT = 0.
        
        #El siguiente loop irá por los impares k = 1, 3, ..., npanels-1
        for k in range(1,npanels,2):
            sumT += func(a + k*h, param)
            
        #Calculamos las entradas de la tabla de Romberg
        R[i,1] = 0.5*R[i-1,1] +  h*sumT
        m = 1
        for j in range(2,i+1):
            m *= 4
            R[i,j] = R[i,j-1] + (R[i,j-1] - R[i-1,j])/(n-1)
    
    return R

#%%
    
"Funcion errintg"
#Entrega el error de la funcion rombf

import numpy as np

def errintg(x,param):
    """Error function integrand
    Valores de entrada -
    x     = Valor donde el integrando es evaluado
    param = Lista de parametros
    
    Valores de salida - 
    f     = Error del integrando
    """
    
    f = np.exp(-x**2)
    return f
    
    
    
    
    
    