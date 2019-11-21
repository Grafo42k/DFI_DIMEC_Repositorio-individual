# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Program lsfdemo"
#Genera un Data Set y ajusta la curva a los datos

import numpy as np
import matplotlib.pyplot as plt

"En primer lugar se generan las funciones necesarias para el programa"

#Funcion linreg ajusta una linea recta a un Data Set
def linreg(x,y,sigma):
    """ Funcion para realizar una regresion lineal
    Valores de entrada -
    x     = Variable independiente
    y     = Variable dependiente
    sigma = Error estimado en y
    
    Valores de salida -
    a_fit  = Ajuste de parametros; a(1) = intercepto; a(2) = pendiente
    sig_a  = Error estimado en los parametros a()
    yy     = Curva de ajuste a los datos
    chisqr = Estadistico χ²
    """
    
    #Evaluar varias sumas sobre sigma
    s = 0; sx = 0; sy = 0; sxy = 0; sxx = 0
    for i in range(len(x)):
        sigmaTerm = sigma[1]**(-2)
        
        s   += sigmaTerm
        sx  += x[i]*sigmaTerm
        sy  += y[i]*sigmaTerm
        sxy += x[i]*y[i]*sigmaTerm
        sxx += x[i]**2 * sigmaTerm    
    denom = s*sxx - sx**2
    
    #Calcular el intercepto y la pendiente
    a_fit    = np.empty(2)
    a_fit[0] = (sxx*sy - sx*sxy)/denom
    a_fit[1] = (s*sxy - sx*sy)/denom
    
    #Calcular el error para pendiente e intercepto
    sig_a    = np.empty(2)
    sig_a[0] = np.sqrt(sxx/denom)
    sig_a[1] = np.sqrt(s/denom)
    
    #Evalua el ajuste de curva en cada punto y computa χ².
    yy     = np.empty(len(x))
    chisqr = 0
    for i in range(len(x)):
        yy[i]   = a_fit[0] + a_fit[1]*x[i] #Ajuste de curva a los datos
        chisqr += ((y[i] - yy[i])/sigma[i])**2 #Estadistico χ²
        
    return [a_fit, sig_a, yy, chisqr]


#Se define la funcion pollsf que ajusta un polinomio a un Data Set
def pollsf(x, y, sigma, M):
    """Funcion de ajuste polinomial
    
    Variables de entrada -
    x     = Variable independiente
    y     = Variable dependiente
    sigma = Error estimado en y
    M     = Numero de parametros usados para ajustar la curva
    
    Variables de salida- 
    a_fit  = Ajuste de parametros; a(1) es el intercepto; a(2) es la pendiente
    sig_a  = Error estimado en el parametro a()
    yy     = Curva de ajuste a los datos
    chisqr = Estadistico χ²
    """
    
    #Escribir el vector b y designar la matriz A
    N = len(x)
    b = np.empty(N)
    A = np.empty((N,M))
    for i in range(N):
        b[i] = y[i]/sigma[i]
        for j in range(M):
            A[i,j] = x[i]**j / sigma[i]
            
    #Calculamos la matriz de correlacion C
    C = np.linalg.inv(np.dot(np.transpose(A),A))
    
    #Calculamos los coeficientes de minimos cuadrados a_fit
    a_fit = np.dot(C, np.dot(np.transpose(A), np.transpose(b)))
    
    #Calculamos el error estimado para los coeficientes
    sig_a = np.empty(M)
    for j in range(M):
        sig_a[j] = np.sqrt(C[j,j])
        
    #Evaluamos la curva y ajustamos cada punto. Calculamos χ²
    yy = np.zeros(N)
    chisqr = 0
    for i in range(N):
        for j in range(M):
            yy[i] += a_fit[j]*x[i]**j #Ajuste de curva
        
        chisqr += ((y[i] - yy[i]) / sigma[i])**2
    
    return [a_fit, sig_a, yy, chisqr]

"Ahora comenzamos con el modelo del programa"

#Se inicializan los datos a ser ajustados. 
# Se utilizara una Data cuadrática a la cual se le agregará un numero random.
print('Curve fit data is created using the quadratic')
print('y(x) = c(0) + c(1)*x + c(2)*x²')

c = np.array(input('Ingrese los coeficientes como [c(0) c(1) c(2)]: '))
N = 50                 #Numero de puntos en el Data Set
x = np.arange(1,N+1)   #x = [1, 2, ..., N]
y = np.empty(N)

alpha= input('Ingrese el error estimado: ')
sigma = alpha* np.ones(N) #Vector de error constante
np.random.seed(0)
for i in range(N):
    r    = alpha * np.random.normal() #Vector aleatorio distribuye Gaussiana
    y[i] = c[0] + c[1]*x[i] + c[2]*x[i]**2 + r

#Ajustar la Data a una linea o polinomio mas general
M = input('Ingrese el numero de parametros de ajuste (2 para una linea): ')
if M == 2:
    #Regresion Lineal
    [a_fit, sig_a, yy, chisqr] = linreg(x, y, sigma)
else:
    #Ajuste polinomial
    [a_fit, sig_a, yy, chisqr] = pollsf(x, y, sigma, M)
    
#Imprimir los parametros de ajuste, incluyendo sus errores
print('Parametros de Ajuste')
for i in range(M):
    print('a[',i,'] = ',a_fit[i],' +/- ',sig_a[i])

#Graficar los datos, los errores y la funcion de ajuste
plt.errorbar(x,y,sigma,None,'*')
plt.plot(x,yy,'-')
plt.xlabel('$x_i$')
plt.ylabel('$y_i$ and $Y(x)$')
plt.title('chi^2 = %d, N - M = %d' % (chisqr,N - M))
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    