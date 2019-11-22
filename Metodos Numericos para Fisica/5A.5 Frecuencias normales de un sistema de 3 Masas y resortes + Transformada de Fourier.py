# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Program sprfft"
#Calcula la evolucion de un sistema masa-resorte usando la Transformada de 
#Fourier para obtener las autofrecuencias

import numpy as np
import matplotlib.pyplot as plt

#Se define la función sprrk
def sprrk(s, t, param):
    """Devuelve el lado derecho de un sistema de 3 masas unidas a resortes
    Valores de entrada -
    s     = Vector de estado [x(1) x(2) ... v(3)] 
    t     = Tiempo
    param = (Constante del resorte)/(Masa del bloque)
    
    Valores de salida -
    deriv = [dx(1)/dt dx(2)/dt ... dx(3)/dt]
    """
    
    deriv    = np.empty(6)
    deriv[0] = s[3]
    deriv[1] = s[4]
    deriv[2] = s[5]
    param2   = -2.*param
    deriv[3] = param2*s[0] + param*s[1]
    deriv[4] = param2*s[1] + param*(s[0] + s[2])
    deriv[5] = param2*s[2] + param*s[1]
    return deriv

#Se define la funcion Runge-Kutta
def rk4(x,t,tau,derivsRK,param):
    """Elementos de entrada -
     x        = Valor actual de la variable dependiente
     t        = Variable independiente
     tau      = Tamaño del paso
     derivsRK = Funcion que devuelve dx/dt
     param    = Parametros extra
    """
    
    half_tau = 0.5*tau
    F1     = derivsRK(x,t,param)
    t_half = t + half_tau
    xtemp  = x + half_tau*F1
    F2     = derivsRK(xtemp, t_half, param)
    xtemp  = x + half_tau*F2
    F3     = derivsRK(xtemp, t_half, param)
    t_full = t + tau
    rtemp  = x + tau*F3
    F4     = derivsRK(rtemp, t_full, param)
    xout   = x + tau/6.*(F1 + F4 + 2.*(F2 + F3))
    return xout


#Fijamos los parametros para el sistema
x = np.array(input('Ingrese el desplazamiento inicial [x1 x2 x3]: '))
v = np.array([0., 0., 0.]) #Masas inicialmente en reposo

state        = np.array([x[0], x[1], x[2], v[0], v[1], v[2]])
tau          = input('Ingrese el paso del tiempo: ')
k_over_m     = 1. #Cuocuente de (Cte Resorte)/(Masas)

time   = 0.       #Fijamos el tiempo inicial
nStep  = 256      #Numero de pasos en el loop principal
nprint = nStep/8  #Numero de pasos entre el progreso del plot

tplot = np.empty(nStep)
xplot = np.empty((nStep,3))

for i in range(nStep):
    #Usamos Runge-Kutta para encontrar los desplazamientos de las masas
    state = rk4(state,time,tau,sprrk,k_over_m)
    time  = time + tau

    xplot[i,:] = np.copy(state[0:3]) #Guardar posiciones
    tplot[i]   = time
    if i % nprint < 1:
        print('Finished ',i,' out of ',nStep,' steps')

#Graficamos los desplazamientos de las 3 masas
plt.plot(tplot,xplot[:,0],'-',tplot,xplot[:,1],'-.',tplot, xplot[:,2],'--')
plt.legend(['Mass #1 ','Mass #2 ','Mass #3 '])
plt.title('Desplazamiento de las masas (Relativo a la posicion de equilibrio)')
plt.xlabel('Tiempo')
plt.ylabel('Desplazamiento')
plt.show()

#Calcular el espectro de potencia de la serie de tiempo para la masa #1
f     = np.arange(nStep)/(tau*nStep)   #Frecuencia
x1    = xplot[:,0]
x1fft = np.fft.fft(x1)                 #Transformada de fourier
spect = np.empty(len(x1fft))           #Espectro de potencias

for i in range(len(x1fft)):
    spect[i] = abs(x1fft[i])**2
    
    
"""Hanning Window se refiere a un tipo de ventana utilizado para mejorar la
visualizacion de los datos; Existen muchos tipos de ventana, por ej Gauss"""
#Aplicamos la Hanning Window a la serie de tiempo y calculamos el resultado del
#espectro de potencias 
x1w = np.empty(len(x1))
for i in range(len(x1)):
    window = 0.5 * (1. - np.cos(2*np.pi*i/nStep)) #Hanning Window
    x1w[i] = x1[i] * window                       #Windowed time series
    
x1wfft = np.fft.fft(x1w)  #Transformada de Fourier sobre los datos Windowed
spectw = np.empty(len(x1wfft))
for i in range(len(x1wfft)):
    spectw[i] = abs(x1wfft[i])**2
    
#Graficamos el espectro de potencias de ambos tipos de datos
plt.semilogy(f[0:(nStep/2)],spect[0:(nStep)/2], '-',
             f[0:(nStep/2)],spectw[0:(nStep)/2], '--')
plt.title('Espectro de potencias')
plt.xlabel('Frecuencias')
plt.ylabel('Potencia')
plt.show()


"""Mejora en los extremos la señal haciendo que comience y termine mas cerca 
del 0. Ayuda a reducir considerablemente el error debido al Leakage.

El leakage se producen cuando la señal que ha sido transformada a partir de
Fourier no es periodica con periodo T.
"""