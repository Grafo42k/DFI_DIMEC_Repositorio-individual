# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Program pendulum"
#Se calcula la evolucion de un pendulo simple usando metodo de Euler y Verlet.

import numpy as np
import matplotlib.pyplot as plt

#Seleccionar el tipo de metodo numerico a utilizar
Metodo = input('Elegir un metodo (Euler: 1, Verlet: 2) ')

#Ingresar posicion y velocidad inicial del pendulo
theta0 = input(' Ingrese angulo inicial (degrees): ')
theta = theta0*np.pi/180
omega = 0.0 #Velocidad angular inicial

"Configurar las constantes fisicas de interes"
w    = 1.0 #Se refiere a la constante g/L
t    = 0.0 #Tiempo inicial
irev = 0   #Para contar el numero de ciclos

tau = input('Ingrese el paso del tiempo (segundos): ')

#Tomar un paso atras (Backward) para comenzar el metodo de Verlet
accel = -w*np.sin(theta) #Aceleracion de gravedad
theta_old = theta - omega*tau + 0.5*accel*tau**2

nstep = input('Numero de pasos de tiempo: ')
t_plot = np.empty(nstep); th_plot = np.empty(nstep)
T = np.empty(nstep) #Para guardar el periodo

for i in range(nstep):
    #Guardamos el angulo y tiempo para plotear
    t_plot[i]  = t
    th_plot[i] = theta*180/np.pi
    t = t + tau
    
    #Calculamos la nueva posicion y velocidad usando el metodo respectivo
    accel = -w*np.sin(theta) #Aceleracion de gravedad
    if Metodo == 1:
        theta_old = theta
        theta = theta + tau*omega
        omega = omega + tau*accel
        
    else:
        theta_new = 2*theta - theta_old + tau**2*accel
        theta_old = theta
        theta = theta_new
        
    #Verificar si el pendulo ha pasado por theta = 0
    if theta*theta_old < 0: #test del cambio de signo
        print('Punto de retorno al tiempo t =',t)
        if irev == 0:
            time_old = t
        else:
            T[irev-1]= 2*(t - time_old)
            time_old = t
        irev += 1 #Incrementar el numero de ciclos
        
#Estimar el periodo de oscilacion, incluyendo el error
nT    = irev-1 #Numero de periodos de tiempo medidos
meanT = np.mean(T[0:nT])
error = np.std(T[0:nT])/np.sqrt(nT)
print('Periodo promedio = ',nT, '+/- ',error)

#Graficar las oscilaciones de theta versus t
plt.plot(t_plot, th_plot, 'r-')
plt.xlabel('Tiempo')
plt.ylabel(r'$\theta$ (degrees)')
plt.show()


