# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Program balle"
# Computa la trayectoria de una pelota de baseball, incluyendo la resistencia
# del aire por medio del metodo de Euler.

import numpy as np
import matplotlib.pyplot as plt

#Ingresar posicion inicial de la pelota
y0 = input('Ingrese altura inicial (metros): ')
#Configuramos el vector posicion
r0 = np.array([0,y0])

#Ingresar velocidad y angulo inicial de la pelota
velocidad = input('Ingrese velocidad inicial (m/s): ')
theta = input('Ingrese angulo inicial (degrees): ')

#Configuramos el vector velocidad
v0 = np.array([velocidad*np.cos(theta*np.pi/180),
               velocidad*np.sin(theta*np.pi/180)])

#Generar una copia (nuevo objeto) de la posicion y velocidad
r = np.copy(r0)
v = np.copy(v0)

"Configurar los parametros fisicos (masa, Cd, etc.)"
Cd = 0.35   #Coeficiente de arrastre
A  = 4.3e-3 #Seccion eficaz del proyectil (m²)
g  = 9.81   #Aceleracion de gravedad (m/s²)
m  = 0.145  #Masa del proyectil (m)

roce = input('Existe roce con el aire? (Si: 1,No: 0): ')
if roce == 0:
    rho = 0
else:
    rho = 1.2 #Densidad del aire (kg/m³)
    
roce_aire = -0.5*Cd*rho*A/m #Constante de roce con el aire

"Loop hasta que proyectil toque suelo o alcance max numero de iteraciones"
tau = input('Ingrese paso del tiempo, tau (seg): ')
maxstep = 1000 #Numero maximo de iteraciones

xplot  = np.empty(maxstep);  yplot  = np.empty(maxstep)
xNoAir = np.empty(maxstep);  yNoAir = np.empty(maxstep)

for i in range(maxstep):
    #Guardar posicion (computada y teorica) para graficar
    xplot[i] = r[0]
    yplot[i] = r[1]
    
    t = i*tau #Tiempo actual
    xNoAir[i] = r0[0] + v0[0]*t
    yNoAir[i] = r0[1] + v0[1]*t - 0.5*g*t**2
    
    #Calcular la aceleracion del proyectil
    accel = roce_aire * np.linalg.norm(v)*v #Resistencia del aire
    accel[1] = accel[1] - g
    
    #Calcular la nueva posicion y velocidad usando metodo de Euler
    r = r + tau*v
    v = v + tau*accel
    
    #Si proyectil toca el suelo, detener iteraciones
    if r[1] < 0:
        final = i+1
        xplot[final] = r[0] #Guarda ultimo valor calculado
        yplot[final] = r[1]
        break
    
#Imprimir en pantalla máximo rango y tiempo de vuelo
print('Rango maximo es: ', r[0], ' metros')
print('Tiempo de vuelo es: ', final*tau, ' segundos')

"Graficar la trayectorial del proyectil"
#Se marca una linea recta que representa el nivel del suelo.
xground = np.array([0, xNoAir[final -1]])
yground = np.array([0, 0])

#Plot de la trayectoria calculada
plt.plot(xplot[0:final+1], yplot[0:final+1], '+',
         xNoAir[0:final],   yNoAir[0:final],  '-',
         xground, yground, 'k-')
plt.legend(['Metodo de Euler', 'Teorico (sin aire)'])
plt.xlabel('Rango (metros)')
plt.ylabel('Altura (metros)')
plt.title('Movimiento del proyectil')
plt.show()


