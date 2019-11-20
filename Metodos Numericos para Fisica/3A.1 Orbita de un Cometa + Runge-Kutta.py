# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"program orbit"
# Computar la orbita de un cometa al rededor del Sol usando los métodos de 
# Euler, Euler-Cromer, Runge-Kutta de orden 4.

"Probar con los siguientes valores"
# r = 1 UA; v = pi (UA/yr); n = 200 iteraciones, tau = 0.005 yr

import numpy as np
import matplotlib.pyplot as plt


#Se define la funcion gravrk usada en el metodo de Runge-Kutta
def grvrk(s,t,GM):
    """Devuelve el lado derecho de la ecuacion diferencial de Kepler a traves
       de utilizar Runge-Kutta
       Inputs
       s      Vector de estado [r(1) r(2) v(1) v(2)]
       t      Tiempo (no usado)
       GM     Parametro G*M (cte gravitacional * masa del Sol)
       
       Output
       Derivadas [dr(1)/dt dr(2)/dt dv(1)/dt dv(2)/dt]
    """
    
    #Calcular la aceleracion
    r = np.array([s[0],s[1]]) #Reescribe vector s en posicion y velocidad
    v = np.array([s[2],s[3]])
    accel = -GM*r/np.linalg.norm(r)**3
    
    #dDevolver derivadas
    deriv = np.array([v[0], v[1], accel[0], accel[1]])
    return deriv

#Se define el Runge-Kutta de orden 4
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

# Se define el el Runge-Kutta Adaptativo
def rka(x, t, tau, err, derivsRK, param):
    """Elementos de entrada -
     x        = Valor actual de la variable dependiente
     t        = Variable independiente
     tau      = Tamaño del paso
     err      = Error deseado
     derivsRK = Funcion que devuelve dx/dt
     param    = Parametros extra
     
     Elementos de salida -
     xSmall   = Nuevo valor de la variable dependiente
     t        = Nuevo valor de la variable independiente
     tau      = Paso del tiempo sugerido para el siguiente rka
    """
    
    #Condiciones iniciales
    tSave, xSave = t, x
    safe1, safe2 = 0.9, 4 #Factores de seguridad
    eps = 1e-15
    
    xTemp  = np.empty(len(x))
    xSmall = np.empty(len(x)); xBig = np.empty(len(x))
    
    maxTry = 100
    for i in range(maxTry):
        #Tomar dos pequeños pasos de tiempo
        half_tau = 0.5*tau
        xTemp = rk4(xSave, tSave, half_tau, derivsRK, param)
        t = tSave + half_tau
        xSmall = rk4(xTemp, t, half_tau, derivsRK, param)
        
        #Tomar un gran paso de tiempo
        t = tSave + tau
        xBig = rk4(xSave, tSave, tau, derivsRK, param)
        
        #Calcular el error estimado
        scale = err*(abs(xSmall) +abs(xBig))/2.
        xDiff = xSmall - xBig
        errorRatio = np.max(np.absolute(xDiff) / (scale + eps))
        
        #Estimar el nuevo valor de tau
        tau_old = tau
        tau = safe1*tau_old*errorRatio**(-0.20)
        tau = max(tau, tau_old/safe2)
        tau = min(tau, safe2*tau_old)
        
        #Si el error es aceptable, devolver los valores calculados
        if errorRatio < 1:
            return np.array([xSmall, t, tau])
        
    #Si nunca se satisface la cota de error
    print('ERROR: Adaptative Runge-Kutta routine failed')
    return np.array([xSmall, t, tau])


#Ingresar posicion y velocidad inicial del cometa
r0 = input('Ingrese distancia inicial (AU): ')
v0 = input('Ingrese velocidad tangencial inicial (AU/yr): ')
r = np.array([r0,0])
v = np.array([0,v0])
state = np.array([r[0], r[1], v[0], v[1]]) #Se usará por el R-K

"Fijamos las constantes fisicas importantes"
GM    = 4*np.pi**2 #Cte Gravitacional * Masa del sol (AU³/yr²)
m     = 1.0        #Masa del cometa
error = 1e-3       #Error adaptativo por R-K
t     = 0.0        #Tiempo inicial

#Generar un loop sobre un determinado numero de pasos
nStep  = input('Ingrese el numero de pasos: ')
tau    = input('Ingrese el paso del tiempo (yr) ')
Metodo = input('Metodo: 1- Euler, 2- Euler-Cromer, 3- R-K, 4- R-K Adaptativo ')

rplot     = np.empty(nStep); thplot  = np.empty(nStep)
tplot     = np.empty(nStep); kinetic = np.empty(nStep)
potential = np.empty(nStep)

for i in range(nStep):
    
    rplot[i]     = np.linalg.norm(r)
    thplot[i]    = np.arctan2(r[1],r[0])
    tplot[i]     = t
    kinetic[i]   = 0.5*m*np.linalg.norm(v)**2 #Calculamos la energia
    potential[i] = -GM*m/np.linalg.norm(r)    #Calculamos el potencial
    
    "Calculamos la nueva posicion y velocidad usando el metodo correspondiente"
    if Metodo == 1:
        accel = -GM*r/np.linalg.norm(r)**3
        r = r + tau*v #Paso iterativo de Euler
        v = v + tau*accel
        t = t + tau
    
    elif Metodo == 2:
        accel = -GM*r/np.linalg.norm(r)**3
        v = v + tau*accel
        r = r + tau*v #Paso iterativo de Euler-Cromer
        t = t + tau
        
    elif Metodo == 3:
        state = rk4(state, t, tau, grvrk, GM)
        r = np.array([state[0], state[1]]) #Runge-Kutta orden 4
        v = np.array([state[2], state[3]])
        t = t + tau
        
    else:
        [state, t, tau] = rka(state, t, tau, error, grvrk, GM)
        r = np.array([state[0], state[1]])
        v = np.array([state[2], state[3]])
        
#Graficar la trayectoria del cometa
ax = plt.subplot(111, projection = 'polar') #Grafico polar para la orbita
ax.plot(thplot, rplot, '-')
ax.set_title('Distance (AU)')
ax.grid(True)
plt.show()

#Graficar la energia del cometa en funcion del tiempo
totalE = kinetic + potential 
plt.plot(tplot, kinetic, 'r-.')
plt.plot(tplot, potential, 'b--')
plt.plot(tplot, totalE, 'k-')
plt.legend(['Kinetic','Potential','Total']);
plt.xlabel('Time (yr)')
plt.ylabel(r'Energy ($M AU^2/yr^2$)')
plt.grid()
plt.show()



