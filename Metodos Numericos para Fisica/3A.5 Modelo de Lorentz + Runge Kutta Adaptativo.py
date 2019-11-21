# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Program Lorentz"
# Se utiliza RK para calcular la evolucion temporal del modelo de Lorentz

"Probar utilizando los siguientes valores"
# r0 = [1,1,20], r = 28, n = 200

import numpy as np
import matplotlib.pyplot as plt

#Se define la funcion utilizada en el R-K
def lorzrk(s,t,param):
    """    Valores de entrada -
    s      Vector de estado [x y z]
    t      Tiempo
    param  Parametros [r sigma b]
    
    Valores de salida -
    deriv  Derivadas [dx/dt dy/dt dz/dt]
    """
    
    x, y, z = s[0], s[1], s[2]
    r       = param[0]
    sigma   = param[1]
    b       = param[2]
    
    #Devolver las derivadas
    deriv    = np.empty(3)
    deriv[0] = sigma*(y-x)
    deriv[1] = r*x - y - x*z
    deriv[2] = x*y - b*z
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

#Se configura el estado inicial y los parametros respectivos
state = np.array(input('Ingresar posicion inicial [x, y, z]: '))
r     = input('Ingrese el parametro r: ')
sigma = 10.
b     = 8./3.
param = np.array([r, sigma, b])
tau   = 1.
err   = 1e-3

#Se genera el loop de calculo
time = 0
nstep   = input('Ingresa el numero de pasos: ')
tplot   = np.empty(nstep)
tauplot = np.empty(nstep)
xplot, yplot, zplot = np.empty(nstep), np.empty(nstep), np.empty(nstep)

for i in range(nstep):
    x, y, z    = state[0], state[1], state[2]
    tplot[i]   = time
    tauplot[i] = tau
    xplot[i]   = x; yplot[i] = y; zplot[i] = z
    
    if (i+1)%50 < 1:
        print('Finished ',i,' steps out of ', nstep)
    
    #Calcular el nuevo estado usando RK
    [state, time, tau] = rka(state, time, tau, err, lorzrk, param)
    
#Imprimir los tiempos max y min devueltos por rka
tauMax = np.max(tauplot[1:nstep])
tauMin = np.min(tauplot[1:nstep])
print('Adaptative time step: Max = ',tauMax,' Min = ',tauMin)
    
#Graficar la serie de tiempo x(t)
plt.plot(tplot, xplot, 'k-')
plt.xlabel('Tiempo')
plt.ylabel('x(t)')
plt.title('Modelo de Lorentz')
    
#Graficar el espacio de fase para x, y, z
x_ss = np.empty(3); y_ss = np.empty(3); z_ss = np.empty(3)
x_ss[0] = 0
y_ss[0] = 0
z_ss[0] = 0

x_ss[1] = np.sqrt(b*(r-1))
y_ss[1] = x_ss[1]
z_ss[1] = r-1

x_ss[2] = -np.sqrt(b*(r-1))
y_ss[2] = x_ss[2]
z_ss[2] = r-1

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot(xplot, yplot, zplot, '-')
ax.plot(x_ss, y_ss, z_ss, '*')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_tittle('Espacio de fase del modelo de Lorentz')
plt.show()
