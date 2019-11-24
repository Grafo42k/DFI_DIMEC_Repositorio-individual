# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Program dsmceq"
#Simula relajacion al equilibrio de un gas usando el algoritmo DSMC

import numpy as np
import matplotlib.pyplot as plt


"Se crea la Class sortList"
#Used to manage sorting lists

class sortList:
    """Class used for sorting particles into cells."""
    def __init__(self,ncell_in,npart_in):
        self.ncell = ncell_in
        self.npart = npart_in
        
        self.cell_n = np.zeros(ncell_in, dtype=int)
        self.index  = np.empty(ncell_in, dtype=int)
        self.Xref   = np.empty(npart_in, dtype=int)

"Se crea la sub rutina Sorter"
#Produce una lista ordenada con particulas en celdas
def sorter(x,L,sD):
    """Variables de entrada -
    x   = Posicion de particulas
    L   = Tamaño del sistema
    sD  = Objeti contenido en listas ordenadas
    """
    
    #Find the cell address for each particle
    npart = sD.npart
    ncell = sD.ncell
    
    jx = np.empty(npart,dtype=int)
    for ipart in range(npart):
        jx[ipart] = int(x[ipart]*ncell/L)
        jx[ipart] = min(jx[ipart],(ncell-1))
        
    #Count the number of particles in each cell
    sD.cell_n = np.zeros(ncell)
    for ipart in range(npart):
        sD.cell_n[ jx[ipart] ] += 1
    
    #Build index list as cumulative sum of the number of particles in each cell
    m = 0
    for jcell in range(ncell):
        sD.index[jcell] = m
        m              += sD.cell_n[jcell]
        
    #Build cross reference list
    temp = np.zeros(ncell,dtype=int)
    for ipart in range(npart):
        jcell        = jx[ipart]
        k            = sD.index[jcell] + temp[jcell]
        sD.Xref[k]   = ipart
        temp[jcell] += 1
        
"Se crea la sub rutina colider"
#Evalua colisiones usando el algoritmo DSMC
def colider(v,crmax,tau,selxtra,coeff,sD):
    """Variables de entrada -
    v       = Velocidad de las particulas
    crmax   = Rapidez relativa maxima estimada en una celda
    tau     = Paso del tiempo
    selxtra = Seleccion extra sobre el ultimo paso de tiempo
    coeff   = Coeficiente en el computo del numero de pares seleccionados
    sD      = Objeto que contiene listas ordenandose
    
    Valores de salida -
    col     = Numero total de colisiones procesadas
    """
    
    ncell = sD.ncell
    col   = 0
    vrel  = np.empty(3)
    
    for jcell in range(ncell):
        
        number = sD.cell_n[jcell]
        
        if number > 1:
            
            select = coeff*number*(number-1)*crmax[jcell] + selxtra[jcell]
            nsel           = int(select)
            selxtra[jcell] = select - ncell
            crm            = crmax[jcell]
            
            for isel in range(nsel):
                k   = int(np.floor(np.random.uniform(0,number)))
                kk  = int(np.ceil(k + np.random.uniform(0,number-1))%number)
                ip1 = sD.Xref[k + sD.index[jcell]]
                ip2 = sD.Xref[kk + sD.index[jcell]]
                
                cr = np.linalg.norm(v[ip1,:] - v[ip2,:])
                if cr > crm:
                    crm = cr
                    
                if cr/crmax[jcell] > np.random.random():
                    col    += 1
                    vcm     = 0.5*(v[ip1,:] + v[ip2,:])
                    cos_th  = 1. - 2.*np.random.random()
                    sin_th  = np.sqrt(1. - cos_th**2)
                    phi     = 2*np.pi*np.random.random()
                    vrel[0] = cr*cos_th
                    vrel[1] = cr*sin_th*np.cos(phi)
                    vrel[2] = cr*sin_th*np.sin(phi)
                    v[ip1,:] = vcm + 0.5*vrel
                    v[ip2,:] = vcm - 0.5*vrel
                    
            crmax[jcell] = crm
    return col


    
"Aqui inicia el programa para el usuario"
#Inicializamos las constantes (masa de las particulas, diametro, etc)
boltz   = 1.3806e-23    #Cte de Boltzmann (J/K)
mass    = 6.63e-26      #Masa de un atomo de argon (kg)
diam    = 3.66e-10      #Diametro efectivo del atomo de argon(m)
T       = 273.          #Temperatura(K)
density = 1.78          #Densidad del argon a STP(kg/m³)
L       = 1e-6          #Tamaño del sistema (1 micron)

npart   = input('Ingrese el numero de particulas en la simulacion: ')
eff_num = density/mass * L**3 /npart
print('Cada particula representa ',eff_num,' atomos')

np.random.seed(0)
x = np.empty(npart)
for i in range(npart):
    x[i] = np.random.uniform(0.,L)

v_init = np.sqrt(3*boltz*T/mass)
v      = np.zeros((npart,3))
for i in range(npart):
    v[i,0] = v_init * (1 - 2*np.floor(2*np.random.random()))
    
#Plot la distribucion de velocidades inicial
vmag = np.sqrt(v[:,0]**2 + v[:,1]**2 + v[:,2]**2)
plt.hist(vmag, bins=20, range=(0,1000))
plt.title('Initial speed distribution')
plt.xlabel('Speed (m/s)')
plt.ylabel('Number')
plt.show()

#Inicializamos las variables usadas para evaluar colisiones
ncell   = 15
tau     = 0.2*(L/ncell)/v_init
vrmax   = 3*v_init*np.ones(ncell)
selxtra = np.zeros(ncell)
coeff   = 0.5*eff_num*np.pi*diam**2*tau/(L**3/ncell)
coltot  = 0

#Declaramos sortList
sortData = sortList(ncell, npart)

#Loop para el numero de pasos de tiempo deseado
nstep = input('Ingrese el numero total de pasos de tiempo: ')
for istep in range(nstep):
    #todas las particulas se mueven balisticamente
    x = x + v[:,0]*tau
    x = np.remainder(x+L,L)
    
    #Sort las particulas en las celdas
    sorter(x,L,sortData)
    
    #Evaluamos las coliciones de particulas
    col = colider(v,vrmax,tau,selxtra,coeff,sortData)
    coltot = coltot + col
    
    #Periodicamente mostramos el progreso actual
    if (istep+1)%10 < 1:
        vmag = np.sqrt(v[:,0]**2 + v[:,1]**2 + v[:,2]**2)
        plt.hist(vmag,bins=11, range=(0,1000))
        plt.title('Done %d of %d steps; %d collisions'%(istep, nstep, coltot))
        plt.xlabel('Speed (m/s)')
        plt.ylabel('Number')
        plt.show()
        
#Plot the final speed distribution
vmag = np.sqrt(v[:,0]**2 + v[:,1]**2 + v[:,2]**2)
plt.hist(vmag,bins=11, range=(0,1000))
plt.title('Final speed distribution')
plt.xlabel('Speed (m/s)')
plt.ylabel('Number')
plt.show()

    
    
    