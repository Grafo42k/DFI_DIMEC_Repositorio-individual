# -*- coding: utf-8 -*-
"""
Mauricio Muñoz R.
RUT: 17.766.089-2
Github: @Grafo42k
"""

"Program dsmcne"
#Mide la viscocidad en la dilucion de un gas usando DSMC

import numpy as np
import matplotlib.pyplot as plt


"Se crea la Class sampList"
#Use to manage sampling data vectors
class sampList:
    """Class used for sampling density, velocity and temperature"""
    def __init__(self, ncell_in):
        self.ncell = ncell_in
        self.nsamp = 0
        self.ave_n = np.zeros(ncell_in)
        self.ave_u = np.zeros((ncell_in,3))
        self.ave_T = np.zeros(ncell_in)

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

"Se crea la funcion sampler"
#Used to sample the number density, fluid velocity and temperature in the cells
def sampler(x,v,npart,L,sampD):
    """ Variables de entrada -
    x     = Posicion de la particula
    n     = Velocidad de las particulas
    npart = Numero de particulas
    L     = Tamaño del sistema
    sampD = Structure with sampling data
    """
    
    #Compute cell location for each particle
    ncell = sampD.ncell
    jx = np.empty(npart, dtype = int)
    for i in range(npart):
        jx[i] = int(ncell*x[i]/L)
        
    #Initialize running sums of number, velocity and v²
    sum_n  = np.zeros(ncell)
    sum_v  = np.zeros((ncell,3))
    sum_v2 = np.zeros(ncell)
    
    #For each particle, accumulate running sums for its cell
    for ipart in range(npart):
        jcell           = jx[ipart]
        sum_n[jcell]   += 1
        sum_v[jcell,:] += v[ipart,:]
        sum_v2[jcell]  += v[ipart,0]**2 + v[ipart,1]**2 + v[ipart,2]**2
        
    #Use current sums to update sample number, velocity and temperature
    for i in range(3):
        sum_v[:,i] /= sum_n[:]
    sum_v2[:] /= sum_n[:]

    sampD.ave_n[:] += sum_n[:]
    for i in range(3):
        sampD.ave_u[:,i] += sum_n[:]
    sampD.ave_T[:] += sum_v2[:] - (
            sum_v[:,0]**2 + sum_v[:,1]**2 + sum_v[:,2]**2)
    sampD.nsamp += 1
    
"Se crea la funcion mover"
#Used to update particle position. It also processes particles striking the
#thermal walls
def mover(x,v,npart,L,mpv,vwall,tau):
    """Variables de entrada -
    x     = Position of the particles
    v     = Velocities of the particles
    npart = Number of particles in the system
    L     = System length
    mpv   = Most probable velocity off the wall
    vwall = Wall velocities
    tau   = Time step
    
    Variables de salida -
    strikes = Number of particles striking each wall
    delv    = Change of y-velocity at each wall
    """
    
    #Move all particles pretending walls are absent
    x_old = np.copy(x)
    x[:]  = x_old[:] + v[:,0]*tau
    
    #Loop over all particles
    strikes   = np.array([0,0])
    delv      = np.array([0.,0.])
    xwall     = np.array([0.,L])
    vw        = np.array([-vwall,vwall])
    direction = [1,-1]
    stdev     = mpv/np.sqrt(2)
    
    for i in range(npart):
        #Test if particle strikes either wall
        if x[i] <= 0:
            flag = 0
        elif x[i] >= L:
            flag = 1
        else:
            flag = -1
        
        #If particle strikes a wall, reset its position and velocity
        if flag > -1:
            strikes[flag] += 1
            vyInitial = v[i,1]
            
            #Reset velocity components as biased Maxwellian, exponential
            #dist. in x; Gaussian in y and z
            v[i,0] = direction[flag]*np.sqrt(
                    -np.log(1.-np.random.random()))*mpv
            v[i,1] = stdev*np.random.normal() + vw[flag]
            v[i,2] = stdev*np.random.normal()
            
            #Time of flight after leaving wall
            dtr = tau*(x[i] - xwall[flag])/(x[i] - x_old[i])
            
            #Reset position after leaving wall
            x[i] = xwall[flag] + v[i,0]*dtr
            
            #Record velocity change for force measurement
            delv[flag] += v[i,1] - vyInitial
            
    return [strikes, delv]

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
density = 1.685e25      #Densidad del argon a STP(kg/m³)
L       = 1e-6          #Tamaño del sistema (1 micron)
Volume  = L**3          #Volumen del sistema

npart   = input('Ingrese el numero de particulas en la simulacion: ')
eff_num = density*Volume/npart
print('Each simulation particle represents ',eff_num,' atoms')

mfp = Volume/(np.sqrt(2.)*np.pi*diam**2*npart*eff_num)
print('System width is ',L/mfp,' mean free paths')

mpv     = np.sqrt(2*boltz*T/mass)
vwall_m = input('Enter wall velocity as Mach number: ')
vwall   = vwall_m * np.sqrt(5./3. * boltz*T/mass)
print('Wall velocities are ',-vwall,' and ',vwall,' m/s')

#Assign random positions and velocities to particles
np.random.seed(0)
x = np.empty(npart)

for i in range(npart):
    x[i] = np.random.uniform(0.,L)

v = np.zeros((npart,3))
for i in range(npart):
    for j in range(3):
        #Assign thermal velocities using Gausian random numbers
        v[i,j] = np.sqrt(boltz*T/mass)*np.random.normal()
    v[i,1] += 2. * vwall * x[i]/L - vwall
    
#Initialize variables used for evaluating collisions
ncell = 20
tau   = 0.2*(L/ncell)/mpv

vrmax   = 3*mpv*np.ones(ncell)
selxtra = np.zeros(ncell)
coeff   = 0.5*eff_num*np.pi*diam**2*tau/(Volume/ncell)

#Declare sortList object for lists used in sorting
sortData = sortList(ncell, npart)

#Initialize object and variables used in statistical sampling
sampData = sampList(ncell)

tsamp = 0.
dvtot = np.zeros(2)
dverr = np.zeros(2)

#Loop for the desired number of time steps
colSum = 0
strikeSum = np.array([0,0])
nstep = input('Enter total number of time steps: ')

for istep in range(nstep):
    #Move all the particles
    [strikes, delv] = mover(x,v,npart,L,mpv,vwall,tau)
    strikeSum += strikes
    
    #Sort the particles into cells
    sorter(x,L,sortData)
    
    #Evaluate collisions among the particles
    col = colider(v,vrmax,tau,selxtra,coeff,sortData)
    colSum += col
    
    #After initial transient, accumulate statistical samples
    if istep > nstep/10:
        sampler(x,v,npart,L,sampData)
        dvtot += delv
        dverr += delv**2
        tsamp += tau

    #Periodically display the current progress
    if (istep+1)%100 < 1:
        print('Finished ',istep,' of ',nstep,' steps, Collisions = ',colSum)
        print('Total wall strikes: ',strikeSum[0],
              ' (left) ',strikeSum[1],' (right)')
        
#Normalize the accumulated statistics
nsamp = sampData.nsamp
ave_n = (eff_num/(Volume/ncell))*sampData.ave_n/nsamp
ave_u = np.empty((ncell,3))

for i in range(3):
    ave_u[:,i] = sampData.ave_u[:,i]/nsamp

ave_T = mass/(3*boltz) * (sampData.ave_T/nsamp)
dverr = dverr/(nsamp-1) - (dvtot/nsamp)**2
dverr = np.sqrt(dverr*nsamp)

#Compute viscosity from drag force on the walls
force = (eff_num*mass*dvtot)/(tsamp* L**2)
ferr  = (eff_num*mass*dverr)/(tsamp* L**2)
print('Force per unit area is')
print('Left wall: ',force[0],' +/- ', ferr[0])
print('Right wall: ',force[1],' +/- ', ferr[1])

vgrad=2*vwall/L
visc = 1./2.*(-force[0]+force[1])/vgrad
viscerr = 1./2.*(ferr[0]+ferr[1])/vgrad
print('Viscosity = ',visc,' +/- ',viscerr,' N s/m²')

eta = 5.*np.pi/32.*mass*density*(2./np.sqrt(np.pi)*mpv)*mfp
print('Theorical value of viscosity is ',eta,' N s/m²')

#Plot average density, velocity and temperature
xcell = (np.arange(ncell)+0.5)/ncell * L
plt.plot(xcell,ave_n)
plt.xlabel('position')
plt.ylabel('Number density')
plt.show()
plt.plot(xcell,ave_u[:,0],xcell,ave_u[:,1],xcell,ave_u[:,2])
plt.xlabel('position')
plt.ylabel('Velocities')
plt.legend(['x-component','y-component','z-component'])
plt.show()
plt.plot(xcell,ave_T)
plt.xlabel('position')
plt.ylabel('Temperature')
plt.show()
