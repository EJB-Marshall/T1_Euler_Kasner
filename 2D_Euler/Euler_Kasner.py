"""
Euler_Kasner.py

Finite volume code to evolve the relativistic Euler 
equations on fixed Kasner spacetimes in T1 symmetry.

Equations are evolved using the Kurganov-Tadmor
scheme with Runge-Kutta timestepping.

Boundary conditions are implemented using ghost points
Created by Elliot Marshall 2025-07-16
"""

#import python libraries
import sys
import numpy as np
import h5py
import argparse
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import time


t_start = time.time()


#######################################
#Parser Settings
#######################################

# Initialise parser
parser = argparse.ArgumentParser(description=\
"""This program numerically solves an IVP for the T1-symmetric
Euler Equations on Kasner spacetimes.""")

# Parse files
parser.add_argument('-d','-display', default = False, 
    action='store_true', help=\
"""A flag to indicate if visual display is required.""")
parser.add_argument('-K', type=float,help=\
"""The value of the parameter K.""")
parser.add_argument('-Nx', type=int,help=\
"""The number of grid points in the x-direction.""")
parser.add_argument('-Ny', type=int,help=\
"""The number of grid points in the y-direction.""")
parser.add_argument('-f','-file', help=\
"""The name of the hdf file to be produced.""")
args = parser.parse_args()

#Output Settings
display_output = args.d
store_output = args.f is not None
if store_output and args.f is None:
    print("Euler_Kasner.py: error: argument -f/-file is required")
    sys.exit(1)

#Check Inputs
if args.K is None:
    print("Euler_Kasner.py: error: argument -K is required.")
    sys.exit(1)
if args.Nx is None:
    print("Euler_Kasner.py: error: argument -Nx is required.")
    sys.exit(1)
if args.Ny is None:
    print("Euler_Kasner.py: error: argument -Ny is required.")
    sys.exit(1)

    


##############################################################
# Piecewise Linear Reconstruction
##############################################################

#----------------------------------
# Flux Limiters 
#----------------------------------
def Minmod(r):

    df = np.zeros(np.shape(r))

    df = np.fmax(0,np.fmin(1,r))

    return df

#-----------------------------------------------------
# Piecewise Linear Reconstruction
#-----------------------------------------------------
def linear_reconstruct(u):

    u_plus_x = np.zeros_like(u) ### "Plus" = Reconstruction approaching from the right
    u_minus_x = np.zeros_like(u) ### "Minus" = Reconstruction approaching from the left
    ratio_x = np.zeros_like(u)
    n_x = np.shape(u)[1]
    u_plus_y = np.zeros_like(u) 
    u_minus_y = np.zeros_like(u) 
    ratio_y = np.zeros_like(u)
    n_y = np.shape(u)[2]

    
    ### Compute the ratio of slopes for the limiter
    ### NB: We add small number to the denominator to 
    ### stop NaN issues when neighbouring points are close to equal

    ratio_x[:,1:n_x-1,:] = (u[:,1:n_x-1,:]-u[:,0:n_x-2,:])/(u[:,2:n_x,:]-u[:,1:n_x-1,:]+1e-16)
    
    ### Compute the slope-limited linear reconstruction:

    u_minus_x[:,0:n_x-1,:] = u[:,0:n_x-1,:] + 0.5*Minmod(ratio_x)[:,0:n_x-1,:]*(u[:,1:n_x,:]-u[:,0:n_x-1,:])

    u_plus_x[:,0:n_x-2,:] = u[:,1:n_x-1,:] - 0.5*Minmod(ratio_x)[:,1:n_x-1,:]*(u[:,2:n_x,:]-u[:,1:n_x-1,:])


    # Reconstruction in y

    ratio_y[:,:,1:n_y-1] = (u[:,:,1:n_y-1]-u[:,:,0:n_y-2])/(u[:,:,2:n_y]-u[:,:,1:n_y-1]+1e-16)

    u_minus_y[:,:,0:n_y-1] = u[:,:,0:n_y-1] + 0.5*Minmod(ratio_y)[:,:,0:n_y-1]*(u[:,:,1:n_y]-u[:,:,0:n_y-1])

    u_plus_y[:,:,0:n_y-2] = u[:,:,1:n_y-1] - 0.5*Minmod(ratio_y)[:,:,1:n_y-1]*(u[:,:,2:n_y]-u[:,:,1:n_y-1])


    return u_minus_x, u_plus_x, u_minus_y, u_plus_y



# def local_lax_friedrichs(fluxes_pm,cons_pm,char_speed):

def local_lax_friedrichs(flux_plus,flux_minus,cons_plus,cons_minus,char_speed):

    """ Inputs:
        
        fluxes_pm: Reconstructed fluxes
        cons_pm: Reconstructed conservative variables
        char_speed: The maximum of the absolute value of the characteristic speeds in each cell. (Using reconstructed values)
    """

    flux_LF = 0.5*( (flux_plus + flux_minus)  - (char_speed)*(cons_plus - cons_minus)) 

    return flux_LF



##################################################
# Boundary Conditions
#################################################

def periodic_BC(u, Ngz): # Periodic BCs have to be on boths sides

    """ Inputs:
    
        u - The array to be updated
        Ngz - The number of ghost points
    """

    # X Boundaries
    u[:,:Ngz,:] = u[:,-2*Ngz:-Ngz,:]
    u[:,-Ngz:,:] = u[:,Ngz:2*Ngz,:]

    # Y Boundaries
    u[:,:, :Ngz] = u[:,:,-2*Ngz:-Ngz]
    u[:,:, -Ngz:] = u[:,:,Ngz:2*Ngz]

    return u



#########################################################
# Convert Between Conserved and Primitive Variables
#########################################################
def conserved_to_primitive(cons ,K):
        
        tau, S1, S2 = cons

        Q = ((S1**2+S2**2))/((K+1)**2*tau**2)

        Gamma2 = (1 - 2*K*(1+K)*Q + np.sqrt(1-4*K*Q))\
                /(2*(1 - (1+K)**2*Q))
        
        mu = tau/((K+1)*Gamma2 - K)

        u1 = S1/((K+1)*Gamma2*mu)

        u2 = S2/((K+1)*Gamma2*mu)
        

        return np.array([u1,u2,mu])


def primitive_to_conserved(prims,K):
        
        u1, u2, mu = prims

        Gamma2 = 1/(1-u1**2 - u2**2) 

        tau = (K+1)*Gamma2*mu - K*mu
        S1 = (K+1)*Gamma2*mu*u1
        S2 = (K+1)*Gamma2*mu*u2

        
        return np.array([tau,S1,S2])

#########################################################
# Compute Characteristic Speeds
#########################################################

def compute_char_speeds_x(prims_plus,prims_minus,t,K,p1,p2):
        global CS_x

        """ Compute characteristic speeds in the x-direction"""

        u1_plus, u2_plus, mu_plus = prims_plus
        u1_minus, u2_minus, mu_minus = prims_minus

        lam1_plus = np.exp((p1-1)*t)*u1_plus
        lam1_minus = np.exp((p1-1)*t)*u1_minus
        
        lam2_plus = np.exp((p1-1)*t)*((1-K)*u1_plus - np.sqrt(K*(1-u1_plus**2-u2_plus**2)*(1-u1_plus**2 - K*u2_plus**2)))\
                      /(1 - K*(u1_plus**2 + u2_plus**2))
        lam2_minus = np.exp((p1-1)*t)*((1-K)*u1_minus - np.sqrt(K*(1-u1_minus**2-u2_minus**2)*(1-u1_minus**2 - K*u2_minus**2)))\
                      /(1 - K*(u1_minus**2 + u2_minus**2))

        lam3_plus = np.exp((p1-1)*t)*((1-K)*u1_plus + np.sqrt(K*(1-u1_plus**2-u2_plus**2)*(1-u1_plus**2 - K*u2_plus**2)))\
                      /(1 - K*(u1_plus**2 + u2_plus**2))
        lam3_minus = np.exp((p1-1)*t)*((1-K)*u1_minus + np.sqrt(K*(1-u1_minus**2-u2_minus**2)*(1-u1_minus**2 - K*u2_minus**2)))\
                      /(1 - K*(u1_minus**2 + u2_minus**2))

        a = np.fmax.reduce(np.abs([lam1_plus,lam1_minus,\
                                   lam2_plus,lam2_minus,\
                                    lam3_plus,lam3_minus]))
        

        CS_x = np.max(a[2:-2,2:-2])

        return a


def compute_char_speeds_y(prims_plus,prims_minus,t,K,p1,p2):
        global CS_y

        """ Compute characteristic speeds in the y-direction"""

        
        u1_plus, u2_plus, mu_plus = prims_plus
        u1_minus, u2_minus, mu_minus = prims_minus

        lam1_plus = np.exp((p2-1)*t)*u2_plus
        lam1_minus = np.exp((p2-1)*t)*u2_minus
        
        lam2_plus = np.exp((p2-1)*t)*((1-K)*u2_plus - np.sqrt(K*(1-u1_plus**2-u2_plus**2)*(1-K*u1_plus**2 - u2_plus**2)))\
                      /(1 - K*(u1_plus**2 + u2_plus**2))
        lam2_minus = np.exp((p2-1)*t)*((1-K)*u2_minus - np.sqrt(K*(1-u1_minus**2-u2_minus**2)*(1-K*u1_minus**2 - u2_minus**2)))\
                      /(1 - K*(u1_minus**2 + u2_minus**2))

        lam3_plus = np.exp((p2-1)*t)*((1-K)*u2_plus - np.sqrt(K*(1-u1_plus**2-u2_plus**2)*(1-K*u1_plus**2 - u2_plus**2)))\
                      /(1 - K*(u1_plus**2 + u2_plus**2))
        lam3_minus = np.exp((p2-1)*t)*((1-K)*u2_minus - np.sqrt(K*(1-u1_minus**2-u2_minus**2)*(1-K*u1_minus**2 - u2_minus**2)))\
                      /(1 - K*(u1_minus**2 + u2_minus**2))

        a = np.fmax.reduce(np.abs([lam1_plus,lam1_minus,\
                                   lam2_plus,lam2_minus,\
                                    lam3_plus,lam3_minus]))
        
       
        CS_y = np.max(a[2:-2,2:-2])


        return a


#########################################################
# Compute Fluxes
#########################################################

def compute_flux_F1(prims_plus,prims_minus,t,p1,p2,K):

        """ Compute the fluxes in x-direction"""
        
        u1_plus, u2_plus, mu_plus = prims_plus

        Gamma2_plus = 1/(1 - u1_plus**2 - u2_plus**2)

        F1_tau_plus = np.exp((p1-1)*t)*(K+1)*Gamma2_plus*mu_plus*u1_plus
        F1_S1_plus = np.exp((p1-1)*t)*((K+1)*Gamma2_plus*mu_plus*u1_plus**2 + K*mu_plus)
        F1_S2_plus = np.exp((p1-1)*t)*(K+1)*Gamma2_plus*mu_plus*u1_plus*u2_plus

        flux_plus = np.array([F1_tau_plus,F1_S1_plus,F1_S2_plus])

        u1_minus, u2_minus, mu_minus = prims_minus

        Gamma2_minus = 1/(1 - u1_minus**2 - u2_minus**2)

        F1_tau_minus = np.exp((p1-1)*t)*(K+1)*Gamma2_minus*mu_minus*u1_minus
        F1_S1_minus = np.exp((p1-1)*t)*((K+1)*Gamma2_minus*mu_minus*u1_minus**2 + K*mu_minus)
        F1_S2_minus = np.exp((p1-1)*t)*(K+1)*Gamma2_minus*mu_minus*u1_minus*u2_minus


        flux_minus = np.array([F1_tau_minus,F1_S1_minus,F1_S2_minus])

        return flux_plus, flux_minus


def compute_flux_F2(prims_plus,prims_minus,t,p1,p2,K):
        
        """ Compute the fluxes in y-direction"""

        u1_plus, u2_plus, mu_plus = prims_plus

        Gamma2_plus = 1/(1 - u1_plus**2 - u2_plus**2)

        F1_tau_plus = np.exp((p2-1)*t)*(K+1)*Gamma2_plus*mu_plus*u2_plus
        F1_S1_plus = np.exp((p2-1)*t)*(K+1)*Gamma2_plus*mu_plus*u1_plus*u2_plus
        F1_S2_plus = np.exp((p2-1)*t)*((K+1)*Gamma2_plus*mu_plus*u2_plus**2 + K*mu_plus)

        flux_plus = np.array([F1_tau_plus,F1_S1_plus,F1_S2_plus])

        u1_minus, u2_minus, mu_minus = prims_minus

        Gamma2_minus = 1/(1 - u1_minus**2 - u2_minus**2)

        F1_tau_minus = np.exp((p2-1)*t)*(K+1)*Gamma2_minus*mu_minus*u2_minus
        F1_S1_minus = np.exp((p2-1)*t)*(K+1)*Gamma2_minus*mu_minus*u1_minus*u2_minus
        F1_S2_minus = np.exp((p2-1)*t)*((K+1)*Gamma2_minus*mu_minus*u2_minus**2 + K*mu_minus)


        flux_minus = np.array([F1_tau_minus,F1_S1_minus,F1_S2_minus])

        

        return flux_plus, flux_minus


########################################################################################################
#System of PDEs to Solve
########################################################################################################
def T1_Euler(t,cons,step_sizes,Ngz,K,p1,p2):

    # p1 = 0.5
    # p2 = 0.5

    """ 
    NB: This currently assumes a uniformly spaced grid in x and y!
    
    Inputs:
        t - The current time
        cons - The value of the conserved variables at the current time
        step_size - array[dx,dy], the spatial step size in each direction 
        Ngz - Number of ghost points
        K - Fluid sound speed squared
        p1, p2 - Kasner Exponents"""


    # -------------------------------------------
    # Recover the primitive variables
    # -------------------------------------------
    prims =  conserved_to_primitive(cons, K)

    u1, u2, mu = prims 

    # -------------------------------------------
    # Reconstruct the primitives at cell edges
    # -------------------------------------------

    prims_minus_x, prims_plus_x, prims_minus_y, prims_plus_y = linear_reconstruct(prims)
    
    
    # Compute reconstructed conserved variables

    cons_minus_x = primitive_to_conserved(prims_minus_x,K)
    cons_plus_x = primitive_to_conserved(prims_plus_x,K)
    cons_minus_y = primitive_to_conserved(prims_minus_y,K)
    cons_plus_y = primitive_to_conserved(prims_plus_y,K)


    # -------------------------------------------
    # Compute the Charateristic Speeds
    # -------------------------------------------

    char_speed_x = compute_char_speeds_x(prims_plus_x,prims_minus_x,t,K,p1,p2)

    char_speed_y = compute_char_speeds_y(prims_plus_y,prims_minus_y,t,K,p1,p2)


    # -------------------------------------------
    # Construct the Fluxes
    # -------------------------------------------

    flux_plus_F1, flux_minus_F1 = compute_flux_F1(prims_plus_x,prims_minus_x,t,p1,p2,K)

    flux_plus_F2, flux_minus_F2 = compute_flux_F2(prims_plus_y,prims_minus_y,t,p1,p2,K)


    flux_F1 = local_lax_friedrichs(flux_plus_F1,flux_minus_F1,cons_plus_x,cons_minus_x,char_speed_x)

    flux_F2 = local_lax_friedrichs(flux_plus_F2,flux_minus_F2,cons_plus_y,cons_minus_y,char_speed_y)


    # -------------------------------------------
    # Construct the Source Terms
    # -------------------------------------------

    tau_source = -(((-1 + K + (K*(-2 + p1) + p1)*u1**2 + (K*(-2 + p2) + p2)*u2**2)*mu)/\
                   (-1 + u1**2 + u2**2))

    S1_source = -(((1 + K)*(-1 + p1)*u1*mu)/(-1 + u1**2 + u2**2))

    S2_source = -(((1 + K)*(-1 + p2)*u2*mu)/(-1 + u1**2 + u2**2))


    # -------------------------------------------
    # Update the Evolution Equations
    # -------------------------------------------

    dttau = np.zeros_like(u1)
    dtS1 = np.zeros_like(u1)
    dtS2 = np.zeros_like(u1)

    dx, dy = step_sizes

    dttau[Ngz:-Ngz,Ngz:-Ngz] = -1/(dx)*(flux_F1[0,Ngz:-Ngz,Ngz:-Ngz] - flux_F1[0,Ngz-1:-Ngz-1,Ngz:-Ngz])\
                                -1/(dy)*(flux_F2[0,Ngz:-Ngz,Ngz:-Ngz] - flux_F2[0,Ngz:-Ngz,Ngz-1:-Ngz-1])\
                                + tau_source[Ngz:-Ngz,Ngz:-Ngz]
    
    dtS1[Ngz:-Ngz,Ngz:-Ngz] = -1/(dx)*(flux_F1[1,Ngz:-Ngz,Ngz:-Ngz] - flux_F1[1,Ngz-1:-Ngz-1,Ngz:-Ngz])\
                                -1/(dy)*(flux_F2[1,Ngz:-Ngz,Ngz:-Ngz] - flux_F2[1,Ngz:-Ngz,Ngz-1:-Ngz-1])\
                                + S1_source[Ngz:-Ngz,Ngz:-Ngz]

    dtS2[Ngz:-Ngz,Ngz:-Ngz] = -1/(dx)*(flux_F1[2,Ngz:-Ngz,Ngz:-Ngz] - flux_F1[2,Ngz-1:-Ngz-1,Ngz:-Ngz])\
                                -1/(dy)*(flux_F2[2,Ngz:-Ngz,Ngz:-Ngz] - flux_F2[2,Ngz:-Ngz,Ngz-1:-Ngz-1])\
                                + S2_source[Ngz:-Ngz,Ngz:-Ngz]


        
    dtU = np.array([dttau,dtS1,dtS2])

    
    return dtU


###########################################################################
# Evolution Routines
##########################################################################

def rk4(t0,y0,dt,rhs,BCs,Nghosts): # "Classic" RK4

    k1 = rhs(t0,y0)

    y1 = y0 + 0.5*dt*k1
    y1 = BCs(y1,Nghosts) ### Updates ghost points using physical boundaries


    k2 = rhs(t0 + 0.5*dt, y1)

    y2 = y0 + 0.5*dt*k2
    y2 = BCs(y2,Nghosts)


    k3 = rhs(t0 + 0.5*dt, y2)
    y3 = y0 + dt*k3
    y3 = BCs(y3,Nghosts) 

    k4 = rhs(t0 + dt, y3)

    y0 = y0 + dt*(1/6*k1 + 1/3*k2 +1/3*k3 +1/6*k4)
    y0 = BCs(y0,Nghosts) 


    return y0

def rk3(t0,y0,dt,rhs,BCs,Nghosts): # Shu-Osher SSP RK3
     
    k1 = rhs(t0, y0)
    y1 = y0 + dt*k1
    y1 = BCs(y1,Nghosts) 

    k2 = rhs(t0 + dt, y1)
    y2 = y0 + 0.25*dt*(k1 + k2)
    y2 = BCs(y2,Nghosts) 

    k3 = rhs(t0 + 0.5*dt, y2)

    y0 = y0 + dt*(1/6*k1 + 1/6*k2 + 2/3*k3)
    y0 = BCs(y0,Nghosts) 

    return y0



def evolve_system(yinit,t0,tend,spatial_steps,rk,rhs,BCs,K,Nghosts):

    global CS_x, CS_y

    y0 = yinit
    y_store = []
    y_store.append(yinit)
    t_store = []
    t_store.append(t0)
    i = 0

    dx, dy = spatial_steps
    CFL = 0.2
    while t0<tend:
        
        i += 1

        dt = CFL*np.min([dx/CS_x,dy/CS_y])
        if dt > 0.4*dx:
            dt = 0.4*dx

        if t0 + dt > tend:
             dt = tend-t0
             print("Simulation ending, adjusting timestep to dt = " +str(dt))

        y0 = rk(t0,y0,dt,rhs,BCs,Nghosts)
        t0 += dt

        # Store solution every i timesteps
        if i == 1:
            t_store.append(t0)
            y_store.append(y0)
            i = 0

    return t_store, y_store


    
############################################################################
# Create Grid
############################################################################
Npoints_x = args.Nx 
Npoints_y = args.Ny 
Npoints = np.array([Npoints_x,Npoints_y]) 
Nghosts = 2 # Number of ghost points

# Set grid domain
interval_x = (0,2*np.pi) 
interval_y = (0,2*np.pi)

dx = (interval_x[1]-interval_x[0])/Npoints_x
dy = (interval_y[1]-interval_y[0])/Npoints_y
spatial_steps = np.array([dx,dy])


# Create cell arrays
x_start = interval_x[0] + (0.5 - Nghosts)*dx
x_end = interval_x[1] + (Nghosts - 0.5)*dx

y_start = interval_y[0] + (0.5 - Nghosts)*dy
y_end = interval_y[1] + (Nghosts - 0.5)*dy

coordinates_x = np.linspace(x_start, x_end, Npoints_x + 2*Nghosts)
coordinates_y = np.linspace(y_start, y_end, Npoints_y + 2*Nghosts)

# NB: These meshes include the ghost points
x_cell_mesh, y_cell_mesh = np.meshgrid(coordinates_x,coordinates_y,indexing='ij')


############################################################################
# Set Initial Data + Equation Parameters
############################################################################

# Sound speed and Kasner exponents

K = args.K 
p1 = 0.2
p2 = 0.2

# Set Primitive Variables

mu = 0*x_cell_mesh + 1.0 
u1 = 0.4*np.sin(x_cell_mesh) + 0.4*np.sin(y_cell_mesh)
u2 = 0.5*np.sin(y_cell_mesh) + 0.2*np.cos(x_cell_mesh)

Gamma2 = 1/(1 - u1**2 - u2**2) # Gamma2 := Gamma^{2}


# print(u1**2 + u2**2)
# breaks

# Define Conserved Variables:

tau = (K+1)*Gamma2*mu - K*mu
S1 = (K+1)*Gamma2*mu*u1
S2 = (K+1)*Gamma2*mu*u2


y0 = np.array([tau, S1, S2])

y0 = periodic_BC(y0,Nghosts) # Enforce BCs initially





#########################################################################
# Simulation Parameters
########################################################################

t_init = 0.0
t_final = 10.0

# Initial Max Characteristic Speeds:
lam1_x = np.exp((p1-1)*t_init)*u1
        
lam2_x = np.exp((p1-1)*t_init)*((-1+K)*u1 + np.sqrt(K*(1-u1**2-u2**2)*(1-u1**2 - K*u2**2)))\
                      /(-1 + K*(u1**2 + u2**2))
lam3_x = np.exp((p1-1)*t_init)*((-1+K)*u1 - np.sqrt(K*(1-u1**2-u2**2)*(1-u1**2 - K*u2**2)))\
                      /(-1 + K*(u1**2 + u2**2))

CS_x = np.max(np.array([lam1_x,lam2_x,lam3_x]))


lam1_y = np.exp((p2-1)*t_init)*u2
        
lam2_y = np.exp((p2-1)*t_init)*((-1+K)*u2 + np.sqrt(K*(1-u1**2-u2**2)*(1-K*u1**2 - u2**2)))\
                      /(-1 + K*(u1**2 + u2**2))
lam3_y = np.exp((p2-1)*t_init)*((-1+K)*u2 - np.sqrt(K*(1-u1**2-u2**2)*(1-K*u1**2 - u2**2)))\
                      /(-1 + K*(u1**2 + u2**2))

CS_y = np.max(np.array([lam1_y,lam2_y,lam3_y]))


############################################################################
# Evolve System
############################################################################
rhs = lambda t,y: T1_Euler(t,y,spatial_steps,Nghosts,K,p1,p2)

t_soln, y_soln = evolve_system(y0,t_init,t_final,spatial_steps,rk4,rhs,periodic_BC,K,Nghosts)
t = np.array(t_soln)
y = np.array(y_soln)

# [t,y] = mixed_rk4(rhs, 20, 10, y0, 0.5*(dx),dx,spherical_symmetry_BC)


# [t,y] = rk4_old(rhs, np.logspace(np.log10(1),-15,600), y0, 0.5*(dx),spherical_symmetry_BC)



t_end = time.time()
print("Elapsed time is",t_end-t_start) #Prints simulation length of time 


############################################################################
# Recover Primtive Variables
############################################################################
tau = y[:,0,2:-2,2:-2]
S1 = y[:,1,2:-2,:2-2]
S2 = y[:,1,2:-2,:2-2]

mu = np.zeros_like(y[:,0,:,:])
u1 = np.zeros_like(y[:,0,:,:])
u2 = np.zeros_like(y[:,0,:,:])


for i in range(np.shape(t)[0]):
    u1[i,:,:], u2[i,:,:], mu[i,:,:] = conserved_to_primitive(y[i,:,:,:],K)


####################################
#Create HDF File
####################################
if args.f is not None:
    
    h5f = h5py.File(\
        #"/Users/elliotmarshall/Documents/MATLAB/HDF_Euler_Conformal/"\
        "/Users/elliotmarshall/Desktop/AF_Euler/UltraRelativistic_Files/"\
        +args.f, 'w')
    #y_hdf = np.swapaxes(y,1,2) #Have to swap axes for HDF order
    h5f.create_dataset('Solution Matrices', data=y,compression="gzip")#_hdf.T)
    h5f.create_dataset('Time Vector', data=t)
    h5f.close()


############################
#Plot with MatPlotLib
############################

if display_output is True:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for i in range(np.shape(t)[0]):

    
        CS = ax.plot_surface(x_cell_mesh,y_cell_mesh,u1[i,:,:]**2 + u2[i,:,:]**2, cmap=cm.coolwarm)
        # plt.colorbar()
        plt.draw()
        plt.title(t_soln[i])
        plt.pause(0.01)
        ax.cla()




            
            



