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
from matplotlib.animation import FuncAnimation
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
    n_x = np.shape(u)[0]
    u_plus_y = np.zeros_like(u) 
    u_minus_y = np.zeros_like(u) 
    ratio_y = np.zeros_like(u)
    n_y = np.shape(u)[1]

    

    ratio_x[1:n_x-1,:] = (u[1:n_x-1,:]-u[0:n_x-2,:])/(u[2:n_x,:]-u[1:n_x-1,:]+1e-16)
    
    ### Compute the slope-limited linear reconstruction:

    u_minus_x[0:n_x-1,:] = u[0:n_x-1,:] + 0.5*Minmod(ratio_x)[0:n_x-1,:]*(u[1:n_x,:]-u[0:n_x-1,:])

    u_plus_x[0:n_x-2,:] = u[1:n_x-1,:] - 0.5*Minmod(ratio_x)[1:n_x-1,:]*(u[2:n_x,:]-u[1:n_x-1,:])


    ratio_y[:,1:n_y-1] = (u[:,1:n_y-1]-u[:,0:n_y-2])/(u[:,2:n_y]-u[:,1:n_y-1]+1e-16)

    
    # ### Compute the slope-limited linear reconstruction:
    u_minus_y[:,0:n_y-1] = u[:,0:n_y-1] + 0.5*Minmod(ratio_y)[:,0:n_y-1]*(u[:,1:n_y]-u[:,0:n_y-1])

    u_plus_y[:,0:n_y-2] = u[:,1:n_y-1] - 0.5*Minmod(ratio_y)[:,1:n_y-1]*(u[:,2:n_y]-u[:,1:n_y-1])


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


########################################################################################################
#System of PDEs to Solve
########################################################################################################
def Burgers(t,cons,step_sizes,Ngz):

    global CS_x, CS_y

    u = cons[0,:,:]


    u_minus_x, u_plus_x, u_minus_y, u_plus_y = linear_reconstruct(u)
    

    char_speed_x = np.fmax.reduce(np.abs([2*u_plus_x,2*u_minus_x]))

    char_speed_y = np.fmax.reduce(np.abs([2*u_plus_y,2*u_minus_y]))

    CS_x = np.max(np.abs(char_speed_x))
    CS_y= np.max(np.abs(char_speed_y))


    # -------------------------------------------
    # Construct the Fluxes
    # -------------------------------------------

    flux_F1 = local_lax_friedrichs(u_plus_x**2,u_minus_x**2,u_plus_x,u_minus_x,char_speed_x)

    flux_F2 = local_lax_friedrichs(u_plus_y**2,u_minus_y**2,u_plus_y,u_minus_y,char_speed_y)

    # -------------------------------------------
    # Update the Evolution Equations
    # -------------------------------------------

    dtu= np.zeros_like(u)

    dx, dy = step_sizes

    dtu[Ngz:-Ngz,Ngz:-Ngz] = -1/(dx)*(flux_F1[Ngz:-Ngz,Ngz:-Ngz] - flux_F1[Ngz-1:-Ngz-1,Ngz:-Ngz])\
                                -1/(dy)*(flux_F2[Ngz:-Ngz,Ngz:-Ngz] - flux_F2[Ngz:-Ngz,Ngz-1:-Ngz-1])

    
    dtU = np.array([dtu])

    
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



def evolve_system(yinit,t0,tend,spatial_steps,rk,rhs,BCs,Nghosts):

    global CS_x, CS_y

    y0 = yinit
    y_store = []
    y_store.append(yinit)
    t_store = []
    t_store.append(t0)
    i = 0

    dx, dy = spatial_steps
    CFL = 0.4
    dt = 0.4*dx
    while t0<tend:
        
        i += 1

        dt = CFL*np.min([dx/CS_x,dy/CS_y])
        # dt = CFL*np.min([dx,dy])/(np.max([CS_x,CS_y]))
        # dt = CFL*dx
        # print(dt)
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

if dx != dy:
     print("Must have uniform spacing in x and y directions!")
     print("Closing simulation...")
     sys.exit()


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


# Set Primitive Variables

u = np.sin(x_cell_mesh) + np.sin(y_cell_mesh)

y0 = np.array([u])

y0 = periodic_BC(y0,Nghosts) # Enforce BCs initially


#########################################################################
# Simulation Parameters
########################################################################

t_init = 0
t_final = 5.0

# Initial Max Characteristic Speeds:
CS_x = np.max(u)
CS_y = np.max(u)


############################################################################
# Evolve System
############################################################################
rhs = lambda t,y: Burgers(t,y,spatial_steps,Nghosts)

t_soln, y_soln = evolve_system(y0,t_init,t_final,spatial_steps,rk4,rhs,periodic_BC,Nghosts)
t = np.array(t_soln)
y = np.array(y_soln)

# [t,y] = mixed_rk4(rhs, 20, 10, y0, 0.5*(dx),dx,spherical_symmetry_BC)


# [t,y] = rk4_old(rhs, np.logspace(np.log10(1),-15,600), y0, 0.5*(dx),spherical_symmetry_BC)



t_end = time.time()
print("Elapsed time is",t_end-t_start) #Prints simulation length of time 


############################################################################
# Recover Primtive Variables
############################################################################
u = y[:,0,2:-2,2:-2]
x_plot = x_cell_mesh[2:-2,2:-2]
y_plot = y_cell_mesh[2:-2,2:-2]


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

    
        CS = ax.plot_surface(x_cell_mesh[2:-2,2:-2],y_cell_mesh[2:-2,2:-2],u[i,:,:])
        # CS = plt.contourf(r.meshes[0][:,:],r.meshes[1][:,:],v_norm)
        # ax.plot_surface(r1.meshes[0][:,:],r1.meshes[1][:,:],rho[:,:])
        # plt.plot(DG_1[:,0,0])#, '.', markersize = 1, linestyle = 'None')
        # plt.colorbar()
        plt.draw()
        plt.title(t_soln[i])
        plt.pause(0.01)
        ax.cla()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Create the initial surface
# surf = [ax.plot_surface(x_plot, y_plot, u[0], cmap=cm.viridis)]

# surf = ax.plot_surface(
#     x_plot, y_plot, u[0],
#     cmap=cm.viridis,
#     linewidth=0,      # remove gridlines on surface
#     antialiased=True, # smoother edges
#     rstride=1, cstride=1,  # finer resolution
#     edgecolor='none'
# )

# def update(frame):
#     ax.clear()
#     ax.set_title(f"Time = {t[frame]:.2f}")
#     surf[0] = ax.plot_surface(x_plot, y_plot, u[frame], cmap=cm.viridis)
#     return surf

# anim = FuncAnimation(fig, update, frames=len(t), interval=50)
# plt.show()




            
            



