# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:51:03 2020

@author: joher
"""

# Imports libraries
import numpy as np
import math as math
import statistics as stat
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.linalg
import pandas as pd
import scipy.stats
import time
from decimal import *


"""---------------------------------------- Initialising graphic parameters."""

#   Parameters for printing graphs.
scale = 2.5

latex2col_width = (252.0 / 72.27) * scale
latex2col_height = latex2col_width * ( (math.sqrt(5)-1.0)/2.0 ) * 1.2

params = {
    'backend': 'ps',
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'axes.labelsize':   8*scale, # fontsize for x and y labels (was 10)
    'axes.titlesize':   8*scale,
    'font.size':        8*scale, # was 10
    'legend.fontsize':  8*scale, # was 10
    'xtick.labelsize':  8*scale,
    'ytick.labelsize':  8*scale,
    'text.usetex':      True,
    #'figure.width':     latex2col_width,
    'figure.figsize': [latex2col_width,latex2col_height],
    'font.family': 'serif',
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
}

mpl.rcParams.update(params)


"""------------------------------------------------------ Graphic functions."""

# Formats query into labeled 2D array.
def query_to_array( query, label, size ):
    '''
    :params query: Query of the coordinates for each individual turn. 
        Taken from a data frame.
    :params label: Turn of the coordinates.
    :params size: Number of coordinates per turn.
    '''
    search = query.values.tolist()
    search = np.concatenate(search).ravel()
    search = np.reshape(search,(-1,2))
    search = np.hstack((
        search,
        np.array( [label]*size ).reshape( (-1,1) )
        ))

    return search

# Function that graphs the scatter plot of a set of points (states of the game,
# actions of the players), labeled with the respective turn.
def complete_scatterplot( labeled_coord, reverse, file_name = False  ):
    '''
    :params labeled_coord: 2D array. Each row represents an xy coordinate with 
        its corresponding turn. The first and second columns are the x and y
        coordinates, respectively, and the last one is the turn.
    :params reverse: Boolean. Overlay in reverse order.
    :params file_name: String (optional). Saves figure with the given name.
        Otherwise, no figure is saved.
    '''
    # Components.
    x1=(not reverse)*labeled_coord[:,0] + (reverse)*labeled_coord[::-1,0]
    x2=(not reverse)*labeled_coord[:,1] + (reverse)*labeled_coord[::-1,1]
    
    # Label.
    la=(not reverse)*labeled_coord[:,2] + (reverse)*labeled_coord[::-1,2]
    
    plt.figure()
    plt.scatter(
        x=x1,
        y=x2,
        c=la,
        s=0.5,
        #alpha=0.35,
        cmap=mpl.cm.jet
        )
    clb = plt.colorbar( ticks = [ i*3 for i in range(9) ] )
    clb.ax.set_ylabel(
        'Turn', 
        rotation = 270,
        labelpad = 20)
    clb.ax.invert_yaxis()
    
    if isinstance(file_name,str):
        plt.savefig(file_name)
    
    plt.show()
    
    return True

# Function that graphs the individual scatter plot of a set of labeled points 
# (states of the game, actions of the players), only in predefined turns.
def individual_scatterplots( labeled_coord, size, file_name = False ):
    '''
    :params labeled_coord: 2D array. Each row represents an xy coordinate with 
        its corresponding turn. The first and second columns are the x and y
        coordinates, respectively, and the last one is the turn.
    :params size: Number of coordinates per turn.
    :params file_name: String (optional). Saves figure with the given name.
        Otherwise, no figure is saved.
    '''
    
    fig, ax = plt.subplots()
    
    plt.subplot( 3, 3, 1 )
    im = plt.scatter(
        x=labeled_coord[ 0, 0],
        y=labeled_coord[ 0, 1],
        c=labeled_coord[ 0, 2],
        s=0.5,
        #alpha=0.35,
        cmap=mpl.cm.jet
        )
    plt.xlim( min( labeled_coord[:,0] ), max( labeled_coord[:,0] ) )
    plt.ylim( min( labeled_coord[:,1] ), max( labeled_coord[:,1] ) )
    
    for i in range(1,9):
        plt.subplot( 3, 3, i+1 )
        im = plt.scatter(
            x=labeled_coord[ np.r_[0,(3*i-1)*size + 1: (3*i)*size + 1,0], 0],
            y=labeled_coord[ np.r_[0,(3*i-1)*size + 1: (3*i)*size + 1,0], 1],
            c=labeled_coord[ np.r_[-1,(3*i-1)*size + 1: (3*i)*size + 1,0], 2],
            s=0.5,
            #alpha=0.35,
            cmap=mpl.cm.jet
            )
        #subtitle = 'Turn %d'%(i*3) 
        #plt.title(subtitle)
        plt.xlim( min( labeled_coord[:,0] ), max( labeled_coord[:,0] ) )
        plt.ylim( min( labeled_coord[:,1] ), max( labeled_coord[:,1] ) )
    
    fig.tight_layout()
    fig.subplots_adjust(right=0.825)
    cax = fig.add_axes([0.85, 0.06, 0.035, 0.91])
    clb = fig.colorbar(
        im, 
        cax=cax,
        ticks = [ i*3 for i in range(9) ] ) 
    clb.ax.set_ylabel(
        'Turn', 
        rotation = 270,
        labelpad = 20
        )
    clb.ax.invert_yaxis()
    
    if isinstance(file_name,str):
        plt.savefig(file_name)
    
    plt.show()

    return True

"""------------------------------------- Initialising experiment parameters."""

#   Initialize seed.
np.random.seed(20200803)

""" Dimensions of the parameters.
 d : X, A, Q
 m : U, B, R
 n : V, D, T
"""
d = 2
m = d
n = d

#   Turns: trimesters over 6 years + initial state
N = (4*6) + 1 

#   Identity matrix.
Id = np.identity(d)

#   Dynamic's coefficients.
A = np.mat(np.diag(np.array([1.07, 1.03])))
B = np.mat(np.diag(np.array([0.02, 0.80])))
D = np.mat(np.diag(np.array([0.05, 0.75])))

#   Payoff's coefficients.
Q = np.mat( [[1.02,0.10],[0.10,1.00]] ) # np.mat(np.diag(np.array([1.02, 1.00])))
R = np.mat(np.diag(np.array([0.03, 0.50]))) # [0.00, 0.20]
T = np.mat(np.diag(np.array([1.50, 1.80]))) # [2.45, 2.90]

""" Noise's coefficients.
Notes: 
    1. Replace values with commented code in order to obtain Simulation 4.
    2. Multiply M * (-1) to obtain Simulation 5.
"""
M = Id # np.mat( [[2, 0.5], [-1.3, 0.8]] )
Sig = Id # np.mat( [[.5, 0.01], [0.01, .5]] )

""" Obtains adjusted coeffients.
Note: The indexes of the variables in the article and the program are out of
sync by one:
    Article            |   Program
    -------------------|--------------
    S_{k+1}            |   S[k]
    \tilde{B}_{k+1}    |   Btilde[k]
    \tilde{D}_{k+1}    |   Dtilde[k]
"""
S = []
Btilde = []
Dtilde = []

# Adjusted coefficients.
for k in reversed( range(1,N+1) ):

    if k == N:
        S_k = Q

    else:
        S_k = (
            Q +
            ( np.transpose( A ) * S[0] * A ) -
            (
                ( np.transpose(A) * S[0] * B ) *
                np.linalg.inv(S[0]) *
                np.transpose( np.transpose(A) * S[0] * B )
            )
        )

    S.insert(0,S_k)

    BSB = np.transpose( B ) * S_k * B
    DSD = np.transpose( D ) * S_k * D

    Btilde.insert(0,BSB + R)
    Dtilde.insert(0,DSD - T)

# Uncomment lines to review adjusted coefficients. 

print("Adjusted coefficients ------------------------")
print("Checking positive definiteness by turn.")
print("\nTurn \t| Btilde \t| -Dtilde")
for k in range(N):
    
    print(k,
          "\t\t|",
          np.all( scipy.linalg.eigvals( Btilde[k] ) > 0 ),
          "\t\t|",
          np.all( scipy.linalg.eigvals( -Dtilde[k] ) > 0 )
         )
    
    
# Coefficients within {F_k} for Y_k.
CoFY =[]
for k in range(N):
    # Fixed k-th F_k:
    suma = np.mat(np.diag(np.array([0.00, 0.00])))

    for j in range(k,N):

        # Within j-th sum term
        if j==k:
            CoFY_k = S[j] * M

        else:
            prod = Id
            for i in range(k+1,j):
                factor = (
                    np.transpose(A) * (
                        Id - ( S[i] * B * np.linalg.inv( Btilde[i] ) * np.transpose( B ) )
                    )
                )
                prod_anterior = prod
                prod = prod_anterior * factor
            CoFY_k = prod * S[j] * np.linalg.matrix_power(M, j + 1 - k )

        suma_anterior = suma
        suma = suma_anterior + CoFY_k

    CoFY.append(suma)
    #print(CoFY[k])

# Inverse matrix for controls.
h12=[]
h21=[]
for k in range(N):
    h12_k = Btilde[k] - (
        ( np.transpose( B ) * S[k] * D ) *
        np.linalg.inv( Dtilde[k] ) *
        np.transpose( np.transpose( B ) * S[k] * D )
    )
    h21_k = Dtilde[k] - (
            np.transpose( np.transpose(B) * S[k] * D ) *
            np.linalg.inv( Btilde[k] ) *
            ( np.transpose(B) * S[k] * D )
    )

    h12.append( h12_k )
    h21.append( h21_k )

# Controls' coeficients.
b0=[]
b1=[]
d0=[]
d1=[]
for k in range(N):
    # In disadvantage.
    b0_k = np.linalg.inv( h12[k] ) * np.transpose( B ) * (
        ( S[k] * D * np.linalg.inv( Dtilde[k] ) * np.transpose( D ) ) - Id
    )
    d1_k = np.linalg.inv( h21[k] ) * np.transpose( D ) * (
        ( S[k] * B * np.linalg.inv( Btilde[k] ) * np.transpose( B ) ) - Id
    )

    # In advantage.
    b1_k = (-1) * np.linalg.inv( Btilde[k] ) * np.transpose( B )
    d0_k = (-1) * np.linalg.inv( Dtilde[k] ) * np.transpose( D )

    b0.append( b0_k )
    b1.append( b1_k )
    d0.append( d0_k )
    d1.append( d1_k )

"""------------------------------------------------------------ Simulations.
Two games are played at the same time. One with advantage probability
p, and one with q=1-p. 
The corresponding values are differenced by prefix p_ and q_, respectively.
"""

# Repetitions.
NRep = 10000

# List of Payoffs.
p_VecJ = []
q_VecJ = []

# List of Probabilities of advantage.
p_VecP = [ [0]*(N) for c in range(NRep) ]
q_VecP = [ [0]*(N) for c in range(NRep) ]

# Components
#VecXN0 = []
#VecXN1 = []

# Matrix of simulated System's states.
p_MatX = [ [0]*(N+1) for c in range(NRep) ]
q_MatX = [ [0]*(N+1) for c in range(NRep) ]

# Matrix of simulated Controls (p game)
p_MatU = [ [0]*(N) for c in range(NRep) ]
p_MatV = [ [0]*(N) for c in range(NRep) ]

# Matrix of simulated Controls (q game)
q_MatU = [ [0]*(N) for c in range(NRep) ]
q_MatV = [ [0]*(N) for c in range(NRep) ]

# Simulations of the game.
for c in range(NRep):
    # Uncomment to print current iteration.
    #print("****************************************")
    #print("Simulation ",c)

    # Lists for System and Noise (p game).
    p_X=[]
    p_Y=[]
    
    # Lists for System and Noise (q game).
    q_X=[]
    q_Y=[]

    """ Initial Condition.
    Replace with commented code to obtain Simulation 2.
    """
    X_0 = (0.80, 1.20) # np.random.multivariate_normal((0, 0), Id)
    p_X.append( np.transpose( np.mat(X_0) ) )
    q_X.append( np.transpose( np.mat(X_0) ) )

    # Initialize cost.
    p_J = 0
    q_J = 0

    # Noise (p game): Hidden Markov.
    p_Z = np.random.multivariate_normal((0, 0), Id, N-1)
    
    # Noise (q game): Hidden Markov.
    q_Z = np.random.multivariate_normal((0, 0), Id, N-1)

    # Evolution of the game.
    for k in range(N):

        # Uncomment to check the state of the System in each turn.
        #print("--------------------------------")
        #print("Turn ", k)
        #print("X[",k,"]:",X[k])
        
        # Current state (p game).
        p_MatX[c][k] = p_X[k]
        
        # Current state (q game).
        q_MatX[c][k] = q_X[k]
        
        # Controls (p game)
        p_U_k = np.transpose( np.mat( [0,0] ) )
        p_V_k = np.transpose( np.mat( [0,0] ) )
        
        # Controls (q game)
        q_U_k = np.transpose( np.mat( [0,0] ) )
        q_V_k = np.transpose( np.mat( [0,0] ) )

        """ Turn's Noise and factor F_k.
        Uncomment to check the values of F_k on each turn, as well as the
        Noise's one.
        """
        # p game.
        p_F_k = ( S[k] * A * p_X[k] )
        
        # q game.
        q_F_k = ( S[k] * A * q_X[k] )
        
        #print("F_k without noise:",p_F_k)
        
        # Addition of noise.
        if k == 0:
            p_Y_k = np.transpose(np.mat((0, 0)))
            q_Y_k = np.transpose(np.mat((0, 0)))
        else:
            # p game
            p_Y_k = (M * p_Y[k - 1]) + (Sig * np.transpose(np.mat(p_Z[k - 1])))
            p_F_k_previous = p_F_k
            p_F_k = p_F_k_previous + ( CoFY[k] * p_Y[k-1] )
            
            # q game
            q_Y_k = (M * q_Y[k - 1]) + (Sig * np.transpose(np.mat(q_Z[k - 1])))
            q_F_k_previous = q_F_k
            q_F_k = q_F_k_previous + ( CoFY[k] * q_Y[k-1] )
        
        p_Y.append(p_Y_k)
        q_Y.append(q_Y_k)


        """ Coin toss: p benefits U, 1-p benefits V"""
        # p game.
        p = - math.expm1( (-0.5) * math.sqrt(
            float( np.transpose( p_X[k] ) * Q * p_X[k] )
        ) )
        #p = 1 - p
        p_unif = np.random.random_sample()
        p_VecP[c][k] = p
        
        
        # q game.
        q = - math.expm1( (-0.5) * math.sqrt(
            float( np.transpose( q_X[k] ) * Q * q_X[k] )
        ) )
        q = 1 - q
        q_unif = np.random.random_sample()
        q_VecP[c][k] = q
        
        """ Choosing controls (p game). """
        # xi = 1
        if p_unif <= p:
            p_V_k = d1[k] * p_F_k
            p_U_k = b1[k] * ( p_F_k + ( S[k] * D * p_V_k ) )

        # xi = 0
        else:
            p_U_k = b0[k] * p_F_k
            p_V_k = d0[k] * ( p_F_k + ( S[k] * B * p_U_k ) )

        p_MatU[c][k] = p_U_k
        p_MatV[c][k] = p_V_k

        """ Choosing controls (q game) U/V sub optimal, higher/lower value. """
        # xi = 1
        if q_unif <= q:
            q_V_k = d1[k] * q_F_k
            q_U_k = b1[k] * ( q_F_k + ( S[k] * D * q_V_k ) )

        # xi = 0
        else:
            q_U_k = b0[k] * q_F_k
            q_V_k = d0[k] * ( q_F_k + ( S[k] * B * q_U_k ) )

        q_MatU[c][k] = q_U_k
        q_MatV[c][k] = q_V_k

        """Running cost (p game)."""
        p_J_k = float(
            ( np.transpose( p_X[k] ) * Q * p_X[k] ) +
            ( np.transpose( p_U_k ) * R * p_U_k ) -
            ( np.transpose( p_V_k ) * T * p_V_k )
        )

        # Contribution to the Payoff.
        p_J += p_J_k

        # New state of the System.
        p_X_k = (A * p_X[k]) + (B * p_U_k) + (D * p_V_k) + p_Y[k]
        p_X.append(p_X_k)
        
        """Running cost (q game)."""
        q_J_k = float(
            ( np.transpose( q_X[k] ) * Q * q_X[k] ) +
            ( np.transpose( q_U_k ) * R * q_U_k ) -
            ( np.transpose( q_V_k ) * T * q_V_k )
        )

        # Contribution to the Payoff.
        q_J += q_J_k
        
        # New state of the System.
        q_X_k = (A * q_X[k]) + (B * q_U_k) + (D * q_V_k) + q_Y[k]
        q_X.append(q_X_k)

    # Final cost.
    p_J += float( np.transpose( p_X[N] ) * Q * p_X[N] )
    q_J += float( np.transpose( q_X[N] ) * Q * q_X[N] )

    # Save of the final state and total cost.
    p_MatX[c][N] = p_X[N]
    p_VecJ.append( p_J )
    
    q_MatX[c][N] = q_X[N]
    q_VecJ.append( q_J )
    
    #VecXN0.append( float(X[N][0]) )
    #VecXN1.append( float(X[N][1]) )

    """ 
    Uncomment to check the final state of the simulation, the last runing
    cost, and the cost of the simulated game.
    """
    #print("--------------------------------")
    #print("Final state of the System:")
    #print("X[",N,"]:",X[N])
    #print("Final running cost:", float( np.transpose( X[N] ) * Q * X[N] ))
    #print("Cost of the Game: ",VecJ[c])
    
    # Reset of lists for the next iteration.
    #print("--------------------")
    p_X[:] = []
    q_X[:] = []
    p_Y[:] = []
    q_Y[:] = []

"""---------------------------------------------------------------- Results."""


print("****************************************")

print("Number of repetitions: ",NRep)

print("----------------------------------------")

# Payoffs.
p_Val_J = stat.mean(p_VecJ)
q_Val_J = stat.mean(q_VecJ)

# Deviation
p_Dev_J = stat.stdev(p_VecJ)
q_Dev_J = stat.stdev(q_VecJ)

print("\nPayoff \t\t| Optimal game \t| Sub-optimal game")
print("E[J] \t\t|",
      round(p_Val_J,2),
      "\t\t|",
      round(q_Val_J,2))
print("Deviation \t|",
      round(p_Dev_J,2),
      "\t\t|",
      round(q_Dev_J,2))



"""----------------------------------------------------- Payoff's histogram."""

print("----------------------------------------")
print("Creating Histograms.")

title = "Game's Payoff"
plt.figure()
# Histograms.
plt.hist(p_VecJ,
         alpha = 0.35,
         label= r'$p(X_k)$',
         bins = 100,
         color="red"
        )
plt.hist(q_VecJ,
         alpha = 0.35,
         label= r'$q(X_k)$', #'Sub-optimal',
         bins = 100,
         color= "green",#"blue"
        )

# Lines of deviation.
for cc in range(3):
    p_dev_hist =  p_Val_J + (cc * p_Dev_J)
    q_dev_hist = q_Val_J + (cc * q_Dev_J)
    linsty = (cc != 0)*'dotted' + ( cc == 0 )*'dashed'
    plt.axvline(p_dev_hist,
                color="red",
                linestyle=linsty
                )
    plt.axvline(q_dev_hist,
                color= "green",#"blue",
                linestyle=linsty
                )

plt.xlim( 0, max( p_Val_J + (3 * p_Dev_J), p_Val_J + (3 * q_Dev_J)) )

#plt.title(title)
plt.legend(loc="best")
#plt.savefig('J_q.png')
plt.show()


"""------------------------------------------ Testing for diffeence in mean."""



# Use 'greater' when V is suboptimal, and 'less' when U is suboptimal.
test_tail = 'grater'
mann_stat, mann_pval = scipy.stats.mannwhitneyu(p_VecJ,q_VecJ)#, alternative=test_tail)
wilc_stat, wilc_pval = scipy.stats.wilcoxon(p_VecJ,q_VecJ)#, alternative=test_tail)

print("\nTest \t\t| Statistic \t| p-value\t| < 0.05")
print("Mann \t\t|",
      round(mann_stat,2),
      "\t|",
      mann_pval,
      "\t\t|",
      (mann_pval < 0.05),
     )
print("Wilcoxon \t|",
      round(wilc_stat,2),
      "\t|",
      wilc_pval,
      "\t\t|",
      (wilc_pval < 0.05)
     )


"""----------------------------------------------------- State evolution."""

print("----------------------------------------")
print("Scatter plot.")

# Coordinates of X, U and V through the game, for each experiment. 
# IT ONLY TAKES INTO ACCOUNT GRAPHS FOR OPTIMAL GAME.
coordX = []
coordU = []
coordV = []

# Converst list of matrices into Data Frames.
for k in range(N):
    x_turn = []
    u_turn = []
    v_turn = []
    for c in range(NRep):
        x_turn.append(
            np.squeeze( np.array( p_MatX[c][k].transpose() ) )
            )
        u_turn.append(
            np.squeeze( np.array( p_MatU[c][k].transpose() ) )
            )
        v_turn.append(
            np.squeeze( np.array( p_MatV[c][k].transpose() ) )
            )
    coordX.append( x_turn )
    coordU.append( u_turn )
    coordV.append( v_turn )
coordX = pd.DataFrame(coordX)
coordU = pd.DataFrame(coordU)
coordV = pd.DataFrame(coordV)

# Column of Data frame is number of experiment.
coordX.columns = range(1,NRep + 1)
coordU.columns = range(1,NRep + 1)
coordV.columns = range(1,NRep + 1)

# Row of data frame is turn.
coordX.index.name = 'Turn'
coordU.index.name = 'Turn'
coordV.index.name = 'Turn'

""" Converts data frames into 2d Arrays for plotting."""
# Initial state.
x_scatter = np.hstack( (np.array(X_0), np.array([0]) ) )
u_scatter = np.hstack( (np.array(X_0), np.array([0]) ) )
v_scatter = np.hstack( (np.array(X_0), np.array([0]) ) )

# Rest of the turns.
for k in range(1,N):
    
    """ Formats coordinates of individual turn into 2D array. """
    # For X.
    x_search = query_to_array( coordX.query('Turn == %d'%k), k, NRep )
    x_scatter = np.vstack((x_scatter,x_search))
    
    # For U.
    u_search = query_to_array( coordU.query('Turn == %d'%k), k, NRep )
    u_scatter = np.vstack((u_scatter,u_search))
    
    # For V.
    v_search = query_to_array( coordV.query('Turn == %d'%k), k, NRep )
    v_scatter = np.vstack((v_scatter,v_search))

""" Overlayed plots of the states and actions """
complete_scatterplot(x_scatter,True)#, file_name = "X.png")
#complete_scatterplot(u_scatter,True)
#complete_scatterplot(v_scatter,False)

""" Individual plots of the states and actions """
individual_scatterplots(x_scatter,NRep)#, file_name = "Individual_X.png")
individual_scatterplots(u_scatter,NRep)#, file_name = "Individual_U.png")
individual_scatterplots(v_scatter,NRep)#, file_name = "Individual_V.png")



"""
traj = np.int(scipy.stats.randint.rvs(0,NRep,size=1))
x_pos = [ p_coordX[traj].values.tolist()[k][0] for k in range(N) ]
y_pos = [ p_coordX[traj].values.tolist()[k][1] for k in range(N) ]
x_dir = [ x_pos[k+1] - x_pos[k] for k in range(N-1) ]
y_dir = [ y_pos[k+1] - y_pos[k] for k in range(N-1) ]

plt.figure()
plt.quiver(
    x_pos[:-1],
    y_pos[:-1],
    x_dir,
    y_dir,
    angles='xy', 
    scale_units='xy', 
    scale = 1,
    width = 0.0035,
    #color=mpl.cm.jet
    )
plt.scatter(x_pos,y_pos,c='red')
plt.title("Trajectory %d"%traj)
plt.show()
"""

"""----------------------------------------------------- Payoff's histogram."""

print("----------------------------------------")
print("Probabilities Histograms.")

# Advantage prob. for each turn in every game, for each experiment.
p_advP = []
q_advP = []

for k in range(N):
    p_turn = []
    q_turn = []
    for c in range(NRep):
        p_turn.append( p_VecP[c][k] ) 
        q_turn.append( q_VecP[c][k] ) 
    p_advP.append( p_turn )
    q_advP.append( q_turn )

p_advP = pd.DataFrame(p_advP)
q_advP = pd.DataFrame(q_advP)

p_advP.columns = range(1,NRep + 1)
q_advP.columns = range(1,NRep + 1)

p_advP.index.name = 'Turn'
q_advP.index.name = 'Turn'

fig, ax = plt.subplots()
for k in range(1,9):
    
    plt.subplot( 3, 3, k )
    
    p_search = p_advP.query('Turn == %d'%(3*k))
    p_search = p_search.values.tolist()
    p_search = np.concatenate(p_search).ravel()
    
    im = plt.hist(
        p_search,
        color = "red",
        alpha= 0.5,
        bins = 30
        )
    
    q_search = q_advP.query('Turn == %d'%(3*k))
    q_search = q_search.values.tolist()
    q_search = np.concatenate(q_search).ravel()
    
    im = plt.hist(
        q_search,
        color = "green",
        alpha= 0.5,
        bins = 30
        )
    
    plt.title( 'Turn %d'%(3*k) )

#plt.colorbar()
#plt.legend(loc='best')
plt.tight_layout()
#plt.savefig('H_q.png')
plt.show()









