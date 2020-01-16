# Imports libraries
import numpy as np
import math as math
import statistics as stat
import matplotlib.pyplot as plt
import matplotlib as mpl

from decimal import *


"""------------------------------------------------ Initialising parameters."""

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

#   Initialize seed.
np.random.seed(20190604)

""" Dimensions of the parameters.
 d : X, A, Q
 m : U, B, R
 n : V, D, T
"""
d = 2
m = d
n = d

#   Turns.
N = 6

#   Identity matrix.
Id = np.mat(np.diag(np.array([1.00, 1.00])))

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

""" Declare the lists where adjusted coefficients will be kept.
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
"""        
print("Adjusted coefficients ------------------------")
for k in range(N):

    print("-------Turno ",k,":")

    print("S_", k+1, ":")
    print(S[k])

    print("Btilde_", k+1, ":")
    print(Btilde[k])

    print("Dtilde_", k+1, ":")
    print(Dtilde[k])
"""

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

"""------------------------------------------------------------ Simulations."""

# Repetitions.
NRep = 10000

# List of Payoffs.
VecJ = []

# List of Probabilities of advantage.
VecP = [ [0]*(N) for c in range(NRep) ]

VecXN0 = []
VecXN1 = []

# Matrix of simulated System's states.
MatX = [ [0]*(N+1) for c in range(NRep) ]

# Matrix of simulated Controls.
MatU = [ [0]*(N) for c in range(NRep) ]
MatV = [ [0]*(N) for c in range(NRep) ]

# Simulations of the game.
for c in range(NRep):
    # Uncomment to print current iteration.
    #print("****************************************")
    #print("Simulation ",c)

    # Lists for System and Noise.
    X=[]
    Y=[]

    """ Initial Condition.
    Replace with commented code to obtain Simulation 2.
    """
    X_0 = (0.80, 1.20) # np.random.multivariate_normal((0, 0), Id)
    X.append(
        np.transpose(
            np.mat(
                X_0
            )
        )
    )

    # Initialize cost.
    J = 0

    # Noise: Hidden Markov.
    Z = np.random.multivariate_normal((0, 0), Id, N-1)

    # Evolution of the game.
    for k in range(N):

        # Uncomment to check the state of the System in each turn.
        #print("--------------------------------")
        #print("Turno ", k)
        #print("X[",k,"]:",X[k])

        MatX[c][k] = X[k]

        U_k = np.transpose( np.mat( [0,0] ) )
        V_k = np.transpose( np.mat( [0,0] ) )

        """ Turn's Noise and factor F_k.
        Uncomment to check the values of F_k on each turn, as well as the
        Noise's one.
        """
        F_k = ( S[k] * A * X[k] )
        #print("F_k without noise:",F_k)
        if k == 0:
            Y_k = np.transpose(np.mat((0, 0)))
        else:
            Y_k = (M * Y[k - 1]) + (Sig * np.transpose(np.mat(Z[k - 1])))
            F_k_previo = F_k
            F_k = F_k_previo + ( CoFY[k] * Y[k-1] )
        #print("Y_k:", Y_k)
        #print("F_k:", F_k)
        Y.append(Y_k)


        """ Coin toss.
        Replace 'p' with '1-p' in the value Vec[c][k] and the conditional
        'unif < p' to obtain Simulation 3.
        """
        p = - math.expm1( (-0.5) * math.sqrt(
            float( np.transpose( X[k] ) * Q * X[k] )
        ) )
        unif = np.random.random_sample()
        VecP[c][k] = p

        """ 
        Uncomment to check the Advantage probability of Player I in each turn 
        and on each simulation.
        """
        #print("p(X_",k,"):",p)


        """ Choosing controls. 
        Uncomment print in each case to check the coin toss in each turn and 
        on each simulation.
        """
        # xi = 1
        if unif <= p:
            #print("Xi = 1.")
            V_k = d1[k] * F_k
            U_k = b1[k] * ( F_k + ( S[k] * D * V_k ) )

        # xi = 0
        else:
            #print("Xi = 0.")
            U_k = b0[k] * F_k
            V_k = d0[k] * ( F_k + ( S[k] * B * U_k ) )

        MatU[c][k] = U_k
        MatV[c][k] = V_k

        # Uncomment to check the controls in each turn and on each simulation.
        #print("U[", k, "]:", MatU[c][k])
        #print("V[", k, "]:", MatV[c][k])

        # Running cost.
        J_k = float(
            ( np.transpose( X[k] ) * Q * X[k] ) +
            ( np.transpose( U_k ) * R * U_k ) -
            ( np.transpose( V_k ) * T * V_k )
        )

        # Contribution to the Payoff.
        J += J_k
        #print("Turn's Cost J_k:", J_k)
        #print("Cost up to this turn:", J)


        # New state of the System.
        X_k = (A * X[k]) + (B * U_k) + (D * V_k) + Y[k]
        X.append(X_k)

    # Final cost.
    J += float( np.transpose( X[N] ) * Q * X[N] )

    # Save of the final state and total cost.
    MatX[c][N] = X[N]
    VecJ.append( J )
    VecXN0.append( float(X[N][0]) )
    VecXN1.append( float(X[N][1]) )

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
    X[:] = []
    Y[:] = []

"""---------------------------------------------------------------- Results."""


print("****************************************")

print("Number of repetitions: ",NRep)

print("----------------------------------------")

Val_J = stat.mean(VecJ)
print("Payoff: E[J]=",Val_J)

# Uncomment to print the deviation of the cost J.
Desv_J = stat.stdev(VecJ)
#print("Deviation: ",Desv_J)

print("----------------------------------------")

print("Mean advantege Probability.")

""" 
Uncomment to check the mean of p(X_k) in each iteration or through all the 
simulations.
"""

VecPmedia = []

# By iteration.
for c in range(NRep):
    VecPmedia.append( stat.mean(VecP[c]) )
    #print("Iteration",c,":",VecPmedia[c])

# In total.
print("Throughout all iterations:",stat.mean(VecPmedia))

"""----------------------------------------------------- Payoff's histogram."""

print("----------------------------------------")
print("Creating Histograms.")

title = "Game's Payoff"

desv_hist = [Val_J]
for cc in range(1,3):
    desv_hist.append( Val_J + (cc * Desv_J) )

text_box = ("Mean (Payoff): $\\mathbf{E}\\ J_N^0(X_0)  = $" + str( round( Decimal( Val_J ), 2 ) ))


fig_histo,ax_histo = plt.subplots()

chartBox = ax_histo.get_position()
ax_histo.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])

ax_histo.hist(
    VecJ,
    density=True,
    bins=100,
    range= ( 0, desv_hist[len(desv_hist)-1] + Desv_J ),
)

# Draw dispersion lines.
for cc in desv_hist:
    if cc == Val_J:
        linea = 'red'
    else:
        linea = 'k'
    plt.axvline(cc, color=linea, linestyle='dashed', linewidth=0.5 )



plt.title(title)
plt.ylabel("Frequency")
plt.xlabel("Cost of the game $J_N^0(X_0)(\\omega)$")

ax_histo.text(
    0.5,-0.35,
    text_box,
    ha='center',
    #va='center',
    transform=ax_histo.transAxes,
    bbox = dict(
        facecolor = 'white',
        edgecolor = 'black',
        pad=10,
    ),
)


plt.tight_layout()

#plt.savefig('Figures/Fig_01.png')
plt.show()

print("Payoff.")

"""---------------------------------- Histogram of Advantage probabilities. """

"""
Lists where the components of processes {X}, {U}, {V} and {p} will be kept. 
These will be used throughout the rest of the graphs.
"""

XComp0 = [ [0]*NRep for k in range(N+1) ]
XComp1 = [ [0]*NRep for k in range(N+1) ]

UComp0 = [ [0]*NRep for k in range(N) ]
UComp1 = [ [0]*NRep for k in range(N) ]

VComp0 = [ [0]*NRep for k in range(N) ]
VComp1 = [ [0]*NRep for k in range(N) ]

pComp = [ [0]*NRep for k in range(N) ]
pCompAlt = [ [0]*NRep for k in range(N-2) ]

for c in range(NRep):
    for k in range(N):
        XComp0[k][c] = float( MatX[c][k][0] )
        XComp1[k][c] = float( MatX[c][k][1] )

        UComp0[k][c] = float( MatU[c][k][0] )
        UComp1[k][c] = float( MatU[c][k][1] )

        VComp0[k][c] = float( MatV[c][k][0] )
        VComp1[k][c] = float( MatV[c][k][1] )

        pComp[k][c] = VecP[c][k]

        if k >= 2:
            pCompAlt[k-2][c] = VecP[c][k]
    XComp0[N][c] = float(MatX[c][N][0])
    XComp1[N][c] = float(MatX[c][N][1])


# Histograms of p(X_k) for all k.
title = "Adventage probabilities"

fig_phisto,ax_phisto = plt.subplots()

chartBox = ax_phisto.get_position()
ax_phisto.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])

for k in range(N):
    ax_phisto.hist(pComp[k],
                   density=True,
                   alpha = 0.3,
                   bins = 50,
                   label = 'Turn '+str(k),
                   range= (0,1),
                  )

plt.legend(bbox_to_anchor=(1.05, 0.5),
           loc='center left',
           borderaxespad=0.,
           frameon=True,
           )

#plt.legend().get_frame().set_edgecolor=('black')

plt.title(title)
plt.ylabel("Frequency")
plt.xlabel("$p(X_k)$")

plt.tight_layout()

#plt.savefig('Figures/Fig_02.png')
plt.show()

# Histograms of p(X_k) for k > 1.
title = "Adventage probabilities (Turns 2 and onwards)."

fig_phistoAlt,ax_phistoAlt = plt.subplots()

chartBox = ax_phistoAlt.get_position()
ax_phistoAlt.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])

for k in range(N-2):
    ax_phistoAlt.hist(pCompAlt[k],
                   density=True,
                   alpha = 0.3,
                   bins = 50,
                   label = 'Turn '+str(k+2),
                   range= (0,1),
                  )

plt.legend(bbox_to_anchor=(1.05, 0.5),
           loc='center left',
           borderaxespad=0.,
           frameon=True,
           )
plt.title(title)
plt.ylabel("Frequency")
plt.xlabel("$p(X_k)$")

plt.tight_layout()

#plt.savefig('Figures/Fig_02b.png')
plt.show()


print("Advantage probabilities.")

"""---------------------------------------------------- Sates of the System."""

print("----------------------------------------")
print("Creating Dot Graphs.")

print("\n---------- System's states:")



""" Tags for processes {X} {U} and {V}.
These will be used throughout the rest of the graphs.
"""
turno = []
U_turno =[]
V_turno =[]
etiqueta = []
for k in range(N):
    turno.append( ( XComp0[k], XComp1[k] ) )

    U_turno.append( ( UComp0[k], UComp1[k] ) )

    V_turno.append( ( VComp0[k], VComp1[k] ) )

    etiqueta.append( "Turn "+str(k) )
turno.append( ( XComp0[N], XComp1[N] ) )
etiqueta.append( "Turn "+str(N) )



""" Color palette.
Uncomment for random color palette. Works for N > 6 turns.
"""
colores =  ("black","green","brown","blue","purple","red","orange")
#hexa = ('0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F')
#colores = ["#"+''.join([np.random.choice( hexa ) for j in range(6)]) for i in range(N+1)]
#colores = sorted(colores)

U_colores = colores
V_colores = colores

U_etiqueta = etiqueta
V_etiqueta = etiqueta


U_turno_ind = U_turno
U_colores_ind = U_colores
U_etiqueta_ind = U_etiqueta

V_turno_ind = V_turno
V_colores_ind = V_colores
V_etiqueta_ind = V_etiqueta

turno_ind = turno
colores_ind = colores
etiqueta_ind = etiqueta



""" Realisations of X_k (per turn). """
nn = 1
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for turno_ind, colores_ind, etiqueta_ind in zip(turno_ind, colores_ind, etiqueta_ind):
    ax = fig.add_subplot( 3, 3, nn )
    x, y = turno_ind
    ax.scatter(x, y,
               alpha=0.6,
               c=colores_ind,
               edgecolors='none',
               s=30,
               )
    ax.title.set_text('$X_'+str(nn-1)+'$')

    #ax.set_xlabel('1st Componet of $X_'+str(nn-1)+'$' )
    #ax.set_ylabel('2nd Componet of $X_'+str(nn-1)+'$' )

    nn += 1

plt.suptitle('System states (individual turns)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#plt.savefig('Figures/Fig_03.png')
plt.show()

print("Individual realisations of X_k")




""" Realisations of {X_k} (all turns). """
fig_total = plt.figure()
ax_total = fig_total.add_subplot(1, 1, 1)  # , axisbg="1.0")

chartBox = ax_total.get_position()
ax_total.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])

for turno, colores, etiqueta in zip(turno, colores, etiqueta):

    x_total, y_total = turno
    ax_total.scatter(x_total, y_total,
                     alpha=0.4,
                     c=colores,
                     edgecolors='none',
                     s=30,
                     label=etiqueta)

plt.title('System states (all turns)')
plt.legend(bbox_to_anchor=(1.05, 0.5),
           loc='center left',
           borderaxespad=0.,
           frameon=True,
           )

#plt.xlabel('1st Componet de $X$')
#plt.ylabel('2nd Componet de $X$')

plt.tight_layout()

#plt.savefig('Figures/Fig_04.png')
plt.show()
print("Realisations of the process {X_k}.")




"""--------------------------------------------------- Controls of player I."""


""" Realisations of U_k (per turn). """

print("\n---------- Actions of player I:")
nn = 0
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for U_turno_ind, U_colores_ind, U_etiqueta_ind in zip(U_turno_ind, U_colores_ind, U_etiqueta_ind):
    ax = fig.add_subplot(2, 3, nn+1)  # , axisbg="1.0")
    x, y = U_turno_ind
    ax.scatter(x, y,
               alpha=0.6,
               c=U_colores_ind,
               edgecolors='none',
               s=30)
    ax.title.set_text('$U_'+str(nn)+'$')

    #ax.set_xlabel('1st Componet de $U_' + str(nn) + '$')
    #ax.set_ylabel('2nd Componet de $U_' + str(nn) + '$')

    nn += 1

plt.suptitle('Actions of player I (individual turns)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#plt.savefig('Figures/Fig_05.png')
plt.show()
print("Individual realisations of U_k")



""" Realisations of {U_k} (all turns). """
fig_total = plt.figure()
ax_total = fig_total.add_subplot(1, 1, 1)  # , axisbg="1.0")

chartBox = ax_total.get_position()
ax_total.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])

for U_turno, U_colores, U_etiqueta in zip(U_turno, U_colores, U_etiqueta):

    x_total, y_total = U_turno
    ax_total.scatter(x_total, y_total,
                     alpha=0.4,
                     c=U_colores,
                     edgecolors='none',
                     s=30,
                     label=U_etiqueta)

plt.title('Actions of player I (all turns)')
plt.legend(bbox_to_anchor=(1.05, 0.5),
           loc='center left',
           borderaxespad=0.,
           frameon=True,
           )


#plt.xlabel('1st Componet de $U$')
#plt.ylabel('2nd Componet de $U$')

plt.tight_layout()

#plt.savefig('Figures/Fig_06.png')
plt.show()
print("Realisations of the process {U_k}.")




"""-------------------------------------------------- Controls of player II."""


""" Realisations of V_k (per turn). """
print("\n---------- Actions of player II:")
nn = 0
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for V_turno_ind, V_colores_ind, V_etiqueta_ind in \
        zip(V_turno_ind, V_colores_ind, V_etiqueta_ind):
    ax = fig.add_subplot(2, 3, nn+1)  # , axisbg="1.0")
    x, y = V_turno_ind
    ax.scatter(x, y,
               alpha=0.6,
               c=V_colores_ind,
               edgecolors='none',
               s=30)
    ax.title.set_text('$V_'+str(nn)+'$')

    #ax.set_xlabel('1st Componet de $V_' + str(nn) + '$')
    #ax.set_ylabel('2nd Componet de $V_' + str(nn) + '$')

    nn += 1

plt.suptitle('Actions of player II (individual turns)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#plt.savefig('Figures/Fig_07.png')
plt.show()
print("Individual realisations of V_k")



""" Realisations of {V_k} (all turns). """
fig_total = plt.figure()
ax_total = fig_total.add_subplot(1, 1, 1)  # , axisbg="1.0")

chartBox = ax_total.get_position()
ax_total.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])

for V_turno, V_colores, V_etiqueta in zip(V_turno, V_colores, V_etiqueta):

    x_total, y_total = V_turno
    ax_total.scatter(x_total, y_total,
                     alpha=0.4,
                     c=V_colores,
                     edgecolors='none',
                     s=30,
                     label=V_etiqueta)

plt.title('Actions of player II (all turns)')
plt.legend(bbox_to_anchor=(1.05, 0.5),
           loc='center left',
           borderaxespad=0.,
           frameon=True,
           )


#plt.xlabel('1er Componete de $V$')
#plt.ylabel('2da Componete de $V$')
plt.tight_layout()

#plt.savefig('Figures/Fig_08.png')
plt.show()
print("Realisations of the process {V_k}.")

""" Code to check indivdual parts of the program.

# To check states per turn:
print("----------------------------------------")
print("Escenario ",0,", turno ",1)
print(MatX[0][1])
print("Escenario ",2,", turno ",3)
print(MatX[2][3])
print("Escenario ",4,", turno ",5)
print(MatX[4][5])

# To check the components of the states per turn:
#print("----------------------------------------")
#print("Componentes NO son flotantes")
#print("Escenario ",0,", turno ",1," componente ",0)
#print(MatX[0][1][0])
#print("Escenario ",2,", turno ",3," componente ",1)
#print(MatX[2][3][1])
#print("Escenario ",4,", turno ",5," componente ",0)
#print(MatX[4][5][0])

# To check states in an iteration:
#print("----------------------------------------")
#print("Estados")
#print("Escenario ",0,", todos los turnos.")
#print(MatX[0])
#print("Escenario ",2,", todos los turnos.")
#print(MatX[2])
#print("Escenario ",4,", todos los turnos.")
#print(MatX[4])
"""

print("----------------------------------------")
print("End")