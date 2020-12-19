# Import packages
import numpy as np
import matplotlib.pyplot as plt
import corner
from copy import deepcopy



# Define prior and likelihood function for linear fit
def logLikelihoodRho0(a, b, x, y, sigX, sigY):
    return -0.5*np.sum(((y-a-b*x)*(y-a-b*x))/((b*b*sigX*sigX)+(sigY*sigY)))

def logPrior1(b):
    return 0.

def logPrior2(b):
    return -np.log(np.abs(b))

def logPrior3(b):
    return -np.log(1+b*b)

def logProb(a, b, x, y, sigX, sigY, priorFunct):
    return logLikelihoodRho0(a, b, x, y, sigX, sigY) + priorFunct(b)


# 2D MCMC code (logarithmic)
def MCMC2D_Log_General(LogProbabilityDistrib, N, sigG, initialGuess, args=()):
    """
    Markov Chain Monte Carlo for Log Probability, written by Soley Hyman 12/3/2020
    (modified for general number of samplings)

    Inputs:
    LogProbabilityDistrib: distribution function in log space (function)
    N: number of gaussian steps (integer)
    sigG: standard deviation of epsilon Gaussian (float)
    args: parameters for distribution function (tuple)

    Outputs:
    xValues: resulting values from MCMC (list)
    acceptanceRate: acceptance rate of MCMC (float)
    """
    # get number of free parameters
    freeParams = len(initialGuess)
    
    # make acceptance counter and acceptance rate calculator
    acceptanceCounter = 0
    totalNumberPoints = 0
    values = np.zeros([int(N), freeParams])
    ##
    # step 1: draw initial xi
    currentVals = initialGuess
    ##
    # for x in range(0,int(N)):
    while totalNumberPoints < int(N):
        # step 2: take step to xi+1 = xi+epsilon
        epsilons = np.random.normal(scale=sigG, size=freeParams)
        newVals = currentVals+epsilons
        ##
        # step 3: calc R = P(xi+1)/P(xi)
        R = LogProbabilityDistrib(*newVals, *args)-LogProbabilityDistrib(*currentVals, *args)
        ##
        if R < 1:
            p = np.log(np.random.uniform(low=0., high=1., size=1) [0])
            if p > R:
                currentVals= currentVals
                values[totalNumberPoints] = deepcopy(currentVals)
                totalNumberPoints += 1
            else:
                currentVals = newVals
                values[totalNumberPoints] = deepcopy(currentVals)
                acceptanceCounter += 1
                totalNumberPoints += 1
        else:
            currentVals = newVals
            values[totalNumberPoints] = deepcopy(currentVals)
            acceptanceCounter += 1
            totalNumberPoints += 1
    ##
    acceptanceRate = acceptanceCounter/totalNumberPoints
    print('\nAcceptance Rate = {}\n'.format(acceptanceRate))
    ##
    return values, acceptanceRate


#### EXAMPLE RUN ####
# MCMC draw - prior case 2
params = (BMAGs, Vrots, e_BMAGs, err_Vrots, logPrior2)
values, acceptanceRate = MCMC2D_Log_General(logProb, 5e5, sigG=0.01, initialGuess=(-1.0,-0.15), args=params)

aVals, bVals = values.T
plt.plot(aVals,bVals,marker='.',ls='None')
plt.xlabel('a')
plt.ylabel('b')
plt.show()

plt.hist2d(aVals,bVals,bins=100)
plt.xlabel('a')
plt.ylabel('b')
plt.show()

combined = np.vstack((aVals,bVals)).T
figure = corner.corner(combined, labels=["a", "b"],
                       levels=[0.683,0.954,0.997],
                       plot_datapoints=False,
                       plot_density=False,
                       plot_contours=True,
                       show_titles=True, title_kwargs={"fontsize": 12})
plt.show()