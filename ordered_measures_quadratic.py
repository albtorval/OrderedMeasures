
### ORDERED MEASURES: QUADRATIC MEASURES

import numpy as np
import pandas as pd 

import math
import statistics
from scipy import stats
from scipy.stats.mstats import winsorize, mquantiles, trim

import time
import random

import gurobipy as gp
from gurobipy import GRB

#############################
############################# QUADRATIC MEASURES
#############################

## - Formulation 1: sorting constraints
def OM_quadratic_binary(n, M, X, timeLimit = 7200, rseed = 1234): 
    
    start = time.time() 

    N = range(n)

    m = gp.Model("OM_quad_1")
    m.Params.LogToConsole = 0
    m.Params.Seed = rseed
    m.setParam("TimeLimit", timeLimit)
    m.Params.NonConvex  = 2
    m.Params.Threads    = 8
    m.Params.PreCrush   = 0
    m.Params.Presolve   = 0
    m.Params.Heuristics = 0
    m.Params.Cuts       = 0

    # Variables:
    z = {}
    for i in N:
        for k in N:
            z[i,k] = m.addVar(vtype=GRB.BINARY, lb=0, ub=1, name="z_"+str(i)+"_"+ str(k))
    y = {}
    for i in N:
        for k in N:
            for j in N:
                for l in N:
                    y[i,j,k,l] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="y_"+str(i)+"_"+ str(j)+"_"+ str(k)+"_"+ str(l))

    # Objective:
    m.setObjective( sum([ M[k,l]*y[i,j,k,l]*X[i]*X[j]  # z[i,k]*z[j,l]
                        for i in N for j in N
                        for k in N for l in N]), GRB.MINIMIZE)

    # Constraints:
    for i in N:
        m.addConstr(sum([z[i,k] for k in N]) == 1)
    for k in N:
        m.addConstr(sum([z[i,k] for i in N]) == 1)
    for k in N:
        if k < n-1:
            m.addConstr(sum([z[i,k]*X[i] for i in N]) <= sum([z[i,k+1]*X[i] for i in N]))

    for i in N:
        for k in N:
            for j in N:
                for l in N:
                    m.addConstr( y[i,j,k,l] <= z[i,k] )
                    m.addConstr( y[i,j,k,l] <= z[j,l] )
                    m.addConstr( z[i,k] + z[j,l] - 1 <= y[i,j,k,l] )
    
    m.optimize()

    done = time.time()
    elapsed = done - start

    # rl = m.relax()
    # rl.optimize()

    # Results:
    results = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    if m.Status == 3 or m.Status == 4:
        print("-- Model infeasible or unbounded")
        results[0]  = "quadratic"
        results[1]  = "general"
        results[2]  = n
        results[3]  = m.runtime
        results[4]  = np.nan
        results[5]  = np.nan
        results[6]  = X
        results[7]  = np.nan
        results[8]  = elapsed
    else:
        if m.SolCount == 0:
            print("-- No solution found in time limit")
            results[0]  = "quadratic"
            results[1]  = "general"
            results[2]  = n
            results[3]  = m.runtime
            results[4]  = np.nan
            results[5]  = m.objBound
            results[6]  = X
            results[7]  = np.nan
            results[8]  = elapsed
        else:
            results[0]  = "quadratic"
            results[1]  = "general"
            results[2]  = n
            results[3]  = m.runtime
            results[4]  = m.objVal
            results[5]  = m.objBound
            results[6]  = X
            results[7] = {(i+1,k):z[i,k].x for i in N for k in N if z[i,k].x>0.5 }
            results[8]  = elapsed

    m.dispose()

    return results

