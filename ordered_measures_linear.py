
### ORDERED MEASURES: LINEAR MEASURES

import numpy as np
import pandas as pd 

import math
import statistics
from scipy import stats
from scipy.stats.mstats import winsorize, mquantiles

import time
import random

import gurobipy as gp
from gurobipy import GRB

#############################
############################# LINEAR MEASURES
#############################

## - Formulation 1: sorting constraints
def OM_linear_1(n, Lambda, X, timeLimit = 7200, rseed = 1234): 

    start = time.time()

    N = range(n) ### Empieza en cero

    m = gp.Model("OM_lin_1")
    m.Params.LogToConsole = 0
    m.Params.Seed = rseed
    m.setParam("TimeLimit", timeLimit)
    m.Params.Threads = 8
    # m.Params.NonConvex = 2
    m.Params.PreCrush   = 0
    m.Params.Presolve   = 0
    m.Params.Heuristics = 0
    m.Params.Cuts       = 0

    # Variables:
    z = {}
    for i in N:
        for k in N:
            z[i,k] = m.addVar(vtype=GRB.BINARY, lb=0, ub=1, name="z_"+str(i)+"_"+ str(k))

    # Objective:
    m.setObjective(sum([Lambda[k]*z[i,k]*X[i] for i in N for k in N]), GRB.MINIMIZE)

    # Constraints:
    for i in N:
        m.addConstr(sum([z[i,k] for k in N]) == 1)
    for k in N:
        m.addConstr(sum([z[i,k] for i in N]) == 1)
    for k in N:
        if k < n-1:
            m.addConstr(sum([z[i,k]*X[i] for i in N]) <= sum([z[i,k+1]*X[i] for i in N]))
    
    m.optimize()

    # rl = m.relax()
    # rl.optimize()
    
    done = time.time()
    elapsed = done - start

    # Results:
    results = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    if m.Status == 3 or m.Status == 4:
        print("-- Model infeasible or unbounded")
        results[0]  = "linear"
        results[1]  = "1"
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
            results[0]  = "linear"
            results[1]  = "1"
            results[2]  = n
            results[3]  = m.runtime
            results[4]  = np.nan
            results[5]  = m.objBound
            results[6]  = X
            results[7]  = np.nan
            results[8]  = elapsed
        else:
            results[0]  = "linear"
            results[1]  = "1"
            results[2]  = n
            results[3]  = m.runtime #elapsed
            results[4]  = m.objVal
            results[5]  = m.objBound
            results[6]  = X
            results[7] = {(i+1,k):z[i,k].x for i in N for k in N if z[i,k].x>0.5 }
            results[8]  = elapsed
    


    m.dispose()

    return results

## - Formulation 2: k-sums
def OM_linear_2(n, Lambda, X, timeLimit = 7200, rseed = 1234):

    start = time.time()
    
    N = range(n) ### Empieza en cero

    m = gp.Model("OM_lin_2")
    m.Params.LogToConsole = 0
    m.Params.Seed = rseed
    m.setParam("TimeLimit", timeLimit)
    m.Params.Threads = 8
    # m.Params.NonConvex = 2
    m.Params.PreCrush   = 0
    m.Params.Presolve   = 0
    m.Params.Heuristics = 0
    m.Params.Cuts       = 0

    DELTA = [t - s for s, t in zip([0] + Lambda, ([0] + Lambda)[1:])]

    # Variables:
    d = {}
    for i in N:
        for k in N:
            if DELTA[k] < 0:
                d[i,k] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="d_"+str(i)+"_"+str(k))
    t = {}
    for k in N:
        if DELTA[k] > 0:
            t[k] = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t_"+str(k))
    v = {}
    for i in N:
        for k in N:
            if DELTA[k] > 0:
                v[i,k] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="v_"+str(i)+"_"+str(k))

    # Objective:
    m.setObjective(
        sum([DELTA[k] *
             ((n-k)*t[k] + sum([v[i,k] for i in N])) for k in N if DELTA[k] > 0]) +
        sum([DELTA[k] * 
             sum([X[i]*d[i,k] for i in N]) for k in N if DELTA[k] < 0]), 
        GRB.MINIMIZE)

    # Constraints:
    for i in N:
        for k in N:
            if DELTA[k] > 0:
                m.addConstr(t[k] + v[i,k] >= X[i])
    for k in N:
        if DELTA[k] < 0:
            m.addConstr(sum([d[i,k] for i in N]) == n-k)

    # Optimize:
    m.optimize()

    done = time.time()
    elapsed = done - start

    # rl = m.relax()
    # rl.optimize()

    # Results:
    results = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    if m.Status == 3 or m.Status == 4:
        print("-- Model infeasible or unbounded")
        results[0]  = "linear"
        results[1]  = "2"
        results[2]  = n
        results[3]  = m.runtime
        results[4]  = np.nan
        results[5]  = np.nan
        results[6]  = X
        results[7]  = np.nan
        results[8]  = np.nan
        results[9]  = np.nan
        results[10]  = elapsed
    else:
        if m.SolCount == 0:
            print("-- No solution found in time limit")
            results[0]  = "linear"
            results[1]  = "2"
            results[2]  = n
            results[3]  = m.runtime
            results[4]  = np.nan
            results[5]  = m.objBound
            results[6]  = X
            results[7]  = np.nan
            results[8]  = np.nan
            results[9]  = np.nan
            results[10]  = elapsed
        else:
            results[0]  = "linear"
            results[1]  = "2"
            results[2]  = n
            results[3]  = m.runtime #elapsed
            results[4]  = m.objVal
            results[5]  = m.objBound
            results[6]  = X
            results[7] = {(i+1,k):d[i,k].x for i in N for k in N if DELTA[k] < 0 if d[i,k].x>0.5}  
            results[8] = {(i+1,k):v[i,k].x for i in N for k in N if DELTA[k] > 0 if v[i,k].x>0.5}  
            results[9] = {k:t[k].x for k in N if DELTA[k] > 0 if t[k].x>0.5}  
            results[10]  = elapsed

    m.dispose()

    return results

## - Formulation 3: quantile loss
def OM_linear_3(n, Lambda, X, timeLimit = 7200, rseed = 1234):

    start = time.time()
    
    N = range(n) ### Empieza en cero

    m = gp.Model("OM_lin_3")
    m.Params.LogToConsole = 0
    m.Params.Seed = rseed
    m.setParam("TimeLimit", timeLimit)
    m.Params.Threads = 8
    # m.Params.NonConvex = 2
    m.Params.PreCrush   = 0
    m.Params.Presolve   = 0
    m.Params.Heuristics = 0
    m.Params.Cuts       = 0

    DELTA = [t - s for s, t in zip([0] + Lambda, ([0] + Lambda)[1:])]

    # Variables:
    alpha = {}
    for i in N:
        for k in N:
            if DELTA[k] < 0:
                alpha[i,k] = m.addVar(vtype=GRB.CONTINUOUS, lb=-k/n, ub=1-k/n, name="alpha_"+str(i)+"_"+str(k))
    gamma = {}
    for k in N:
        if DELTA[k] > 0:
            gamma[k] = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="gamma_"+str(k))
    u = {}
    for i in N:
        for k in N:
            if DELTA[k] > 0:
                u[i,k] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="u_"+str(i)+"_"+str(k))
    v = {}
    for i in N:
        for k in N:
            if DELTA[k] > 0:
                v[i,k] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="v_"+str(i)+"_"+str(k))

    # Objective:
    m.setObjective(
        sum([DELTA[k] *
            ((k/n)*sum([u[i,k] for i in N]) + (1-k/n)*sum([v[i,k] for i in N])) 
            for k in N if DELTA[k] > 0]) +
        sum([DELTA[k] * 
            (sum([X[i]*alpha[i,k] for i in N])) 
            for k in N if DELTA[k] < 0]) +
        sum([X[i]*(1-k/n)*DELTA[k] for i in N for k in N]),
        GRB.MINIMIZE)

    # Constraints:
    for i in N:
        for k in N:
            if DELTA[k] > 0:
                m.addConstr(u[i,k] - v[i,k] + gamma[k] == X[i])
    for k in N:
        if DELTA[k] < 0:
            m.addConstr(sum([alpha[i,k] for i in N]) == 0)

    m.optimize()

    done = time.time()
    elapsed = done - start

    # rl = m.relax()
    # rl.optimize()

    # Results:
    results = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    if m.Status == 3 or m.Status == 4:
        print("-- Model infeasible or unbounded")
        results[0]  = "linear"
        results[1]  = "3"
        results[2]  = n
        results[3]  = m.runtime #elapsed
        results[4]  = np.nan
        results[5]  = np.nan
        results[6]  = X
        results[7]  = np.nan
        results[8]  = np.nan
        results[9]  = np.nan
        results[10] = np.nan
        results[11]  = elapsed
    else:
        if m.SolCount == 0:
            print("-- No solution found in time limit")
            results[0]  = "linear"
            results[1]  = "3"
            results[2]  = n
            results[3]  = m.runtime #elapsed
            results[4]  = np.nan
            results[5]  = m.objBound
            results[6]  = X
            results[7]  = np.nan
            results[8]  = np.nan
            results[9]  = np.nan
            results[10] = np.nan
            results[11]  = elapsed
        else:
            results[0]  = "linear"
            results[1]  = "3"
            results[2]  = n
            results[3]  = m.runtime #elapsed
            results[4]  = m.objVal
            results[5]  = m.objBound
            results[6]  = X
            results[7]  = {(i+1,k):alpha[i,k].x for i in N for k in N if DELTA[k] < 0 if alpha[i,k].x>0.5}  
            results[8]  = {(i+1,k):u[i,k].x for i in N for k in N if DELTA[k] > 0 if u[i,k].x>0.5}  
            results[9]  = {(i+1,k):v[i,k].x for i in N for k in N if DELTA[k] > 0 if v[i,k].x>0.5}  
            results[10] = {k:gamma[k].x for k in N if DELTA[k] > 0 if gamma[k].x>0.5}  
            results[11]  = elapsed

    m.dispose()

    return results


