
### ORDERED MEASURES: NESTED MEASURES

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
############################# NESTED MEASURES
#############################

## - Formulation 1: k-sums bilevel
def OM_nested_1(n, Lambda, X, which_rho, timeLimit = 7200, rseed = 1234):

    start = time.time()
    
    N = range(n) ### Empieza en cero

    m = gp.Model("OM_nested_1")
    m.Params.LogToConsole = 0
    m.Params.Seed = rseed
    m.setParam("TimeLimit", timeLimit)
    m.Params.NonConvex  = 2
    m.Params.Threads    = 8
    m.Params.PreCrush   = 0
    m.Params.Presolve   = 0
    m.Params.Heuristics = 0
    m.Params.Cuts       = 0

    DELTA = [t - s for s, t in zip([0] + Lambda, ([0] + Lambda)[1:])]

    # Variables:
    theta = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="theta")
    d = {}
    for i in N:
        for k in N:
            d[i,k] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="d_"+str(i)+"_"+str(k))
    t = {}
    for k in N:
        t[k] = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t_"+str(k))
    v = {}
    for i in N:
        for k in N:
            v[i,k] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="v_"+str(i)+"_"+str(k))

    # Constraints:
    m.addConstr( theta == sum([ abs(DELTA[k]) * ((n-k)*t[k] + sum([v[i,k] for i in N])) for k in N if DELTA[k] > 0]) - 
                sum([ abs(DELTA[k]) * sum([X[i]*d[i,k] for i in N]) for k in N if DELTA[k] < 0]) )
    m.addConstr( theta == - sum([ abs(DELTA[k]) * ((n-k)*t[k] + sum([v[i,k] for i in N])) for k in N if DELTA[k] < 0]) + 
                sum([ abs(DELTA[k]) * sum([X[i]*d[i,k] for i in N]) for k in N if DELTA[k] > 0]) )
    for i in N:
        for k in N:
            # if DELTA[k] > 0:
            m.addConstr(t[k] + v[i,k] >= X[i])
    for k in N:
        # if DELTA[k] < 0:
        m.addConstr(sum([d[i,k] for i in N]) == n-k)    

    # Function type
    if which_rho == "squaredev":        
        def rho(X,theta_var):
            return (1/n)*sum( [ (X[i]-theta_var)**2 for i in N ] )
        m.setObjective( rho(X, theta), GRB.MINIMIZE)
        m.optimize()
        value = m.objVal
        objbound = np.nan
    if which_rho == "absdev":
        r     = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="r")
        s     = m.addVars(n, lb=0, ub=GRB.INFINITY, name="s")
        for i in range(n):
            m.addConstr( r[i] == X[i] - theta )
            m.addGenConstrAbs(s[i], r[i])
        m.setObjective( (1/n)*sum([ s[i] for i in N ]), GRB.MINIMIZE)
        m.optimize()
        value    = m.objVal
        objbound = m.ObjBound
    if which_rho == "skew":
        x = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="x")
        xx = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="xx")
        y = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="y")
        z = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="z")
        yy = {}
        yyy = {}
        for i in N:
            yy[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="yy_"+str(i))
            yyy[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="yyy_"+str(i))
        for i in N:
            m.addConstr( yy[i] == (X[i]-theta)**2 )
            m.addConstr( yyy[i] == yy[i]*(X[i]-theta) )
        m.addConstr(x == (1/n)*sum( [ yy[i] for i in N ] ))
        m.addConstr(xx == x*x)
        m.addConstr(y*y==xx*x)
        m.addConstr(z*y == 1)
        m.setObjective( z*(1/n)*sum( [ yyy[i] for i in N ] ), GRB.MINIMIZE)
        m.optimize()
        value    = m.objVal
        objbound = np.nan
    if which_rho == "kurt":
        x  = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="x")
        xx = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="xx")
        y  = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="y")
        yy = {}
        yyyy = {}
        for i in N:
            yy[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="yy_"+str(i))
            yyyy[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="yyy_"+str(i))
        for i in N:
            m.addConstr( yy[i] == (X[i]-theta)**2 )
            m.addConstr( yyyy[i] == yy[i]*yy[i] )
        m.addConstr(x == (1/n)*sum( [ yy[i] for i in N ] ))
        m.addConstr(xx == x*x)
        m.addConstr(y*xx == 1)
        m.setObjective( y * (1/n) * sum( [ yyyy[i] for i in N ] ) -3, GRB.MINIMIZE) #1
        m.optimize()
        value = y.x * (1/n) * sum( [ yyyy[i].x for i in N ] ) -3
        objbound = np.nan

    done = time.time()
    elapsed = done - start

    # rl = m.relax()
    # rl.optimize()

    # Results:
    results = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    if m.Status == 3 or m.Status == 4:
        print("-- Model infeasible or unbounded")
        results[0]  = "nested"
        results[1]  = "1"
        results[2]  = n
        results[3]  = m.runtime
        results[4]  = np.nan
        # results[5]  = np.nan
        results[6]  = X
        results[7]  = np.nan
        results[8]  = np.nan
        results[9]  = np.nan
        results[10] = np.nan
        results[11] = elapsed
    else:
        if m.SolCount == 0:
            print("-- No solution found in time limit")
            results[0]  = "nested"
            results[1]  = "1"
            results[2]  = n
            results[3]  = m.runtime
            results[4]  = np.nan
            # results[5]  = m.objBound
            results[6]  = X
            results[7]  = np.nan
            results[8]  = np.nan
            results[9]  = np.nan
            results[10] = np.nan
            results[11] = elapsed
        else:
            results[0]  = "nested"
            results[1]  = "1"
            results[2]  = n
            results[3]  = m.runtime #elapsed
            results[4]  = value
            results[5]  = objbound
            results[6]  = X
            results[7]  = {(i+1,k):d[i,k].x for i in N for k in N if DELTA[k] < 0 if d[i,k].x>0.5}  
            results[8]  = {(i+1,k):v[i,k].x for i in N for k in N if DELTA[k] > 0 if v[i,k].x>0.5}  
            results[9]  = {k:t[k].x for k in N if DELTA[k] > 0 if t[k].x>0.5}  
            results[10] = theta.x
            results[11] = elapsed

    m.dispose()

    return results

## - Formulation 2: quantile bilevel
def OM_nested_2(n, Lambda, X, which_rho, timeLimit = 7200, rseed = 1234):

    start = time.time()
    
    N = range(n) ### Empieza en cero

    m = gp.Model("OM_nested_2")
    m.Params.LogToConsole = 0
    m.Params.Seed = rseed
    m.setParam("TimeLimit", timeLimit)
    m.Params.NonConvex  = 2
    m.Params.Threads    = 8
    m.Params.PreCrush   = 0
    m.Params.Presolve   = 0
    m.Params.Heuristics = 0
    m.Params.Cuts       = 0

    DELTA = [t - s for s, t in zip([0] + Lambda, ([0] + Lambda)[1:])]

    # Variables:
    theta = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="theta")
    alpha = {}
    for i in N:
        for k in N:
            alpha[i,k] = m.addVar(vtype=GRB.CONTINUOUS, lb=-k/n, ub=1-k/n, name="alpha_"+str(i)+"_"+str(k)) #lb=(k/n-1), ub=k/n, 
    gamma = {}
    for k in N:
        gamma[k] = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="gamma_"+str(k))
    u = {}
    for i in N:
        for k in N:
            u[i,k] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="u_"+str(i)+"_"+str(k))
    v = {}
    for i in N:
        for k in N:
            v[i,k] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="v_"+str(i)+"_"+str(k))

    # Constraints:
    m.addConstr(theta == sum([ abs(DELTA[k]) * ((k/n)*sum([u[i,k] for i in N]) + (1-k/n)*sum([v[i,k] for i in N])) for k in N if DELTA[k] > 0]) +
                sum([ abs(DELTA[k]) * (sum([X[i]*alpha[i,k] for i in N])) for k in N if DELTA[k] < 0]) +
                sum([ DELTA[k] * X[i]*(1-k/n) for i in N for k in N]) )
    m.addConstr(theta == -sum([ abs(DELTA[k]) * ((k/n)*sum([u[i,k] for i in N]) + (1-k/n)*sum([v[i,k] for i in N])) for k in N if DELTA[k] < 0]) -
                sum([ abs(DELTA[k]) * (sum([X[i]*alpha[i,k] for i in N])) for k in N if DELTA[k] > 0]) +
                sum([ DELTA[k] * X[i]*(1-k/n) for i in N for k in N]) ) 
    for i in N:
        for k in N:
            m.addConstr(u[i,k] - v[i,k] + gamma[k] == X[i])
    for k in N:
        m.addConstr(sum([alpha[i,k] for i in N]) == 0)

    # Function type
    if which_rho == "squaredev":        
        def rho(X,theta_var):
            return (1/n)*sum( [ (X[i]-theta_var)**2 for i in N ] )
        m.setObjective( rho(X, theta), GRB.MINIMIZE)
        m.optimize()
        value = m.objVal
        objbound = np.nan
    if which_rho == "absdev":
        r     = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="r")
        s     = m.addVars(n, lb=0, ub=GRB.INFINITY, name="s")
        for i in range(n):
            m.addConstr( r[i] == X[i] - theta )
            m.addGenConstrAbs(s[i], r[i])
        m.setObjective( (1/n)*sum([ s[i] for i in N ]), GRB.MINIMIZE)
        m.optimize()
        value    = m.objVal
        objbound = m.ObjBound
    if which_rho == "skew":
        x = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="x")
        xx = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="xx")
        y = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="y")
        z = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="z")
        yy = {}
        yyy = {}
        for i in N:
            yy[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="yy_"+str(i))
            yyy[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="yyy_"+str(i))
        for i in N:
            m.addConstr( yy[i] == (X[i]-theta)**2 )
            m.addConstr( yyy[i] == yy[i]*(X[i]-theta) )
        m.addConstr(x == (1/n)*sum( [ yy[i] for i in N ] ))
        m.addConstr(xx == x*x)
        m.addConstr(y*y==xx*x)
        m.addConstr(z*y == 1)
        m.setObjective( z*(1/n)*sum( [ yyy[i] for i in N ] ), GRB.MINIMIZE)
        m.optimize()
        value    = m.objVal
        objbound = np.nan
    if which_rho == "kurt":
        x  = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="x")
        xx = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="xx")
        y  = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="y")
        yy = {}
        yyyy = {}
        for i in N:
            yy[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="yy_"+str(i))
            yyyy[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="yyy_"+str(i))
        for i in N:
            m.addConstr( yy[i] == (X[i]-theta)**2 )
            m.addConstr( yyyy[i] == yy[i]*yy[i] )
        m.addConstr(x == (1/n)*sum( [ yy[i] for i in N ] ))
        m.addConstr(xx == x*x)
        m.addConstr(y*xx == 1)
        m.setObjective( y * (1/n) * sum( [ yyyy[i] for i in N ] ) -3, GRB.MINIMIZE) #1
        m.optimize()
        value = y.x * (1/n) * sum( [ yyyy[i].x for i in N ] ) -3
        objbound = np.nan

    done = time.time()
    elapsed = done - start

    # rl = m.relax()
    # rl.optimize()

    # Results:
    results = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    if m.Status == 3 or m.Status == 4:
        print("-- Model infeasible or unbounded")
        results[0]  = "nested"
        results[1]  = "2"
        results[2]  = n
        results[3]  = m.runtime #elapsed
        results[4]  = np.nan
        # results[5]  = np.nan
        results[6]  = X
        results[7]  = np.nan
        results[8]  = np.nan
        results[9]  = np.nan
        results[10] = np.nan
        results[11] = np.nan
        results[12] = elapsed
    else:
        if m.SolCount == 0:
            print("-- No solution found in time limit")
            results[0]  = "nested"
            results[1]  = "2"
            results[2]  = n
            results[3]  = m.runtime #elapsed
            results[4]  = np.nan
            # results[5]  = m.objBound
            results[6]  = X
            results[7]  = np.nan
            results[8]  = np.nan
            results[9]  = np.nan
            results[10] = np.nan
            results[11] = np.nan
            results[12] = elapsed
        else:
            results[0]  = "nested"
            results[1]  = "2"
            results[2]  = n
            results[3]  = m.runtime #elapsed
            results[4]  = value
            results[5]  = objbound 
            results[6]  = X
            results[7]  = {(i+1,k):alpha[i,k].x for i in N for k in N if DELTA[k] < 0 if alpha[i,k].x>0.5}  
            results[8]  = {(i+1,k):u[i,k].x for i in N for k in N if DELTA[k] > 0 if u[i,k].x>0.5}  
            results[9]  = {(i+1,k):v[i,k].x for i in N for k in N if DELTA[k] > 0 if v[i,k].x>0.5}  
            results[10] = {k:gamma[k].x for k in N if DELTA[k] > 0 if gamma[k].x>0.5}  
            results[11] = theta.x
            results[12] = elapsed

    m.dispose()

    return results

