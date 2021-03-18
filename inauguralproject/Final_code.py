from scipy import optimize
#Solve consumer problem
# Defining parameters and entering them into a dictionary
# Defining function
params={}
params1={}
def solve_func(m,  par_dict):
    """ This function solves the consumer problem for given parameter values and income
    Args:
         m: Income
         par_dict: Dictionary with parameter values

    Returns:
        Consumption and housing levels that solve the problem for housing equals its price.

    """

    phi=par_dict['phi']
    tg=par_dict['tg']
    tp=par_dict['tp']
    pbar=par_dict['pbar']
    r=par_dict['r']
    epsilon=par_dict['epsilon']
    #Creating utility function:
    def utility_function(c,ph):
        utility =c**(1-phi)*ph**(phi)
        return utility
    #Creating cost-function 
    def cost_func(ph):
        cost=r*ph+tg*epsilon*ph+tp*max(ph*epsilon-pbar,0)
        return cost
#     Creating constraint - x is a vector with x=(c,ph)
    constraints = ({'type': 'eq', 'fun': lambda x: m- x[0]-cost_func(x[1])})
#  Defining bounds - maximum are maximal consumption levels - upper bound for housing is set in the case where the price is as low as possible
#  thus when p_tilde is smaller than pbar. 
    bounds=[(0,m), (0,(m+tp*pbar)/(r+tg*epsilon))]
# Feasible initial guess - price of housing is always lower than 1
    initial_guess=(m*(1-phi),m*phi )
# Optimizing function
    res=optimize.minimize(
        lambda x: -utility_function(x[0], x[1]), initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    solution=res.x
    return solution


#Calculate average tax revenue
import numpy as np
def calc_rev(mean,var, N, par_dict):
    """ This function calculates average tax revenue for a given distribution - log-normal - as a function of poarameters
    distribution mean, distribution variance, number of draws and given parameters
    Args:
         mean: Distribution mean
         par_dict: Dictionary with parameter values
         var: Distribution variance
         N: Number of draws


    Returns:
        Average tax revenue

    """
    #Set seed
    np.random.seed(1)
    #Define parameteres
    tg=par_dict["tg"]
    epsilon=par_dict["epsilon"]
    tp=par_dict["tp"]
    pbar=par_dict["pbar"]
    #Draw M
    m_draw = np.random.lognormal(mean, var, N)
    #Find solutions for draws of m
    solutions=np.array([solve_func(m, par_dict) for m in m_draw])
    #Pull all solutions for h and c into diffeent np.arrays that can be summed over
    unwrap = np.array([np.array([solution[i] for solution in solutions[:][:]]) for i in range(0,2)])
    #Calculate each element in the sum of tax revenue
    T_aux=np.array([tg*epsilon*h+tp*max(epsilon*h-pbar,0) for h in unwrap[1]])
    #T_aux=tg*epsilon*h+tp*fmax(epsilon*h-pbar,0)

    #calculate revenue
    T=np.sum(T_aux)
    #Calculate average revenue
    average_revenue=T/N
    print(solutions)
    print(unwrap)
    return average_revenue



#Redefine average tax revenue, so that the general tax rate is not set from the dictionary
import numpy as np
def calc_rev_5(tg_set, par_dict, draw, N):
    """ This function calculates average tax revenue for a given distribution - log-normal - as a function of poarameters
    distribution mean, distribution variance, number of draws and given parameters
    Args:
         mean: Distribution mean
         par_dict: Dictionary with parameter values
         var: Distribution variance
         N: Number of draws
         tg: level of base housing tax - set separately from other parameters


    Returns:
        Average tax revenue

    """
    #Set base tax rate
    tg=tg_set
    #set other parameters
    epsilon=par_dict["epsilon"]
    tp=par_dict["tp"]
    pbar=par_dict["pbar"]
    #Find solutions
    solutions=np.array([solve_func(m, par_dict) for m in draw])
    #Move all c and all h into their own arrays -.
    unwrap = np.array([np.array([solution[i] for solution in solutions[:][:]]) for i in range(0,2)])
    #Calculate each element in the sum. 
    T_aux=np.array([tg*epsilon*h+tp*max(epsilon*h-pbar,0) for h in unwrap[1]])
    T=np.sum(T_aux)
    average_revenue=T/N
    return average_revenue

#Define value objective as the squared difference between tax revenues




#Create a function that gets equal average tax revenue from different parameter values. 
def equal_value_solution(par_dict1=params ,par_dict2=params1 , mean=-0.4, variance=0.35,N=1000):
    """ This function sets average tax revenue equal under different sets of parameters by changing the base tax rate
    Args:
         mean: Distribution mean
         par_dict1: Original parameters - will be fixed after calculation
         par_dict2: New parameters from which a new base tax rate is calculated
         var: Distribution variance
         N: Number of draws
         tg: level of base housing tax - set separately from other parameters


    Returns:
        Tax rate required for different parameters to give the same average tax revenue.

    """
    np.random.seed(1)

    m_draw = np.random.lognormal(mean, variance, N)

    #Set parameters
    tg_set=par_dict1["tg"]
    original_rev=calc_rev_5(tg_set,par_dict1, m_draw, N)
    print(f'Tax revenue under the first set of parameters: {original_rev}')

    def equal_value_objective(tg_set):
        value_of_choice=(calc_rev_5(tg_set, par_dict2, m_draw, N)-original_rev)**2
        return value_of_choice        
    solution_g=optimize.minimize_scalar(equal_value_objective,method='bounded', bounds=(0,1))
    tg_final=solution_g.x
#    print(solution_g)
    print(f'Tax revenue under new set of parameters with solution base tax rate {calc_rev_5(tg_final, par_dict2, m_draw, N)}')
    print(f' Solution tax rate {tg_final}')
    return tg_final



def solution_with_root(par_dict1=params ,par_dict2=params1 , mean=-0.4, variance=0.35,N=1000):
    """ This function sets average tax revenue equal under different sets of parameters by changing the base tax rate
    Args:
         mean: Distribution mean
         par_dict1: Original parameters - will be fixed after calculation
         par_dict2: New parameters from which a new base tax rate is calculated
         var: Distribution variance
         N: Number of draws
         tg: level of base housing tax - set separately from other parameters


    Returns:
        Tax rate required for different parameters to give the same average tax revenue.

    """ 
    np.random.seed(1)
    m_draw = np.random.lognormal(mean, variance, N)
        #Set parameters
    tg_set=par_dict1["tg"]
    original_rev=calc_rev_5(tg_set,par_dict1, m_draw,N)
    print(f'Tax revenue under the first set of parameters: {original_rev}')
    def equal_value_objective(tg_set):
        value_of_choice=(calc_rev_5(tg_set, par_dict2, m_draw, N)-original_rev)**2
        return value_of_choice        
    solution_g=optimize.root_scalar(equal_value_objective, method='bisect', bracket=(0,1))
    tg_final=solution_g.root
#    print(solution_g)
    print(f'Tax revenue under new set of parameters with solution base tax rate {calc_rev_5(tg_final,par_dict2, m_draw,N)}')
    print(f' Solution tax rate {tg_final}')
    return tg_final