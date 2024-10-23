import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

import strategies

class SFEM():
    def __init__(self, data, data2):
        self.n_subjects = 1
        self.n_actions = 100
        
        self.actions=np.empty((self.n_subjects,self.n_actions))
        self.actions.fill(np.nan)
        self.matches=np.empty((self.n_subjects,self.n_actions))
        self.matches.fill(np.nan)
        self.periods=np.empty((self.n_subjects,self.n_actions))
        self.periods.fill(np.nan)
        self.others=np.empty((self.n_subjects,self.n_actions))
        self.others.fill(np.nan)

        i, a, m, p, o, n = self.parse_data(data,data2)

        self.actions[i,:n] = a
        self.matches[i,:n] = m
        self.periods[i,:n] = p
        self.others[i,:n]  = o    

        self.strats, self.strat_names = self.generate_strats()
        self.n_strats = len(self.strats)
        
    def parse_data(self, data, data2):
        # Get the data into matrix format 
        # for i,sub in enumerate(data.subject.unique()):
        i= 0
        a = ["<ans>COOPERATE</ans>" in x for x in data.response.tolist()]
        # m = data.supergame[data.subject==sub].tolist()
        # p = data.period[data.subject==sub].tolist()
        m = 1
        p = 100
        o = ["<ans>COOPERATE</ans>" in x for x in data2.response.tolist()]

        n = len(a)

    def generate_strats(self):
        strats = []
        strat_names = []
        for i in dir(strategies):
            s = getattr(strategies,i)
            if callable(s):
                strats.append(s)
                strat_names.append(s.__name__)
        n_strats = len(strats)
        print("There are",n_strats,'strategies in the strategies.py file. \nThe strategies are:',strat_names)
        
        return strats, strat_names
    
    def compare_strat(self):
        # For each subject n and each strategy k compare subject n's actual play with how strategy k would have played.
        C = np.zeros((self.n_strats,self.n_subjects)) #Number of periods in which play matches
        E = np.zeros((self.n_strats,self.n_subjects)) #Number of periods in which play does not match
        for n in range(self.n_subjects):
            for k in range(self.n_strats): 

                subChoice = self.actions[n]
                otherChoice = self.others[n]
                periodData = self.periods[n]

                stratChoice = self.strats[k](otherChoice,periodData)

                C[k,n]=np.sum(subChoice==stratChoice)
                E[k,n]=np.sum((1-subChoice)==stratChoice)
                
        return C, E
                
    def solver_run(self):
        # Likelhood function takes as an input a vector of proportions of strategies and returns the likelihood value
        #Note cMat and eMat are global matrices that are updated externally for each treatment.
        def objective(x,args):
            
            C = args[0]
            E = args[1]
            
            bc=np.power(x[0],C) #beta to the power of C
            be=np.power(1-x[0],E) #beta to the power of E
            prodBce = np.multiply(bc,be) #Hadamard product
            
            #maximum is taken so that there is no log(0) warning/error
            res = np.log(np.maximum(np.dot(x[1:],prodBce),np.nextafter(0,1))).sum() 
            
            return -res

        def constraint1(x):
            
            return x[1:].sum()-1

        C, E = self.compare_strat()
        #Set up the boundaries and constraints
        b0 = (np.nextafter(0.5,1),1-np.nextafter(0,1))
        b1 = (np.nextafter(0,1),1-np.nextafter(0,1))
        bnds = tuple([b0]+[b1]*self.n_strats) #Beta is at least .5
        con1 = {'type': 'eq', 'fun': constraint1} 
        cons = ([con1])

        #Some random starting point
        x0 = np.zeros(self.n_strats+1)
        x0[0] = .5+.5*np.random.random()
        temp = np.random.random(self.n_strats)
        x0[1:]=temp/temp.sum()

        bestX=x0
        bestObjective=objective(x0,[C,E])

        for k in range(50): #Do many times so that there is low chance of getting stuck in local optimum

            x0 = np.zeros(self.n_strats+1)
            x0[0] = .5+.5*np.random.random()
            temp = np.random.random(self.n_strats)
            x0[1:]=temp/temp.sum()

            #Notice that we are minimizing the negative
            solution = minimize(objective,x0,method='SLSQP',bounds=bnds,constraints=cons,args=([C,E]))
            x = solution.x
            obj = solution.fun
            check = solution.success #solution.success checks whether optimizer exited succesfully

            if bestObjective>obj and check: 
                bestObjective=obj
                bestX=x
                
        return bestX, bestObjective
    
    def save_results(self, path=None):   
        bestX, bestObjective = self.solver_run()
        results=pd.DataFrame(bestX.round(4).tolist()+[np.round(-bestObjective,4)],index=['beta']+self.strat_names+['LL'])
        results=results.rename(columns={0:'Estimates'})
        results=results.sort_values(by=['Estimates'],ascending=False)
        if path is not None:
            print(results)
            results.to_csv(os.path.join(path, "sfem_results.csv"))
        
        # return bestX.round(4).tolist()+[np.round(-bestObjective,4)]
        return results