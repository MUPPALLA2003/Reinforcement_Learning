import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from utils.seed import set_seed

set_seed(42)

class Policy_Iteration_Det():

    def __init__(self,env,mdp,gamma:float = 0.95):

        self.env = env
        self.mdp = mdp
        self.gamma = gamma
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_states = int(env.observation_space.n)
        self.n_actions = int(env.action_space.n)

  
    def bellman_equation(self,s,a,values):
        prob,s_new,reward,_ = self.env.unwrapped.P[s][a][0]
        return prob * (reward + self.gamma * values[s_new])


    def policy_evaluation(self,policy,max_iterations:int=1000,tolerence:float=1e-09):

        V = torch.zeros(self.n_states,dtype = torch.float32, device = self.device)

        for _ in range(max_iterations):

            param = 0
            
            for s in range(self.n_states):

                V_copy = V[s].item()
                action = policy[s].item()
                V[s] = self.bellman_equation(s,action,V)
                param = max(param, abs(V[s].item() - V_copy))

            if param < tolerence:
                break

        return V
    

    def policy_improvement(self,values):

        new_policy = torch.zeros(self.n_states,dtype = torch.long,device=self.device)

        for s in range(self.n_states):

            q_sa = []

            for a in range(self.n_actions):

                Q = self.bellman_equation(s, a, values)
                q_sa.append(Q)

            best_action = torch.argmax(torch.tensor(q_sa,dtype = torch.float32,device = self.device))
            new_policy[s] = best_action

        return new_policy

    def policy_iteration(self,max_iterations:int=1000):

        policy = torch.randint(0,self.n_actions.n,(self.n_states,),device = self.device)

        for _ in range(max_iterations):

            values = self.policy_evaluation(policy)
            new_policy = self.policy_improvement(values)

            if torch.equal(policy,new_policy):
                break   
            policy = new_policy

        return policy,values