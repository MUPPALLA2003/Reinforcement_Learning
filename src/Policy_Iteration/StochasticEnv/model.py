import torch
class PolicyIterationStoc():

    def __init__(self,env,gamma = 0.95):

        self.env = env
        self.gamma = gamma
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_states = int(env.observation_space.n)
        self.n_actions = int(env.action_space.n)

    def bellman_equation(self,state,action,values):

        possible_actions = self.env.unwrapped.P[state][action]
        value_func = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        for prob,s_next,reward,flag in possible_actions:

            value_func += prob * (reward + self.gamma * values[s_next])

        return value_func


    def policy_evaluation(self,policy,max_iterations:int=1000,tolerence:float=1e-09):

        V = torch.zeros(self.n_states,dtype = torch.float32, device = self.device)

        for _ in range(max_iterations):

            delta = 0
            
            for s in range(self.n_states):

                V_copy = V[s].item()
                wanted_action = policy[s].item()
                V[s] = self.bellman_equation(s,wanted_action,V)
                delta = max(delta, abs(V[s].item() - V_copy))
    

            if delta < tolerence:
                break

        return V
    

    def policy_improvement(self,values):

        new_policy = torch.zeros(self.n_states,dtype = torch.long,device=self.device)

        for s in range(self.n_states):

            q_sa = []

            for wanted_action in range(self.n_actions):

                Q = self.bellman_equation(s, wanted_action, values)
                q_sa.append(Q)

            best_action = torch.argmax(torch.tensor(q_sa,dtype = torch.float32,device = self.device))
            new_policy[s] = best_action

        return new_policy
    

    def policy_iteration(self,max_iterations:int=1000):

        policy = torch.randint(0,self.n_actions,(self.n_states,),device = self.device)

        for _ in range(max_iterations):

            values = self.policy_evaluation(policy)
            new_policy = self.policy_improvement(values)

            if torch.equal(policy,new_policy):
                break   
            policy = new_policy

        return policy,values
    

        


        