import torch

class ValueIterationDet():

    def __init__(self,env,gamma:float = 0.95):

        self.env = env
        self.gamma = gamma
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_states = int(env.observation_space.n)
        self.n_actions = int(env.action_space.n)


    def bellman_optimality_eqn(self,s,a,v):

            Q_sa = 0

            for prob, s_next, reward, _ in self.env.unwrapped.P[s][a]:

                Q_sa += prob * (reward + self.gamma * v[s_next])

            return Q_sa       
    

    def value_iteration(self,max_iteration:int = 1000,tolerance:float = 1e-5):

        v = torch.zeros(self.n_states,dtype = torch.float32, device = self.device)
        
        for _ in range(max_iteration):

            delta = 0

            for s in range(self.n_states):

                V_copy = v[s].item()
                Q = []

                for a in range(self.n_actions):

                    Q_sa = self.bellman_optimality_eqn(s,a,v)
                    Q.append(Q_sa)

                v[s] = max(Q)                            
                delta = max(delta, abs(v[s].item() - V_copy))           

            if delta < tolerance:

                break

        return v  


    def policy_improvement(self):

        policy = torch.zeros(self.n_states,dtype = torch.long,device=self.device)
        V = self.value_iteration()

        for s in range(self.n_states):

            Q_sa = []

            for a in range(self.n_actions):

                Q = self.bellman_optimality_eqn(s,a,V)
                Q_sa.append(Q)

            best_action = torch.argmax(torch.tensor(Q_sa,dtype = torch.float32,device = self.device))
            policy[s] = best_action     

        return policy,V    