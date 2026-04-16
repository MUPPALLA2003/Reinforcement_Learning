import torch

class OnlineMonteCarloLR():


    def __init__(self,env,gamma:float = 0.95):

        self.env = env
        self.gamma = gamma
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n


    def sample_trajectory(self,policy,max_steps:int = 50,epsilon:float = 0.1):

        state,_ = self.env.reset()
        done = False
        trajectory = []
        num_steps = 0

        while not done:

            if torch.rand(1).item() < epsilon:

                action = self.env.action_space.sample()

            else:

                action = policy[state].item()

            next_state,reward,done,_,_ = self.env.step(action)
            experience = (state,action,reward,next_state,done)
            trajectory.append(experience)
            num_steps += 1

            if num_steps >= max_steps:

                done = True
                break

            state = next_state

        return trajectory


    def compute_returns(self,trajectory):

        returns = {}
        G = 0

        for i in reversed(trajectory):

            state,action,reward,_,_ = i
            G = reward + self.gamma * G

            if (state,action) not in returns:

                returns[(state,action)] = G

        return returns
    
    def lr_scheduler(self,lr_start_value,lr_min_value,lr_decay_factor,num_episodes):

        alphas = [lr_start_value * lr_decay_factor ** episode for episode in range(num_episodes)]
        alphas = [a if a >= lr_min_value else lr_min_value for a in alphas]
        
        return alphas


    def monte_carlo_estimate_lr(self,policy,max_steps:int = 50,num_episodes:int = 10000,lr_start_value:float=0.8,lr_min_value:float=0.07,lr_decay_factor:float=0.99):

        Q = torch.zeros((self.n_states,self.n_actions))
        alphas = self.lr_scheduler(lr_start_value,lr_min_value,lr_decay_factor,num_episodes)

        for i in range(num_episodes):

            trajectory = self.sample_trajectory(policy,max_steps)
            returns_per_episode = self.compute_returns(trajectory)

            for (state,action),G in returns_per_episode.items():

                Q[state,action] = Q[state,action] + ((G - Q[state,action])*alphas[i])

        return Q        
 

    def policy_improvement(self,Q):

        return torch.argmax(Q,axis = -1)


    def policy_iteration(self,max_steps:int = 50,num_episodes:int = 10000):

        policy = torch.randint(0,self.n_actions,(self.n_states,),dtype = torch.long,device = self.device)

        while True:

            Q = self.monte_carlo_estimate_lr(policy,max_steps,num_episodes)
            new_policy = self.policy_improvement(Q)

            if torch.equal(new_policy, policy):

                break

            policy = new_policy

        return policy,Q
    

    def test_policy(self,policy,num_episodes):

        success_count = 0

        for _  in range(num_episodes):

            state,_ = self.env.reset()
            done = False

            while not done:

                action = policy[state].item()
                state,reward,done,_,_ = self.env.step(action)

                if done and reward == 1.0:

                    success_count += 1

        print(success_count/num_episodes)        



        

