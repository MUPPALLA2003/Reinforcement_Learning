import torch

class OfflineMonteCarlo():


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

            if torch.rand(1) < epsilon:

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


    def monte_carlo_estimate(self,policy,max_steps:int = 50,num_episodes:int = 5000):

        Q = torch.zeros((self.n_states,self.n_actions))
        returns = {(s,a):[] for s in range(self.n_states) for a in range(self.n_actions)}

        for _ in range(num_episodes):

            trajectory = self.sample_trajectory(policy,max_steps)
            returns_per_episode = self.compute_returns(trajectory)

            for (state,action),G in returns_per_episode.items():

                returns[(state,action)].append(G)

        for (state,action),returns_list in returns.items():

            if len(returns_list) > 0:

                Q[state,action] = torch.mean(torch.tensor(returns_list,dtype = torch.float32,device = self.device))

        return Q 
    

    def policy_improvement(self,Q):

        return torch.argmax(Q,axis = -1)


    def policy_iteration(self,max_steps:int = 50,num_episodes:int = 1000):

        policy = torch.randint(0,self.n_actions,(self.n_states,),dtype = torch.long,device = self.device)

        while True:

            Q = self.monte_carlo_estimate(policy,max_steps,num_episodes)
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



        

