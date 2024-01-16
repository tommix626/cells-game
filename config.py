#generate a config class that contains baisc configuration for the grid and simulation

class Config(object):
    def __init__(self):
        self.grid_size = 100
        self.num_agents = 100
        self.agent_input_dim = 22
        self.agent_output_dim = 5
        self.min_resource = 1
        self.max_resource = 10
        self.simulation_time = 100 #time goes from -100 to 100.
        self.num_generations = 10

        # only the top 20% of the agents survive to the next generation, and they will have 2 children each, and 30% of the next generation will be bred from the survivors, and 10% of the next generation will be randomly generated.
        self.num_survivors = int(self.num_agents * 0.2)  # number of agents that survive to the next generation
        self.num_children = 2 # number of children each survivor has other than themselves
        self.num_breeds = int(self.num_agents * 0.3) # number of agents in the next round that combines two survivors' brains
        self.num_new_agents = self.num_agents - self.num_survivors * (1+self.num_children) - self.num_breeds # number of agents that are newly (randomly) generated in the next generation

        self.num_mutations = 10  # number of mutations each offspring has (unit: elements in matrix mutated)