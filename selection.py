# a Selection class that manipulate the survival and evolution of the agents based on their scores.
import copy
import random


class SelectionPool(object):
    def __init__(self,agent_list,config):
        self.agents = agent_list
        # get the next_generation proportion from the config
        self.num_survivors = config.num_survivors
        self.num_children = config.num_children
        self.num_breeds = config.num_breeds
        self.num_new_agents = config.num_new_agents
        self.num_mutations = config.num_mutations
        self.num_agents = config.num_agents # total number of agents in the pool

    def next_generation(self):
        # sort the agents by their scores
        self.agents.sort(key=lambda x: x.score, reverse=True)
        # get the top num_survivors agents
        survivors = self.agents[:self.num_survivors]
        # mutate the survivors by copying them for self.num_children amount and mutate the copyed version and added them to the next generation
        next_generation = survivors[:]
        for survivor in survivors:
            for i in range(self.num_children):
                next_generation.append(self.mutate_survivor(survivor))
            survivor.reset()
        # breed the survivors by combining random two's brains and added them to the next generation
        for i in range(self.num_breeds):
            next_generation.append(self.breed_survivors(random.choice(survivors), random.choice(survivors)))
        self.agents = next_generation

        return next_generation # return the next generation of agents for the env to use for plotting (not needed yet)

    def mutate_survivor(self, survivor):
        # copy the survivor and mutate the copy
        child = copy.deepcopy(survivor)
        child.mutate(self.num_mutations)
        child.tag = survivor.tag + '-' + str(random.randint(0, 99))
        while (child.tag in self.create_dict()):
            child.tag = survivor.tag + '-' + str(random.randint(0, 99))
        child.reset()
        return child

    def breed_survivors(self, survivor1, survivor2):
        # copy the survivor1 and breed it with survivor2
        child = copy.deepcopy(survivor1)
        child.breed(survivor2)

        child.tag = survivor1.tag + '&' + survivor2.tag + '-' + str(random.randint(0,99))
        while(child.tag in self.create_dict()):
            child.tag = survivor1.tag + '&' + survivor2.tag + '-' + str(random.randint(0,99))
        child.reset()
        return child

    def __iter__(self):
        return iter(self.agents)

    def create_dict(self):
        #return a dictionary with key=agent_tag and value=agent
        agent_dict = {}
        for agent in self.agents:
            agent_dict[agent.tag] = agent
        return agent_dict

    def clear_deltascore(self):
        for agent in self.agents:
            agent.deltascore = 0

    def update_score_ranking(self):
        self.agents.sort(key=lambda x: x.score, reverse=True)
        for i in range(len(self.agents)):
            self.agents[i].score_ranking = i+1

    def update_deltascore_ranking(self):
        self.agents.sort(key=lambda x: x.deltascore, reverse=True)
        for i in range(len(self.agents)):
            self.agents[i].deltascore_ranking = i+1

    def update_taken_grid_ranking(self):
        self.agents.sort(key=lambda x: len(x.taken_grid), reverse=True)
        for i in range(len(self.agents)):
            self.agents[i].taken_grid_ranking = i+1

    def update_rankings(self):
        self.update_score_ranking()
        self.update_deltascore_ranking()
        self.update_taken_grid_ranking()



