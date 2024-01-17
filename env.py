# this is a project to test genetic algorithm/ evolution algorithms in a grid enviornment with agents conquering
# lands and try to get the most land. This is the enviornment (grid) that the agents will be in.
import math
import random
from matplotlib import pyplot as plt

from agents import Agent
from selection import SelectionPool


# The grid is a 2D array of GridEntry objects. Each GridEntry object has a resource amount and an agent tag/strength.
class GridEntry(object):
    def __init__(self, resource, initial_wall,x,y):
        self.resource = resource #TODO: refractor to resource_richness, and change agent.score to resource.
        self.agent_tag = "_ENV_"  # the agent conquering this grid.
        self.agent_strength = 1 # the strength of the agent conquering this grid.
        self.x = x
        self.y = y

    def conquer(self, agent_tag, strength=1):
        if (self.agent_strength > strength):
            self.agent_strength -= strength
            return None
        elif (self.agent_strength == strength):
            original_agent_tag = self.agent_tag
            self.agent_tag = "_ENV_"
            self.agent_strength = 0
            return original_agent_tag
        else:
            original_agent_tag = self.agent_tag
            self.agent_tag = agent_tag
            self.agent_strength = strength - self.agent_strength
            return original_agent_tag

    def reinforce(self, agent_tag, strength=1):
        if (self.agent_tag == agent_tag):
            self.agent_strength += strength
        else:
            self.agent_tag = agent_tag
            self.agent_strength = strength

    def drop(self, agent_tag, strength=1): #return the original agent_tag if the a new agent conquers it, else return None
        if (self.agent_tag == agent_tag):
            self.reinforce(agent_tag, strength)
            return None
        else:
            return self.conquer(agent_tag, strength)

    def __str__(self):
        return f"[{self.x},{self.y}]" + "Resource: " + str(self.resource) + ", Agent: " + self.agent_tag + ", Strength: " + str(self.agent_strength)

    # def __int__(self):
    #     return self.agent_strength

class Env(object):
    def __init__(self, config):
        self.config = config
        self.grid_size = config.grid_size
        self.grid = self.reset_grid(config.grid_size)
        self.agent_rank = []
        agents_pool = self.generate_first_gen_agents(config.num_agents)
        self.agents_pool = SelectionPool(agents_pool,config)  # manage evolution for all the agents in the pool
        self.agent_dict = self.agents_pool.create_dict()  # a dictionary with key=agent_tag and value=agent
        self.action_translate = {0: "drop", 1: "move_left", 2: "move_right", 3: "move_up", 4: "move_down"}
        self.simulate_time = None


        self.fig, self.ax = plt.subplots(figsize=(3,3))
        self.im = None
    def reset_grid(self, grid_size):  # generate a grid of size grid_size, with random distributed resources
        grid = []
        for i in range(grid_size):
            grid.append([])
            for j in range(grid_size):
                rsrc = random.randint(self.config.min_resource, self.config.max_resource)
                grid[i].append(GridEntry(rsrc, (rsrc * 0.1),i,j))
        return grid

    def generate_first_gen_agents(self, num_agents):
        # generate a list of agents with same init strength and different tags (Strings from "A" to "Z" to "AA"...)
        name = "A"
        agents_pool = []
        for i in range(num_agents):
            agents_pool.append(Agent(name,location= [random.randint(0,self.grid_size-1),random.randint(0,self.grid_size-1)],input_dim=self.config.agent_input_dim,output_dim=self.config.agent_output_dim))

            name = self.next_name(name)
        return agents_pool

    # generate a sequence of names for agents, starting from "A" to "Z" to "AA"...
    def next_name(self, name):
        if(len(name) == 1):
            if(name[0] == "Z"):
                return "AA"
            return chr(ord(name[0]) + 1)

        if (name[-1] == "Z"):
            return self.next_name(name[:-1]) + "A"
        else:
            return name[:-1] + chr(ord(name[-1]) + 1)

    def simulate(self):
        self.grid = self.reset_grid(self.grid_size)
        self.agent_dict = self.agents_pool.create_dict()  # create a dictionary with key=agent_tag and value=agent
        self.simulate_time = -self.config.simulation_time
        while(self.simulate_time < self.config.simulation_time):
            self.simulate_time += 1
            self.update_agent_information() # update the agent's global input information
            self.add_score()

            # for each round, each agent will take an action, and then the env will update the grid and the agents' scores.
            for agent in self.agents_pool:
                action_input = self.get_agent_action_input(agent)
                action = agent.action(action_input, self.config.agent_output_dim)
                self.update_grid(agent, self.action_translate[action])
            self.visual_grid()
        self.print_scores()
        self.agents_pool.next_generation()


    def get_agent_action_input(self, agent):
        # the input contains information described in README.md, [this_grid, left_grid, right_grid, up_grid, down_grid, timer, agent_strength, rankings]
        action = []
        # append the current grid's resource , conquering_agent's ranking, and strength.
        self.append_grid_entry_info_to_action(agent.location[0],agent.location[1], action,agent)
        # append the other 4 grids' info to action
        self.append_grid_entry_info_to_action(agent.location[0],(agent.location[1]-1)%self.grid_size, action,agent)
        self.append_grid_entry_info_to_action(agent.location[0],(agent.location[1]+1)%self.grid_size, action,agent)
        self.append_grid_entry_info_to_action((agent.location[0]-1)%self.grid_size,agent.location[1], action,agent)
        self.append_grid_entry_info_to_action((agent.location[0]+1)%self.grid_size,agent.location[1], action,agent)

        # append global info to action
        action.append(self.simulate_time)
        action.append(agent.score)
        action.append(agent.score_ranking)
        action.append(agent.deltascore)
        action.append(agent.deltascore_ranking)
        action.append(len(agent.taken_grid))
        action.append(agent.taken_grid_ranking)
        return action

    def append_grid_entry_info_to_action(self, pos0,pos1, action,agent):
        grid_entry = self.grid[pos0][pos1]
        # special case for "ENV" agent
        if (grid_entry.agent_tag == "_ENV_"):
            action.append(grid_entry.resource)
            action.append(grid_entry.agent_strength)
            action.append(self.config.num_agents + 1) # always the top ranking
            return

        grid_agent = self.agent_dict[grid_entry.agent_tag]
        action.append(grid_entry.resource)
        if (grid_agent.tag == agent.tag):
            action.append(grid_entry.agent_strength)
            action.append(0)
        else:
            action.append(-grid_entry.agent_strength)
            action.append(grid_agent.score_ranking)

    def update_grid(self, agent, action):
        # update the grid based on the action of the agent
        if (action == "drop"):
            agent.score -= 1 #dropped on the grid
            old_conquerer_tag = self.grid[agent.location[0]][agent.location[1]].drop(agent.tag)
            if(old_conquerer_tag != None):
                agent.taken_grid.append(self.grid[agent.location[0]][agent.location[1]])
                if(old_conquerer_tag != "_ENV_"): # not newly conquered
                    try:
                        self.agent_dict[old_conquerer_tag].taken_grid.remove(self.grid[agent.location[0]][agent.location[1]])
                    except:
                        print("Error: agent_dict does not have key: " + old_conquerer_tag)
        elif (action == "move_left"):
            agent.location[1] = (agent.location[1] - 1) % self.grid_size
        # generate elif in same fashion for other three directions: move_right, move_up, move_down
        elif (action == "move_right"):
            agent.location[1] = (agent.location[1] + 1) % self.grid_size
        elif (action == "move_up"):
            agent.location[0] = (agent.location[0] - 1) % self.grid_size
        elif (action == "move_down"):
            agent.location[0] = (agent.location[0] + 1) % self.grid_size


    def add_score(self):
        for agent in self.agents_pool:
            agent.score += agent.deltascore

    # calculate the global info for each agent, including deltascore, resource_ranking, deltascore_ranking, taken_grid_ranking by calling functions from selection.py
    def update_agent_information(self):
        # update deltascore
        self.agents_pool.clear_deltascore()
        for agent in self.agents_pool:
            for taken_grid in agent.taken_grid:
                agent.deltascore += self.get_score(taken_grid.resource, taken_grid.agent_strength)
        # update rankings
        self.agents_pool.update_rankings()

    def get_score(self, resource, agent_strength):
        # generate a logistic-like function that maps resource and agent_strength to a score, with dimishing returns only on agent_strength.
        return resource * 1 / (1 + math.exp(-1 * (agent_strength)))

    def print_rankings(self):
        for agent in self.agents_pool:
            print(agent.tag + ": " + str(agent.score_ranking) + ", " + str(agent.deltascore_ranking) + ", " + str(agent.taken_grid_ranking))
    def print_scores(self):
        for agent in self.agents_pool:
            print("Score: " + str(agent.score) + ",\tIncrement: " + str(agent.deltascore) + "\tTaken_grid:" + str(len(agent.taken_grid)))
        #get average score and increment for all agents in the pool
        average_score = sum([agent.score for agent in self.agents_pool])/len(self.agents_pool.agents)
        average_increment = sum([agent.deltascore for agent in self.agents_pool])/len(self.agents_pool.agents)
        print("Average Score: " + str(average_score) + ",\tAverage Increment: " + str(average_increment))
    def visual_grid(self):
        # visualize the grid using matplotlib with animation
        if self.im is None:
            self.im = self.ax.imshow([[self.grid[i][j].agent_strength for j in range(self.grid_size)] for i in range(self.grid_size)], cmap='hot', interpolation='nearest')
            plt.colorbar(self.im, ax=self.ax)
        else:
            self.im.set_data([[self.grid[i][j].agent_strength for j in range(self.grid_size)] for i in range(self.grid_size)])
            self.im.set_clim( vmin=0, vmax=max([self.grid[i][j].agent_strength for j in range(self.grid_size) for i in range(self.grid_size)]))
        self.ax.set_title("Resource Distribution")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        plt.draw()
        plt.pause(0.00001)  # Pause to allow the plot to update
