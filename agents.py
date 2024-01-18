# an Agent class that has its own behavior (action), which is essentially a neural network that maps input to output defined by the env.
import random

import numpy as np



class Agent(object):
    def __init__(self, tag, location, score=0, input_dim=10, output_dim=10):
        self.tag = tag
        # initialize the brain to a random matrix in np, that maps input to output
        self.brain = np.random.rand(input_dim, output_dim)

        self.score = score # which is the resources the agent has accumulated.
        self.score_ranking = None

        self.location = location # the location of the agent in the grid.
        self.deltascore = 0 # the score added to the agent after each round.
        self.deltascore_ranking = None

        self.taken_grid = set([])
        self.taken_grid_ranking = None

    def reset(self):
        self.score = 0
        self.deltascore = 0
        self.taken_grid = set([])


    # given an input of np.array, using matrix multiplication to get an np.array with size output_size, then view it as a probability to sample from to get the argmax of the output.
    def action(self, input, output_size):
        output = np.matmul(input, self.brain)
        # output = [max(0.0001, x) for x in output]
        # output_prob = output / np.sum(output)
        return np.argmax(output) # FIXME: whether add randomness in this place.
        # return np.argmax(np.random.multinomial(10, output_prob, size=1)) # sample 10 times with the output prob and choose the argmax as the action.

    def mutate(self, num_mutations):
        # mutate the brain by changing num_mutations elements in the matrix to random values.
        for i in range(num_mutations):
            self.brain[random.randint(0, self.brain.shape[0]-1)][random.randint(0, self.brain.shape[1]-1)] = random.random()

    def breed(self, other_agent):
        # breed with another agent by averaging their brains. FIXME: or can knit both brains together?
        self.brain = (self.brain + other_agent.brain) / 2

    def __str__(self):
        return "Agent: " + self.tag + ", Score: " + str(self.score) + ", Brain: " + str(self.brain)

    def save_brain(self):
        np.save("./data/brain-rank-" + str(self.score_ranking), self.brain)


# a more complicated agent that has a more complicated brain with more layers.
class ComplicateAgent(Agent):
    def __init__(self, tag, location, score=0, input_dim=10, output_dim=10):
        super(ComplicateAgent, self).__init__(tag, location, score, input_dim, output_dim)
        self.brain1 = np.random.rand(input_dim, 20)
        self.brain2 = np.random.rand(20, 20)
        self.brain3 = np.random.rand(20, output_dim)


    def action(self, input, output_size):
        # do chain of multiplication with activation function tanh in between.
        output = np.matmul(input, self.brain1)
        output = np.tanh(output)
        output = np.matmul(output, self.brain2)
        output = np.tanh(output)
        output = np.matmul(output, self.brain3)
        output = np.tanh(output)
        return np.argmax(output)

    def mutate(self, num_mutations):
        # mutate the brain by changing num_mutations elements in the matrix to random values.
        for i in range(num_mutations):
            if random.random() < 0.33:
                self.brain1[random.randint(0, self.brain1.shape[0]-1)][random.randint(0, self.brain1.shape[1]-1)] = random.random()
            elif random.random() < 0.66:
                self.brain2[random.randint(0, self.brain2.shape[0]-1)][random.randint(0, self.brain2.shape[1]-1)] = random.random()
            else:
                self.brain3[random.randint(0, self.brain3.shape[0]-1)][random.randint(0, self.brain3.shape[1]-1)] = random.random()

    def breed(self, other_agent):
        # breed with another agent by subsitute random rows of each brain with the other brain.
        for i in range(self.brain1.shape[0]):
            if random.random() < 0.5:
                self.brain1[i] = other_agent.brain1[i]
        for i in range(self.brain2.shape[0]):
            if random.random() < 0.5:
                self.brain2[i] = other_agent.brain2[i]
        for i in range(self.brain3.shape[0]):
            if random.random() < 0.5:
                self.brain3[i] = other_agent.brain3[i]

    def __str__(self):
        return "Agent: " + self.tag + ", Score: " + str(self.score) + ", Brain1: " + str(self.brain1) + ", Brain2: " + str(self.brain2) + ", Brain3: " + str(self.brain3)


