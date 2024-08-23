from collections import deque, namedtuple
from operator import ipow
import copy
import random
from Yinsh.yinsh_model import YinshGameRule 
from template import Agent
import numpy as np
from agents.t_019.QL.training import WeightTraining
from agents.t_019.QL.mdp import MDP
from agents.t_019.QL.yinshFeatures import YinshFeature
#Action weight table
actionWeight = {}
#Learning Rate
LR = 0.01
#Discount Reward
DR = 0.9

W = [-0.394693664865715, -0.7375456738492496, -0.6272467558733855, -0.8220544665634474, -5.430342282926175, 4.3246212694846635]
class myAgent(Agent):

    def __init__(self,_id):
        super().__init__(_id)
        self.game_rule = YinshGameRule(2)
        self.qfunction = QFunction(6,W)

    def SelectAction(self, actions, game_state):

        copyState = copy.deepcopy(game_state)
        action, _ = self.qfunction.bestAction(copyState, actions, self.id)

        return action


class QFunction:

    def __init__(self, featureCount, weights=None, default=0.0):
        if weights == None:
            self.weights = [ # idea from notebook in lecture
                default
                for _ in range(0, featureCount)
            ]
        else:
            self.weights = weights

    #Update the weight table
    def update(self, state, action, delta, id):
        feature_values = YinshFeature().createFeatures(state,action,id)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + (delta * feature_values[i])

    #Calcuate the q value of a state-action pair only beased on weight and feature value
    def qValue(self, state, action, id):
        q = 0.0
        feature_values = YinshFeature().createFeatures(state,action,id)

        for i in range(len(feature_values)):
            q += feature_values[i] * self.weights[i]
        return q

    #Get the action that can generate max q value
    def bestAction(self, state, actions, id):

        bestAction = [0]
        value = float("-inf")
        for action in actions:
            if self.qValue(state,action,id) > value:
                value = self.qValue(state,action,id)
                bestAction[0] = action

        return (bestAction[0],value)
