from collections import deque, namedtuple
from operator import ipow
import copy
import random
from re import A

from Yinsh.yinsh_model import YinshGameRule 
from template import Agent
import numpy as np
from agents.t_019.QL.training import WeightTraining
from agents.t_019.QL.mdp import MDP
from agents.t_019.QL.yinshFeatures import YinshFeature
#weight table
#W = [-0.394693664865715, -0.7375456738492496, -0.6272467558733855, -0.8220544665634474, -5.430342282926175, 4.3246212694846635]
W = [34.507178352982265, 42.624193162569526, 1.482035810598033, 6.90188646932196, -21.322422406337466, 92.80774626192247]
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
            self.weights = [
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

        max_q = float("-inf")
        best_actions = []

        for action in actions:
            value = self.qValue(state, action, id)
            if max_q < value:
                best_actions.clear()
                best_actions.append(action)
                max_q = value
            elif (abs(max_q - value) < 1e-6):
                best_actions.append(action)
        return (best_actions[0], max_q)