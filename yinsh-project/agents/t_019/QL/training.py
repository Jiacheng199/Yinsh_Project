import copy
import numpy as np
from agents.t_019.QL.yinshFeatures import YinshFeature
#learning rate
LR = 0.1
#Future Discount
FD = 1
#Epsilon value
EV = 0.1
#Epsiodes
EP = 100

class WeightTraining:
    def __init__(self, mdp, qfunction):
        self.mdp = mdp
        self.qfunction = qfunction
        self.features = YinshFeature()

    #trainning stage
    def training(self, episodes=100):
        id = self.mdp.id
        oid = 1 - id

        for episode in range(episodes):
            print(episode)
            state1 = copy.deepcopy(self.mdp.game_rule.current_game_state)

            # select self action for this state
            self_actions = self.mdp.game_rule.getLegalActions(state1, id)
            self_action = self.selectAction(state1, self_actions, self.qfunction, id)
            self.features.createFeatures(state1,self_action,id)


            while True:

                # calculate q value for current state
                q_value = self.qfunction.qValue(state1, self_action, id)

                # execute self action
                state2, reward = self.mdp.execute(state1, self_action, id)
                
                #If reach goal state
                if self.mdp.gameEnd(state2):
                    #the actual optimal Q value here
                    self.qfunction.update(state1, self_action, LR * reward, id)
                    break;
                
                opactions = self.mdp.game_rule.getLegalActions(state2, oid)
                opponent_action = self.selectAction(state2, opactions, self.qfunction, oid)

                # execute opponent action
                state3, next_reward = self.mdp.execute(state2, opponent_action, oid)

                if self.mdp.gameEnd(state3):
                    break;
                else:
                    selfActions = self.mdp.game_rule.getLegalActions(state3, id)
                    self_next_action = self.selectAction(state3, selfActions, self.qfunction, id)
                    
                    self.qfunction.update(state1, self_action, self.get_delta((reward + next_reward), q_value, state3), id)

                    state1 = state3
                    self_action = self_next_action

            print(self.qfunction.weights)

    #MAB (eplison greedy)
    def selectAction(self,state,actions,qfunction,id):
        p = np.random.random()
        if p < EV:
          
          action = np.random.choice(actions)
        else:
          action,_  = qfunction.bestAction(state, actions, id)
        return action

    #The loss function, core part of RL learning
    def get_delta(self, rewards, q_value, next_state):
        #FD * (r + (q star value) - current q value)
        return LR * (rewards + FD * (self.state_value(next_state)) - q_value)

    def state_value(self, state):
        (_, mq) = self.qfunction.bestAction(state, self.mdp.game_rule.getLegalActions(state, self.mdp.id), self.mdp.id)
        return mq
