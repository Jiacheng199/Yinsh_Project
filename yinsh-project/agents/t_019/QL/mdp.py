import copy
#A mdp enviroment for training
class MDP:
    def __init__(self,game_rule,id):
        self.game_rule = game_rule
        self.id = id
    
    #Execute a action in given game environment
    def execute(self,state,action,id):
        next_state = self.game_rule.generateSuccessor(copy.deepcopy(state), action, id)
        reward = self.getReward(next_state)
        return next_state, reward

    #Check if one plater wins
    def gameEnd(self,state):
        agent,opponent = state.agents[self.id],state.agents[(self.id + 1) % 2]
        if agent.passed or opponent.passed:
            return True
        elif agent.score >= 3 or opponent.score >= 3:
            return True
        return False

    #Reward from action to another state
    def getReward(self,state):
        player_score = state.rings_won[self.id] - state.rings_won[1 - self.id]
        return player_score 