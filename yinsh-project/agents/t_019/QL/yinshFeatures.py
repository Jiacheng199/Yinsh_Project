import copy
from turtle import color
import numpy as np
from sympy import sequence
from Yinsh.yinsh_model import YinshGameRule

class YinshFeature():

    def __init__(self):
        self.game_rule = YinshGameRule(2)

    # this is the f(s,a) will be used in linear q learning
    def createFeatures(self,game_state,action,id):
        copyState = copy.deepcopy(game_state)

        #the new State that if excute the given action
        nextState = self.game_rule.generateSuccessor(copyState,action,id)
        
        #represent features value as a dic
        feature_value = {}

        #opponent id
        oid = 1 - id
        
        #How many yinsh can this move generate
        #feature_value["newYinsh"] = self.yinshCount(action,copyState,id)/2

        #How many nodes from opponent on the board
        feature_value["oppoNode"] = self.nodeCount(nextState,oid)/51

        #How many nodes from agent on the board
        feature_value["agentNode"] = self.nodeCount(nextState,id)/51

        #How many could be turn to your color
        feature_value["jumpOverGood"] = (self.nodeCount(nextState,id) - self.nodeCount(game_state,id))/7

        #How many could be turn to your opponent's color
        feature_value["jumpOverBad"] = (self.nodeCount(nextState,oid) - self.nodeCount(game_state,oid))/7

        #The rings opponent had gained
        feature_value["oppoRings"] = nextState.rings_won[oid]/3

        #The rings agent had gained
        feature_value["gainedRing"] = nextState.rings_won[id]/3

        #Features for rings only(rings placing stage)
        #feature_value["rings"] = self.ringDistance(nextState,id)/50

        return list(feature_value.values())

    def yinshCount(self,action,old_board,id):
        if action["type"] == "place and move":
            possible_seq,_ = self.game_rule.sequenceCheck(old_board,[action["place pos"]])
            selfYinsh = 0
            opponentYinsh = 0

            if possible_seq[id] != None:
                selfYinsh = 1
            if possible_seq[1 - id] != None:
                opponentYinsh = 1

            return selfYinsh - opponentYinsh
        else:
            return 0

    def ringDistance(self,game_state,id):
        rings = game_state.ring_pos[id]
        ringList = []
        for r in rings:
            (x,y) = r
            ringList.append([x,y])
        distance = 0
        npa = np.array(ringList)
        for i in range(len(ringList)):
            if i + 1 < len(ringList):
                distance += np.linalg.norm(npa[i]-npa[i+1])
        return distance

    #The many a node of a given player's color has been put on board
    def nodeCount(self,game_state,id):
        board = game_state.board

        # the color of this player using in game
        # color = board[(game_state.ring_pos[id])[0]] + 1
        if id == 0:
            color = 2
        else:
            color = 4
        #count the color 
        count = np.count_nonzero(board == color)
        
        return count
    
