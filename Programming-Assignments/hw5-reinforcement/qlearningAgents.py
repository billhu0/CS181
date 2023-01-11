# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random
import util
import math
import gridworld  # for typing alias
from typing import Dict, List, Tuple, Union, Optional  # for typing alias
import pacman  # for typing alias
import game  # for typing alias


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args) -> None:
        """You can initialize Q-values here..."""
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.q_values: Dict = util.Counter()

    def getQValue(self, state: gridworld.Gridworld, action: str) -> float:
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # Search in our dict.
        # If no such key, the util.Counter will automatically return zero.
        return self.q_values[(state, action)]

    def computeValueFromQValues(self, state: gridworld.Gridworld) -> float:
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # For all legal actions, pick the one that maximizes the Q-value, and returns the Q-value.
        return max(map(lambda action: self.getQValue(state, action), self.getLegalActions(state)), default=0.0)

    def computeActionFromQValues(self, state) -> str:
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # For all actions, pick the one that maximizes the Q-value. Return the action.
        return max(self.getLegalActions(state), key=lambda a: self.getQValue(state, a), default=None)
        # util.raiseNotDefined()

    def getAction(self, state: gridworld.Gridworld) -> str:
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions: List[str] = self.getLegalActions(state)
        action: str = None
        "*** YOUR CODE HERE ***"

        # exploration with probability self.epsilon
        if util.flipCoin(self.epsilon):
            # randomly act to explore more
            action = random.choice(self.getLegalActions(state))

        # exploitation with probability (1-epsilon)
        else:
            # based on current knowledge, return an action that maximizes the q-value
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action: str, nextState, reward: float) -> None:
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        s: float = reward + self.discount * self.computeValueFromQValues(nextState)

        # alpha is the learning rate.
        # We combine the old knowledge with this newly-added value.
        self.q_values[(state, action)] = ((1 - self.alpha) * self.getQValue(state, action) + self.alpha * s)

    def getPolicy(self, state) -> str:
        return self.computeActionFromQValues(state)

    def getValue(self, state) -> float:
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    """Exactly the same as QLearningAgent, but with different default parameters"""

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features: List = self.featExtractor.getFeatures(state, action)
        return sum(map(lambda f: self.weights[f] * features[f], features.keys()))
        # util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        delta: float = reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)
        features: List = self.featExtractor.getFeatures(state, action)

        # update weight for each feature
        for feature in features:
            # adjust the weight by alpha (learning rate) and delta
            self.weights[feature] = self.weights[feature] + self.alpha * delta * features[feature]

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass


# Returns a ghost's next position
def getGhostPosition(ghostState: game.AgentState) -> Tuple[int, int]:
    x, y = ghostState.getPosition()
    dx, dy = Actions.directionToVector(ghostState.getDirection())
    return int(x + dx), int(y + dy)


# Returns the distance to the closest object
def distanceObject(pos: Tuple[int, int], obj, walls: game.Grid) -> Optional[float]:
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        if (pos_x, pos_y) in obj:
            return dist / (walls.width * walls.height)
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    return None


class BetterExtractor(FeatureExtractor):
    """Your extractor entry goes here.  Add features for capsuleClassic."""

    def getFeatures(self, state: pacman.GameState, action: str) -> util.Counter:
        """*** YOUR CODE HERE ***"""
        features: util.Counter = util.Counter()

        # A food grid. food[i][j] will return a bool indicating whether there is food at position i,j.
        food: game.Grid = state.getFood()
        walls: game.Grid = state.getWalls()  # Wall grid. Usage similar to above.
        ghostStates: List[game.AgentState] = state.getGhostStates()  # All ghosts' states.

        # If we take the action, the next location of pacman
        next_pos: Tuple[int, int] = self.get_next_pacman_location(state.getPacmanPosition(), action)

        # Part 1. Strong ghosts

        # strong ghost one step away
        def strong_ghost_away(pac_pos: Tuple[int, int]) -> float:
            return sum(pac_pos in Actions.getLegalNeighbors(ghostState.getPosition(), walls)
                       for ghostState in self.get_strong_ghost_states(ghostStates))

        # strong ghost two steps away
        def strong_ghost_2_away() -> float:
            return sum(strong_ghost_away(n) for n in
                       [(next_pos[0] + 1, next_pos[1]), (next_pos[0] - 1, next_pos[1]),
                        (next_pos[0], next_pos[1] + 1), (next_pos[0], next_pos[1] - 1)])

        features['strong-ghost-1-step-away'] = -6.57 - 7.8 * strong_ghost_away(next_pos)
        features['strong-ghost-2-step-away'] = -6.8 + 2.56 * strong_ghost_2_away()

        # Part 2. Weak ghosts.

        # Get a list of weak ghost states
        weakGhostStates: List[game.AgentState] = list(filter(lambda x: self.distanceGhost(next_pos, getGhostPosition(x), walls) is not None, self.get_weak_ghost_states(ghostStates)))

        # If there are weak ghosts (scared ghosts)
        if weakGhostStates:
            # get the nearest ghost
            closest_ghost: game.AgentState = min(weakGhostStates, key=lambda x: self.distanceGhost(next_pos, getGhostPosition(x), walls))
            features['closest-weak-ghost'] = -0.47 + 4.85 * self.distanceGhost(next_pos, getGhostPosition(closest_ghost), walls)

        features['weak-ghost'] = -3.4 - 2.01 * len(self.get_weak_ghost_states(ghostStates))

        # Part 3. Food and capsule.

        # If there are any scared ghosts, we tend to eat that ghost regardless of food and capsule.
        # So we set parameters of these to zero, and rely on only '#-of-strong-ghost-1/2-step-away' and 'weak-ghost'
        if self.has_weak_ghosts(ghostStates):
            features['eats-capsule'] = 0
            features['eats-food'] = 0
            features['nearest-capsule'] = 0
            features['nearest-food'] = 0
        else:
            capsules: List[Tuple[int, int]] = state.getCapsules()  # Position of all capsules

            # If there are a ghost 1 step away, we tend to focus on escaping and forget about food here.
            # So in this case, we set 'eats-capsule' and 'eats-food' to zero.
            if features['strong-ghost-1-step-away'] != 0:
                features['eats-capsule'] = 0
                features['eats-food'] = 0

            # Check food and capsule
            else:
                # Check if we can eat capsule in the next step
                if next_pos not in capsules:
                    features['eats-capsule'] = 0
                else:
                    features['eats-capsule'] = 50

                # If there are capsules, we ignore food.
                # If no capsules, we check if we can eat food in the next step.
                if capsules or not food[next_pos[0]][next_pos[1]]:
                    features['eats-food'] = 0
                else:
                    features['eats-food'] = 9

            # Then we consider the closest food and capsule.
            # Similarly, if there are capsules, we ignore food.
            if capsules:
                features['nearest-capsule'] = 5.98 + 4.2 * self.closestCapsule(next_pos, capsules, walls)
                features['nearest-food'] = 0
            else:
                features['nearest-capsule'] = 0

                food_dist: Optional[float] = closestFood(next_pos, food, walls)
                if food_dist:
                    features['nearest-food'] = 5 - 0.015 * food_dist
                else:
                    features['nearest-food'] = 0

        features['bias'] = -1.19
        features.divideAll(10.0)
        return features

    @staticmethod
    def has_weak_ghosts(ghostStates: List[game.AgentState]) -> bool:
        for ghostState in ghostStates:
            if ghostState.scaredTimer > 0:
                return True
        return False

    @staticmethod
    def get_next_pacman_location(pos: Tuple[int, int], action: str) -> Tuple[int, int]:
        direction: Tuple[int, int] = Actions.directionToVector(action)
        return int(pos[0] + direction[0]), int(pos[1] + direction[1])

    @staticmethod
    def get_strong_ghost_states(ghost_states: List[game.AgentState]) -> List[game.AgentState]:
        def is_strong_ghost(ghost: game.AgentState) -> bool:
            return ghost.scaredTimer == 0

        return list(filter(is_strong_ghost, ghost_states))

    @staticmethod
    def get_weak_ghost_states(ghost_states: List[game.AgentState]) -> List[game.AgentState]:
        def is_weak_ghost(ghost: game.AgentState) -> bool:
            return ghost.scaredTimer > 0

        return list(filter(is_weak_ghost, ghost_states))

    @staticmethod
    # Returns the distance to the closest ghost
    def distanceGhost(pos: Tuple[int, int], ghost: List[Tuple[int, int]], walls: game.Grid) -> Optional[float]:
        return distanceObject(pos, [ghost], walls)

    @staticmethod
    # Returns the distance to the closest capsule
    def closestCapsule(pos: Tuple[int, int], capsules: List[Tuple[int, int]], walls: game.Grid) -> Optional[float]:
        return distanceObject(pos, capsules, walls)
