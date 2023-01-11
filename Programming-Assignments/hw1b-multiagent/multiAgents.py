# multiAgents.py
# --------------
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


from game import Directions, Agent
import random
import util

# For typing alias
from typing import List, Tuple
import pacman

# For question 6 only.
q6_count: int = 0  # This variable indicates which game round we are in.


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: pacman.GameState) -> str:
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legal_moves: List[str] = gameState.getLegalActions()
        # Choose one of the best actions
        scores: List[float] = [self.evaluationFunction(gameState, action) for action in legal_moves]
        best_score: float = max(scores)
        best_indices: List[int] = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index: int = random.choice(best_indices)  # Pick randomly among the best
        return legal_moves[chosen_index]

    @staticmethod
    def evaluationFunction(currentGameState, action) -> float:
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)

        # If we take 'action', what will the next game state, new position, new food list, and new ghost positions be?
        child_game_state = currentGameState.getPacmanNextState(action)
        new_pos: Tuple[int, int] = child_game_state.getPacmanPosition()
        new_food: List[Tuple[int, int]] = child_game_state.getFood().asList()
        new_ghost_states = child_game_state.getGhostStates()
        new_ghost_pos: List[Tuple[int, int]] = []
        for g in new_ghost_states:
            new_ghost_pos.append(g.getPosition())
        new_scared_timers: List[int] = [ghostState.scaredTimer for ghostState in new_ghost_states]

        """*** YOUR CODE HERE ***"""
        scared: bool = new_scared_timers[0] > 0
        if not scared and (new_pos in new_ghost_pos):  # we are bumping into a ghost
            return -1.0
        if new_pos in currentGameState.getFood().asList():  # we are eating a food
            return 1.0

        # Given a position, this function returns the manhattan distance from chosen position to that position.
        def dist_from_newPos(pos: Tuple[int, int]) -> float:
            return util.manhattanDistance(pos, new_pos)

        # sort the available food from nearest to furthest
        closest_food_dist: List[Tuple[int, int]] = sorted(new_food, key=dist_from_newPos)
        # sort the ghosts from nearest to furthest
        closest_ghost_dist: List[Tuple[int, int]] = sorted(new_ghost_pos, key=dist_from_newPos)
        # Expectation of food and ghost
        food_gain: float = 1 / dist_from_newPos(closest_food_dist[0])
        ghost_loss: float = 1 / dist_from_newPos(closest_ghost_dist[0])

        return food_gain - ghost_loss


def scoreEvaluationFunction(currentGameState: pacman.GameState) -> float:
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2') -> None:
        super().__init__()
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: pacman.GameState) -> str:
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether the game state is a winning state

        gameState.isLose():
        Returns whether the game state is a losing state
        """
        """*** YOUR CODE HERE ***"""

        # get the list of ghost agent indexes, e.g [1,2,3]
        ghost_indices: List[int] = list(range(1, gameState.getNumAgents()))

        # Checks if a state is a terminal state
        def is_term(state: pacman.GameState, depth: int) -> bool:
            if state.isWin() or state.isLose():
                return True
            if self.depth == depth:
                return True
            return False

        # min-value function of minimax
        def min_value(state: pacman.GameState, depth: int, ghostIndex: int) -> int:
            if is_term(state, depth):
                return self.evaluationFunction(state)

            v: int = 0x3f3f3f3f
            for action in state.getLegalActions(ghostIndex):
                if ghostIndex == ghost_indices[-1]:  # if this is the last ghost, depth + 1, and use max_value
                    v = min(v, max_value(state.getNextState(ghostIndex, action), depth + 1))
                else:  # otherwise, continue to the next ghost. Depth won't increase.
                    v = min(v, min_value(state.getNextState(ghostIndex, action), depth, ghostIndex + 1))
            return v

        # max-value function of minimax
        def max_value(state: pacman.GameState, depth: int) -> int:
            if is_term(state, depth):
                return self.evaluationFunction(state)

            v: int = -0x3f3f3f3f
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.getNextState(0, action), depth, 1))
            return v

        # value(state) function of minimax.
        # Starting from root node, expand all successors (starting with ghost move, min_value) and sort them
        res: List[Tuple[str, int]] = [(action, min_value(gameState.getNextState(0, action), 0, 1)) for action in
                                      gameState.getLegalActions(0)]
        res.sort(key=lambda k: k[1], reverse=True)
        return res[0][0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: pacman.GameState) -> str:
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        curr_value: float = -0x3f3f3f3f
        alpha: float = -0x3f3f3f3f
        beta: float = 0x3f3f3f3f
        next_action: str = Directions.STOP
        legal_actions: List[str] = gameState.getLegalActions(0).copy()

        # search for all valid actions
        for action in legal_actions:
            next_state: pacman.GameState = gameState.getNextState(0, action)
            next_value: float = self.get_node_value(next_state, 0, 1, alpha, beta)
            # same as v = max(v, value(successor))
            if next_value > curr_value:
                curr_value = next_value
                next_action = action
            alpha = max(alpha, curr_value)
        return next_action

    def get_node_value(self, gameState: pacman.GameState,
                       cur_depth: int = 0, agent_index: int = 0,
                       alpha: float = -0x3f3f3f3f, beta: float = 0x3f3f3f3f):
        """
        Using self-defined function, alpha_value(), beta_value() to choose the most appropriate action
        Only when it's the final state, can we get the value of each node, using the self.evaluationFunction(gameState)
        Otherwise we just get the alpha/beta value we defined here.
        """

        # agentIndex=0 means Pacman, ghosts are >= 1
        max_agent_indices: List[int] = [0]  # 🔼
        min_agent_indices: List[int] = list(range(1, gameState.getNumAgents()))  # 🔽

        # If is terminal state, return
        if cur_depth == self.depth:
            return self.evaluationFunction(gameState)
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        # calculate alpha (if pacman) or beta (if ghost)
        if agent_index in max_agent_indices:  # 🔼
            return self.alpha_value(gameState, cur_depth, agent_index, alpha, beta)
        elif agent_index in min_agent_indices:  # 🔽
            return self.beta_value(gameState, cur_depth, agent_index, alpha, beta)

    # max-value (state, alpha, beta)
    def alpha_value(self, gameState: pacman.GameState,
                    cur_depth: int, agent_index: int,
                    alpha: float = -0x3f3f3f3f, beta: float = 0x3f3f3f3f):
        v = -1e10
        legal_actions: List[str] = gameState.getLegalActions(agent_index)
        for action in legal_actions:
            next_v = self.get_node_value(gameState.getNextState(agent_index, action),
                                         cur_depth, agent_index + 1, alpha, beta)
            v = max(v, next_v)
            if v > beta:  # next_agent in which party
                return v
            alpha = max(alpha, v)
        return v

    def beta_value(self, gameState: pacman.GameState,
                   cur_depth: int, agent_index: int,
                   alpha: float = -0x3f3f3f3f, beta: float = 0x3f3f3f3f):
        """
        min_party, search for minimums
        """
        v = 1e10
        legal_actions = gameState.getLegalActions(agent_index)
        for action in legal_actions:
            if agent_index == gameState.getNumAgents() - 1:  # The last ghost. Increase depth
                next_v = self.get_node_value(gameState.getNextState(agent_index, action),
                                             cur_depth + 1, 0, alpha, beta)
                v = min(v, next_v)  # begin next depth
                if v < alpha:
                    return v
            else:
                next_v = self.get_node_value(gameState.getNextState(agent_index, action),
                                             cur_depth, agent_index + 1, alpha, beta)
                v = min(v, next_v)
                if v < alpha:
                    return v
            beta = min(beta, v)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: pacman.GameState) -> str:
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.  # 概率均等分布
        """
        "*** YOUR CODE HERE ***"

        max_value: float = -0x3f3f3f3f
        max_action: str = Directions.STOP

        # Starting from root (pacman move), get successors
        for action in gameState.getLegalActions(agentIndex=0):
            successor_state: pacman.GameState = gameState.getNextState(0, action)  # successor gameState
            successor_value: float = self.expNode(successor_state, 0, 1)
            if successor_value > max_value:
                max_value = successor_value
                max_action = action

        return max_action

    # This function is the same as the 'max-value()' in minimax, where the pacman itself wants the biggest
    def maxNode(self, gameState: pacman.GameState, currentDepth: int) -> float:
        if currentDepth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        max_value: float = -0x3f3f3f3f
        for action in gameState.getLegalActions(agentIndex=0):
            suc_state: pacman.GameState = gameState.getNextState(action=action, agentIndex=0)
            suc_value: float = self.expNode(suc_state, currentDepth=currentDepth, agentIndex=1)
            if suc_value > max_value:
                max_value = suc_value
        return max_value

    def expNode(self, gameState: pacman.GameState, currentDepth: int, agentIndex: int) -> float:
        if currentDepth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        num_action: int = len(gameState.getLegalActions(agentIndex=agentIndex))
        total_value: float = 0.0
        num_agent: int = gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex=agentIndex):
            suc_state: pacman.GameState = gameState.getNextState(agentIndex=agentIndex, action=action)
            if agentIndex == num_agent - 1:
                suc_value = self.maxNode(suc_state, currentDepth=currentDepth + 1)
            else:
                suc_value = self.expNode(suc_state, currentDepth=currentDepth, agentIndex=agentIndex + 1)
            total_value += suc_value
        # This is ⭕ (instead of  🔼or 🔽). We need to get the average value (instead of min or max) of the successors.
        return total_value / num_action


def betterEvaluationFunction(currentGameState: pacman.GameState) -> float:
    """
    Your extreme ghost-hunting, pellet-nabbing 能量药片, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here, so we know what you did>
        Initialize expected score to the current score.
        If there's food left, the closer food is, the higher expected gain is.
        If there's ghost(s):
            If the ghost is scared, the closer the ghost is ,
            the higher the gain is (since we are more likely to eat the ghost).
            If the ghost isn't scared, the closer, the higher the loss is (since
            we are closer to danger).
            If we are bumping into a not-scared ghost, return negative infinity
            since pacman will die in this case.
        Add the gain or loss to current score to get expected score.
    """
    """*** YOUR CODE HERE ***"""

    curr_pos: Tuple[int, int] = currentGameState.getPacmanPosition()
    curr_foods: List[Tuple[int, int]] = currentGameState.getFood().asList()
    curr_ghosts: List = currentGameState.getGhostStates()
    expected_score: float = currentGameState.getScore()  # Get the current score

    # These are score of food (gain), ghost (loss), and scared ghost (gain)
    # We want to encourage pacman to eat scared ghost, so the corresponding weight is much higher.
    WEIGHT_FOOD: float = 10.0
    WEIGHT_GHOST: float = -10.0
    WEIGHT_SCARED_GHOST: float = 100.0

    dist_to_foods: List[int] = [util.manhattanDistance(curr_pos, foodPos) for foodPos in curr_foods]
    # If there's food left, estimate the score we can gain by eating those foods
    if len(dist_to_foods) > 0:
        expected_score += WEIGHT_FOOD / min(dist_to_foods)

    # Evaluate the distance to ghosts
    for ghost in curr_ghosts:
        distance: float = util.manhattanDistance(curr_pos, ghost.getPosition())  # distance to this ghost
        if distance > 0:
            if ghost.scaredTimer > 0:  # Add score if ghost is scared.
                expected_score += WEIGHT_SCARED_GHOST / distance  # The closer the ghost is, the higher score we get.
            else:  # If ghost is not scared, decrease points since we are moving towards danger
                expected_score += WEIGHT_GHOST / distance
        else:  # In this case, distance == 0, pacman is eaten by the ghost
            return -0x3f3f3f3f

    return expected_score


# Abbreviation
better = betterEvaluationFunction


# For Q6 only.
def betterEvaluationFunctionQ6(currentGameState: pacman.GameState) -> float:
    curr_pos: Tuple[int, int] = currentGameState.getPacmanPosition()
    curr_foods: List[Tuple[int, int]] = currentGameState.getFood().asList()
    curr_ghosts: List = currentGameState.getGhostStates()
    curr_capsules: List[Tuple[int, int]] = currentGameState.getCapsules()  # capsules
    expected_score: float = currentGameState.getScore()  # Get the current score

    # In this function, we did the following (similar to what we did in Q5, 'betterEvaluationFunction' func)
    #
    #         Initialize expected score to the current score.
    #         If there's capsules left, consider the nearest capsule.
    #         If there's no capsule (only normal food), consider the nearest ghost.
    #         If there's ghost(s):
    #             If the ghost is scared and close enough (we have a chance to eat them),
    #                 we can gain a scared-ghost-bonus.
    #             If the ghost is scared but far away, we won't do anything to the score.
    #             If the ghost isn't scared, the closer, the higher the loss is (since we are closer to danger).
    #         Add the gain or loss to current score to get expected score.

    # CLAIM:
    # 'WEIGHT_FOOD', 'WEIGHT_CAPSULE', 'WEIGHT_GHOST', 'WEIGHT_SCARED_GHOST' are the basic parameters.
    # 'coefficients' containing a float of four are adjustments for them.
    # I randomly generated these 'coefficients', tried different searching depth (2 or 3),
    # simulated the testcases for more than 63000 times on my own server,
    # and resulted in the following float number results like (0.05727021, 0.22926805, 0.11200294, 0.75189848).
    # These numbers are all randomly generated and tested by myself.
    # If some other students happens to find the same numbers, it must be a coincidence.

    WEIGHT_FOOD: float = 15.0
    WEIGHT_CAPSULE: float = 20.0
    WEIGHT_GHOST: float = -3000.0
    WEIGHT_SCARED_GHOST: float = 2000.0
    coefficients_list: List[Tuple] = [
        (0, 0, 0, 0),  # placeholder
        (0.05727021, 0.22926805, 0.11200294, 0.75189848),  # case 1; depth = 3; legal 2847
        (0.05727021, 0.22926805, 0.11200294, 0.75189848),  # case 2; depth = 3; legal 3037
        (0.71945386, 0.38327929, 0.00785529, 0.87712883),  # case 3; depth = 3; legal 3417
        (0.80605661, 0.37036110, 0.02462315, 0.56388212),  # case 4; depth = 3; legal 2797
        (0.11927982, 0.29239600, 0.04996382, 0.39456618),  # case 5; depth = 3; legal 3637
        (0.03667265, 0.99115328, 0.09935081, 0.57059650),  # case 6; depth = 3; legal 3071
        (0.89790356, 0.26427819, 0.36721456, 0.52057139),  # case 7; depth = 3; legal 3347
        (0.50064533, 0.32248629, 0.13772379, 0.74437512),  # case 8; depth = 3; legal 3224
        (0.42554790, 0.28855117, 0.01935615, 0.34512226),  # case 9; depth = 3; legal 3208
        (0.18157965, 0.61999352, 0.02527364, 0.25005017)  # case 10; depth = 3; legal 2839
    ]

    coefficients: Tuple[float, float, float, float] = coefficients_list[q6_count]

    # If there are some capsules, consider capsules food
    if curr_capsules:  # If there are capsules, consider the nearest one
        min_distance: float = 0x3f3f3f3f
        for capsule in curr_capsules:  # get the nearest capsule
            min_distance = min(util.manhattanDistance(capsule, curr_pos), min_distance)
        expected_score += WEIGHT_CAPSULE * coefficients[0] * (1.0 / min_distance)
    else:  # if there's no capsules, consider normal food only
        min_distance: float = 0x3f3f3f3f
        for food in curr_foods:  # get the nearest food
            min_distance = min(util.manhattanDistance(food, curr_pos), min_distance)
        expected_score += WEIGHT_FOOD * coefficients[1] * (1.0 / min_distance)

    scared_bonus: float = 0
    for ghost in curr_ghosts:
        ghost_pos: Tuple[int, int] = ghost.getPosition()
        ghost_scared = ghost.scaredTimer
        if ghost_scared > 0:  # ghost scared
            # get the distance to the ghost
            distance: float = util.manhattanDistance(curr_pos, ghost_pos)
            # If scared ghost is close enough that we have a chance to eat them.
            # The closer the ghost is, the higher bonus we can get.
            if distance < ghost_scared:
                scared_bonus = max(scared_bonus, WEIGHT_SCARED_GHOST * coefficients[2] * (1.0 / distance))
            # If scared ghost is far away, we won't change the expected score. Nothing here.
        else:  # ghost is not scared
            if curr_pos == ghost_pos:  # The pacman is eaten by the ghost
                # Eating by the ghost is dangerous. Being eaten itself leaves a punishment of 500.
                # Therefore, we would give a higher loss in this case: the expected score would be negative.
                expected_score = WEIGHT_GHOST * coefficients[3]

    return expected_score + scared_bonus


# Q6 class
class ContestAgent(AlphaBetaAgent):
    """
      Your agent for the mini-contest.

      I noticed that in question 6, ghosts are 'directional ghosts'.
      They move towards or away from (if scared) pacman with probability 0.8, and move randomly with probability 0.2.
      According to the lecture, we may use Expectimax given the opponent information.
      However, after tremendous number of simulations, Alpha-Beta performs better under special parameters,
      therefore we use Alpha-Beta search agent in this question.
      The class is inherited from 'AlphaBetaAgent'.
    """

    def __init__(self):
        super().__init__()
        self.depth = 3  # search depth limit
        # change evaluation func
        self.evaluationFunction = util.lookup('betterEvaluationFunctionQ6', globals())

    def getAction(self, gameState: pacman.GameState) -> str:
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        """*** YOUR CODE HERE ***"""

        # check if a new game has started
        if gameState.getScore() == 0 and gameState.getNumFood() == 69:
            # update q6_count
            global q6_count
            q6_count += 1

        # call parent class (a-b search agent) getAction Function
        return super(ContestAgent, self).getAction(gameState)
