# valueIterationAgents.py
# -----------------------
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
import math
import sys

# valueIterationAgents.py
# -----------------------
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

import mdp
import util

from learningAgents import ValueEstimationAgent
import itertools
import gridworld  # for typing alias
from typing import Dict, Tuple, List, Union, Set  # for typing alias


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp: gridworld.Gridworld, discount: float = 0.9, iterations: int = 100) -> None:
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp: gridworld.Gridworld = mdp
        self.discount: float = discount
        self.iterations: int = iterations
        self.values: Dict[str or Tuple[int, int], float] = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self) -> None:
        # Write value iteration code here
        """*** YOUR CODE HERE ***"""
        for _ in range(self.iterations):
            nextValues: Dict[str or Tuple[int, int], float] = util.Counter()
            # get the max Q-value for each state
            for state in self.mdp.getStates():
                # For all possible actions for the mdp, compute the Q-value for each action and get the best.
                # In case there are no possible actions, return 0
                nextValues[state] = max(
                    map(lambda action: self.computeQValueFromValues(state, action), self.mdp.getPossibleActions(state)),
                    default=0.0)
            self.values = nextValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action) -> float:
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Q-value is the reward of (state, action, new_state) + discounted value got from the new state
        # We get all possible transition states and probabilities (if we take the action) and return
        # a weighted sum of all possibilities.
        return sum(map(lambda x: x[1] * (self.mdp.getReward(state, action, x[0]) + self.discount * self.getValue(x[0])),
                       self.mdp.getTransitionStatesAndProbs(state, action)))

    def computeActionFromValues(self, state) -> str:
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        best_action = None  # If no legal actions, we shall return None
        best_q_value = -math.inf

        # Traverse all possible actions to see if there's a better q-value possible
        for action in self.mdp.getPossibleActions(state):
            if self.computeQValueFromValues(state, action) > best_q_value:
                best_action = action
                best_q_value = self.computeQValueFromValues(state, action)
        return best_action
        # return max(self.mdp.getPossibleActions(state), default=None, key=lambda a: self.computeQValueFromValues(state, a))

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        """Returns the policy at the state (no exploration)."""
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

    def getMaxQ(self, state):
        return max(map(lambda a: self.computeQValueFromValues(state, a),
                       self.mdp.getPossibleActions(state)), default=0)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp: gridworld.Gridworld, discount: float = 0.9, iterations: int = 1000) -> None:
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self) -> None:
        """*** YOUR CODE HERE ***"""
        for state in itertools.islice(itertools.cycle(self.mdp.getStates()), self.iterations):
            if self.mdp.isTerminal(state):
                continue
            self.values[state] = self.getQValue(state, self.getAction(state))
        # for state in filter(lambda s: not self.mdp.isTerminal(s), itertools.islice(itertools.cycle(self.mdp.getStates()), self.iterations)):
        #     self.values[state] = self.getQValue(state, self.getAction(state))


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp: gridworld.Gridworld, discount: float = 0.9, iterations: int = 100, theta: float = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta: float = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """*** YOUR CODE HERE ***"""

        # Compute predecessors of all states.
        # The predecessors of a state 's' is all states that have a nonzero probability of reaching s by taking action a

        # Typing alias description: a state either a str (e.g. "TERMINAL_STATE") or a tuple (e.g. (10,20)).
        predecessors: Dict[Union[Tuple[int, int], str], Set] = {state: set() for state in self.mdp.getStates()}
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for st, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    predecessors[st].add(state)

        # Initialize an empty priority queue
        queue: util.PriorityQueue = util.PriorityQueue()

        # For each non-terminal state s, do ...
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue

            # Find the absolute value of the difference between current value of s in self.values
            # and the highest Q-value across all possible actions from s.
            # Call this number 'diff'.
            diff: float = abs(self.values[state] - max(
                map(lambda action: self.computeQValueFromValues(state, action), self.mdp.getPossibleActions(state)),
                default=0.0))

            # Push s into the priority queue with priority -diff.
            queue.push(state, -diff)

        # For 'iteration' in 0, 1, 2, ..., self.iterations - 1
        for iteration in range(self.iterations):
            # If the priority queue is empty, terminate
            if queue.isEmpty():
                return

            # Pop a state s off the priority queue
            s: Union[Tuple[int, int], str] = queue.pop()

            # Update the value of s in self.values
            self.values[s] = max(
                map(lambda action: self.computeQValueFromValues(s, action), self.mdp.getPossibleActions(s)),
                default=0.0)

            # For each predecessor p of s:
            for p in predecessors[s]:
                diff: float = abs(self.values[p] - max(
                    map(lambda act: self.computeQValueFromValues(p, act), self.mdp.getPossibleActions(p)),
                    default=0.0))
                if diff > self.theta:
                    queue.update(p, -diff)
