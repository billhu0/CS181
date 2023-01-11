# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
from typing import List, Tuple

import util


# from searchAgents import PositionSearchProblem

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self) -> Tuple:
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state) -> bool:
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state) -> List[Tuple]:
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


# Our own node type
class Node:
    def __init__(self, state: Tuple[int, int], path: List[str] = None, cost: int = 0):
        if path is None:
            path = []
        self.state: Tuple[int, int] = state  # the current location of this node
        self.path: List[str] = path  # the path by which we can reach this node from start
        self.cost: int = cost  # the current cost reaching from start to this node


'''
Our own dfs or bfs search function. 
The only difference for them is that DFS uses a stack, while BFS uses a queue.
'''


def dfs_or_bfs_search(problem: SearchProblem, fringe: util.Stack or util.Queue) -> List[str]:
    # create a stack/queue fringe, and 'visited' list
    visited: List[Tuple[int, int]] = []

    # push the first node (start state)
    start_state: Tuple[int, int] = problem.getStartState()  # (3,4)  [(3,4), []]
    fringe.push(Node(start_state))

    while not fringe.isEmpty():
        node: Node = fringe.pop()  # pop the first node

        if problem.isGoalState(node.state):  # check if we have reached the goal (goal node popped)
            return node.path

        if node.state not in visited:  # check if we have visited. If visited, skip.
            visited.append(node.state)
        else:
            continue

        successor: List[Tuple[Tuple[int, int], str, int]] = problem.getSuccessors(node.state)
        for s in successor:
            # The type of each successor item looks like this:
            # e.g. (location, direction, step_cost) ((5, 4), 'South', 1)
            location: Tuple[int, int] = s[0]
            direction: str = s[1]
            if location not in visited:
                fringe.push(Node(state=location, path=node.path + [direction]))
    else:
        return []  # Solution not found!


def depthFirstSearch(problem: SearchProblem) -> List[str]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    return dfs_or_bfs_search(problem, util.Stack())


def breadthFirstSearch(problem: SearchProblem) -> List[str]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    return dfs_or_bfs_search(problem, util.Queue())


def uniformCostSearch(problem: SearchProblem) -> List[str]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # create a priority queue as a fringe, and 'visited' list
    fringe: util.PriorityQueue = util.PriorityQueue()
    visited: List[Tuple[int, int]] = []
    # push the first node (start state)
    fringe.push(Node(state=problem.getStartState(), path=[], cost=0), 0)
    while not fringe.isEmpty():
        node: Node = fringe.pop()  # pop a node
        if problem.isGoalState(node.state):  # check if goal reached
            return node.path
        if node.state not in visited:  # check if visited
            visited.append(node.state)
        else:
            continue

        # get and traverse successor
        successors: List[Tuple[Tuple[int, int], str, int]] = problem.getSuccessors(node.state)
        for s in successors:
            s_state: Tuple[int, int] = s[0]
            s_direction: str = s[1]
            s_step_cost: int = s[2]
            new_cost: int = node.cost + s_step_cost
            fringe.push(Node(state=s_state, path=node.path + [s_direction], cost=new_cost), new_cost)
    else:
        return []  # solution not found


def nullHeuristic(state, problem=None) -> int:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[str]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # create a priority queue as a fringe, and 'visited' list
    fringe: util.PriorityQueue = util.PriorityQueue()
    visited: List[Tuple[int, int]] = []
    # push the first node (start state)
    start_state: Tuple[int, int] = problem.getStartState()  # start state
    start_heu: int = heuristic(start_state, problem)  # start state heuristic distance
    fringe.push(Node(state=start_state, path=[], cost=start_heu), start_heu)
    while not fringe.isEmpty():
        node: Node = fringe.pop()  # pop a node
        if problem.isGoalState(node.state):  # check if goal reached
            return node.path
        if node.state not in visited:  # check if visited
            visited.append(node.state)
        else:
            continue

        # get and traverse successor
        successors: List[Tuple[Tuple[int, int], str, int]] = problem.getSuccessors(node.state)
        for s in successors:
            s_state: Tuple[int, int] = s[0]
            s_direction: str = s[1]
            s_step_cost: int = s[2]
            # cost from start to the successor node
            new_cost: int = node.cost + s_step_cost
            # estimated total cost (pass through successor node)
            total_cost: int = new_cost + heuristic(s_state, problem)
            fringe.push(Node(state=s_state, path=node.path + [s_direction], cost=new_cost), total_cost)
    else:
        return []  # solution not found


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
