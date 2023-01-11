# logicPlan.py
# ------------
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
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

from typing import Dict, List, Tuple, Callable, Generator, Union, Any
import util
import sys
import logic
import game

from logic import conjoin, disjoin
from logic import PropSymbolExpr, Expr, to_cnf, pycoSAT, parseExpr, pl_true

import itertools
import copy

pacman_str: str = 'P'
food_str: str = 'FOOD'
wall_str: str = 'WALL'
pacman_wall_str: str = pacman_str + wall_str
ghost_pos_str: str = 'G'
ghost_east_str: str = 'GE'
pacman_alive_str: str = 'PA'
DIRECTIONS: List[str] = ['North', 'South', 'East', 'West']

# blocked_str_map = {'North': 'NORTH_BLOCKED', 'South': 'SOUTH_BLOCKED', 'East': 'EAST_BLOCKED', 'West': 'WEST_BLOCKED'}
blocked_str_map: Dict[str, str] = dict([(direction, (direction + "_blocked").upper()) for direction in DIRECTIONS])

# gem_num_adj_wall_str_map = {1: 'GEQ_1_adj_walls', 2: 'GEQ_2_adj_walls', 3: 'GEQ_3_adj_walls'}
geq_num_adj_wall_str_map: Dict[int, str] = dict([(num, "GEQ_{}_adj_walls".format(num)) for num in range(1, 4)])

DIR_TO_DXDY_MAP: Dict[str, Tuple[int, int]] = {'North': (0, 1), 'South': (0, -1), 'East': (1, 0), 'West': (-1, 0)}


# Print black-background-red-foreground highlighted text.
# For debug use
def print_red(s: str, newLine: bool = True):
    print("\033[31;40m {} \033[0m".format(s), end='\n' if newLine else '')


# ______________________________________________________________________________
# QUESTION 1

def sentence1() -> Expr:
    """Returns an Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    """*** BEGIN YOUR CODE HERE ***"""

    a: Expr = Expr('A')
    b: Expr = Expr('B')
    c: Expr = Expr('C')

    condition_1: Expr = a | b  # A or B
    condition_2: Expr = (~a) % ((~b) | c)  # (not A) if and only if ((not B) or C)
    condition_3: Expr = logic.disjoin([(~a), (~b), c])  # (not A) or (not B) or C

    return conjoin([condition_1, condition_2, condition_3])
    """*** END YOUR CODE HERE ***"""


def sentence2() -> Expr:
    """
    Returns an Expr instance that encodes that the following expressions are all true.

    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** BEGIN YOUR CODE HERE ***"

    a: Expr = Expr('A')
    b: Expr = Expr('B')
    c: Expr = Expr('C')
    d: Expr = Expr('D')

    condition_1: Expr = c % (b | d)  # C if and only if (B or D)
    condition_2: Expr = a >> ((~b) & (~d))  # A implies ((not B) and (not D))
    condition_3: Expr = (~(b & (~c))) >> a  # (not (B and (not C))) implies A
    condition_4: Expr = (~d) >> c  # (not D) implies C

    return conjoin([condition_1, condition_2, condition_3, condition_4])
    '''*** END YOUR CODE HERE *** '''


def sentence3() -> PropSymbolExpr:
    """Using the symbols PacmanAlive_1 PacmanAlive_0, PacmanBorn_0, and PacmanKilled_0,
    created using the PropSymbolExpr constructor, return a PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    Pacman is alive at time 1 if and only if Pacman was alive at time 0 and it was
    not killed at time 0 or it was not alive at time 0 and it was born at time 0.

    Pacman cannot both be alive at time 0 and be born at time 0.

    Pacman is born at time 0.
    (Project update: for this question only, [0] and _t are both acceptable.)
    """
    '''*** BEGIN YOUR CODE HERE ***'''

    p_alive_at_1: PropSymbolExpr = PropSymbolExpr('PacmanAlive', time=1)  # Pacman is alive at time 1
    p_alive_at_0: PropSymbolExpr = PropSymbolExpr('PacmanAlive', time=0)  # Pacman is alive at time 0
    p_killed_at_0: PropSymbolExpr = PropSymbolExpr('PacmanKilled', time=0)  # Pacman is killed at time 0
    p_born_at_0: PropSymbolExpr = PropSymbolExpr('PacmanBorn', time=0)  # Pacman was born at time 0

    # alive at 1 IFF ((alive at 0 and not killed at 0) or (not alive at 0 and born at 0))
    con_1: PropSymbolExpr = p_alive_at_1 % ((p_alive_at_0 & (~p_killed_at_0)) | (~p_alive_at_0 & p_born_at_0))

    # not both (alive at 0 and born at 0)
    con_2: PropSymbolExpr = ~(p_alive_at_0 & p_born_at_0)

    # born at 0
    con_3: PropSymbolExpr = p_born_at_0

    return conjoin([con_1, con_2, con_3])
    '''*** END YOUR CODE HERE ***'''


def findModel(sentence: Expr) -> Union[Dict[Expr, bool], bool]:
    """Given a propositional logic sentence (i.e. a Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """

    '''
    Convert a sentence to cnf form
    For example, sentence1() = ((A | B) & (~A <=> (~B | C)) & (~A | ~B | C))
    to_cnf(sentence1()) = ((A | B) & (B | ~A) & (~C | ~A) & (~B | C | A) & (~A | ~B | C)) 
    
    Then we can solve it. In the above example, the solution is 
    {B: True, A: False, C: True}
    '''
    cnf_sentence = to_cnf(sentence)
    return pycoSAT(cnf_sentence)


def findModelCheck() -> Dict[Any, bool]:
    """Returns the result of findModel(Expr('a')) if lower cased expressions were allowed.
    You should not use findModel or Expr in this method.
    This can be solved with a one-line return statement.
    """

    class dummyClass:
        """dummy('A') has representation A, unlike a string 'A' that has repr 'A'.
        Of note: Expr('Name') has representation Name, not 'Name'.
        """

        def __init__(self, variable_name: str = 'A'):
            self.variable_name = variable_name

        def __repr__(self):
            return self.variable_name

    '''*** BEGIN YOUR CODE HERE ***'''
    return {dummyClass('a'): True}
    '''*** END YOUR CODE HERE ***'''


def entails(premise: Expr, conclusion: Expr) -> bool:
    """Returns True if the premise entails the conclusion and False otherwise.
    """
    "*** BEGIN YOUR CODE HERE ***"
    # In this function, we should return True IFF 'premise' entails 'conclusion'
    # This means it is impossible that in some model, 'premise' is True and 'conclusion' is False.
    # We can judge this to decide the entailment.
    # Therefore, if findModel(premise and not conclusion) is false, we should return True, and vice versa.
    return not findModel(premise & (~conclusion))
    "*** END YOUR CODE HERE ***"


def plTrueInverse(assignments: Dict[Expr, bool], inverse_statement: Expr) -> bool:
    """Returns True if the (not inverse_statement) is True given assignments and False otherwise.
    pl_true may be useful here; see logic.py for its description.
    """
    "*** BEGIN YOUR CODE HERE ***"
    # If (~inverse_statement) is true given the assignments, we should return true.
    # We can directly call pl_true(...) function implemented in logic.py.
    return pl_true((~inverse_statement), assignments)
    "*** END YOUR CODE HERE ***"


# ______________________________________________________________________________
# QUESTION 2

def atLeastOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals (i.e. in the form A or ~A), return a single 
    Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals list is true.
    >>> A = PropSymbolExpr('A')
    >>> B = PropSymbolExpr('B')
    >>> symbols = [A, B]
    >>> atLeast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print(pl_true(atLeast1,model1))
    False
    >>> model2 = {A:False, B:True}
    >>> print(pl_true(atLeast1,model2))
    True
    >>> model3 = {A:True, B:True}
    >>> print(pl_true(atLeast1,model2))
    True
    """
    "*** BEGIN YOUR CODE HERE ***"

    # 'return a single Expr in CNF form that represents that at least one literal is True'
    # As long as one of them is true, result is true.
    # When none of them is true, result is false.
    # Therefore, we can simply return the disjunction ('or' result) of these literals.
    return disjoin(literals)
    "*** END YOUR CODE HERE ***"


def atMostOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    itertools.combinations may be useful here.
    """
    "*** BEGIN YOUR CODE HERE ***"

    # 'at most one of them is true'
    # It means that for arbitrary pair of these literals, they cannot be both true
    # Here 'they cannot both be true' can be represented as '~(x&y)', or equally, '~x|~y'
    # Therefore we can enumerate all 'combinations' and check the above
    clauses: List[Expr] = [(~e1 | ~e2) for e1, e2 in itertools.combinations(literals, 2)]
    # And then, take the conjunction of them (which means all of them have to be true)
    return conjoin(clauses)
    # util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"


def exactlyOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** BEGIN YOUR CODE HERE ***"
    # If at least one of them is true, and at most one of them is true,
    # exactly one of them is true.
    # We can directly call the above two functions to get the result.

    # Additionally,
    # 'If you decide to call your previously implemented atLeastOne and atMostOne,
    # call atLeastOne first to pass our autograder for q3'
    return conjoin([atLeastOne(literals), atMostOne(literals)])
    "*** END YOUR CODE HERE ***"


# ______________________________________________________________________________
# QUESTION 3

def pacmanSuccessorAxiomSingle(x: int, y: int, time: int, walls_grid: List[List[bool]] = None) -> Union[Expr, None]:
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    now, last = time, time - 1
    possible_causes: List[Expr] = []  # enumerate all possible causes for P[x,y]_t
    # the if statements give a small performance boost and are required for q4 and q5 correctness
    if walls_grid[x][y + 1] != 1:
        possible_causes.append(PropSymbolExpr(pacman_str, x, y + 1, time=last)
                               & PropSymbolExpr('South', time=last))
    if walls_grid[x][y - 1] != 1:
        possible_causes.append(PropSymbolExpr(pacman_str, x, y - 1, time=last)
                               & PropSymbolExpr('North', time=last))
    if walls_grid[x + 1][y] != 1:
        possible_causes.append(PropSymbolExpr(pacman_str, x + 1, y, time=last)
                               & PropSymbolExpr('West', time=last))
    if walls_grid[x - 1][y] != 1:
        possible_causes.append(PropSymbolExpr(pacman_str, x - 1, y, time=last)
                               & PropSymbolExpr('East', time=last))
    if not possible_causes:
        return None

    "*** BEGIN YOUR CODE HERE ***"
    # This function returns an expression defining
    # the sufficient & necessary ( <=> ) conditions for Pacman to be at (x,y) at t
    # Implementation: At least one possible cause <=> pacman is at (x,y) at time t
    return PropSymbolExpr(pacman_str, x, y, time=now) % atLeastOne(possible_causes)
    "*** END YOUR CODE HERE ***"


def SLAMSuccessorAxiomSingle(x: int, y: int, time: int, walls_grid: List[List[bool]]) -> Union[Expr, None]:
    """
    Similar to `pacmanSuccessorStateAxioms` but accounts for illegal actions
    where the pacman might not move timestep to timestep.
    Available actions are ['North', 'East', 'South', 'West']
    """
    now, last = time, time - 1
    moved_causes: List[Expr] = []  # enumerate all possible causes for P[x,y]_t, assuming moved to having moved
    if walls_grid[x][y + 1] != 1:
        moved_causes.append(PropSymbolExpr(pacman_str, x, y + 1, time=last)
                            & PropSymbolExpr('South', time=last))
    if walls_grid[x][y - 1] != 1:
        moved_causes.append(PropSymbolExpr(pacman_str, x, y - 1, time=last)
                            & PropSymbolExpr('North', time=last))
    if walls_grid[x + 1][y] != 1:
        moved_causes.append(PropSymbolExpr(pacman_str, x + 1, y, time=last)
                            & PropSymbolExpr('West', time=last))
    if walls_grid[x - 1][y] != 1:
        moved_causes.append(PropSymbolExpr(pacman_str, x - 1, y, time=last)
                            & PropSymbolExpr('East', time=last))
    if not moved_causes:
        return None

    moved_causes_sent: Expr = conjoin(
        [~PropSymbolExpr(pacman_str, x, y, time=last), ~PropSymbolExpr(wall_str, x, y), disjoin(moved_causes)])

    failed_move_causes: List[Expr] = []  # using merged variables, improves speed significantly
    auxilary_expression_definitions: List[Expr] = []
    for direction in DIRECTIONS:
        dx, dy = DIR_TO_DXDY_MAP[direction]
        wall_dir_clause = PropSymbolExpr(wall_str, x + dx, y + dy) & PropSymbolExpr(direction, time=last)
        wall_dir_combined_literal = PropSymbolExpr(wall_str + direction, x + dx, y + dy, time=last)
        failed_move_causes.append(wall_dir_combined_literal)
        auxilary_expression_definitions.append(wall_dir_combined_literal % wall_dir_clause)

    failed_move_causes_sent: Expr = conjoin([
        PropSymbolExpr(pacman_str, x, y, time=last),
        disjoin(failed_move_causes)])

    return conjoin([PropSymbolExpr(pacman_str, x, y, time=now) % disjoin(
        [moved_causes_sent, failed_move_causes_sent])] + auxilary_expression_definitions)


def pacphysicsAxioms(t: int, all_coords: List[Tuple], non_outer_wall_coords: List[Tuple], walls_grid: List[List] = None,
                     sensorModel: Callable = None, successorAxioms: Callable = None) -> Expr:
    """
    Given:
        t: timestep
        all_coords: list of (x, y) coordinates of the entire problem
        non_outer_wall_coords: list of (x, y) coordinates of the entire problem,
            excluding the outer border (these are the actual squares pacman can
            possibly be in)
        walls_grid: 2D array of either -1/0/1 or T/F. Used only for successorAxioms.
            Do NOT use this when making possible locations for pacman to be in.
        sensorModel(t, non_outer_wall_coords) -> Expr: function that generates
            the sensor model axioms. If None, it's not provided, so shouldn't be run.
        successorAxioms(t, walls_grid, non_outer_wall_coords) -> Expr: function that generates
            the sensor model axioms. If None, it's not provided, so shouldn't be run.
    Return a logic sentence containing all the following:
        - for all (x, y) in all_coords:
            If a wall is at (x, y) --> Pacman is not at (x, y)
        - Pacman is at exactly one of the squares at timestep t.
        - Pacman takes exactly one action at timestep t.
        - Results of calling sensorModel(...), unless None.
        - Results of calling successorAxioms(...), describing how Pacman can end in various
            locations on this time step. Consider edge cases. Don't call if None.
    """
    pacphysics_sentences: List[Expr] = []

    "*** BEGIN YOUR CODE HERE ***"

    # We will generate a bunch of physics axioms in this function
    # by considering all of the following.

    # 1. for all (x,y), if a wall is at (x,y), then pacman is not at (x,y)
    for x, y in all_coords:
        pacphysics_sentences.append(PropSymbolExpr(wall_str, x, y) >> (~PropSymbolExpr(pacman_str, x, y, time=t)))

    # 2. pacman is at exactly one position (not-outer-wall-position) at time t.
    pacphysics_sentences.append(
        exactlyOne([PropSymbolExpr(pacman_str, x, y, time=t) for x, y in non_outer_wall_coords]))

    # 3. pacman takes exactly one action at time t
    pacphysics_sentences.append(exactlyOne([PropSymbolExpr(action, time=t) for action in DIRECTIONS]))

    # 4. If sensorModel isn't None, add the results of calling sensorModel.
    if sensorModel is not None:
        pacphysics_sentences.append(sensorModel(t, non_outer_wall_coords))

    # 5. If successorAxioms isn't None, add the results of calling successorAxioms,
    #    (describing how pacman can end in various locations at this time)
    if successorAxioms is not None:
        if t > 0:
            pacphysics_sentences.append(successorAxioms(t, walls_grid, non_outer_wall_coords))

    "*** END YOUR CODE HERE ***"

    return conjoin(pacphysics_sentences)


def checkLocationSatisfiability(x1_y1: Tuple[int, int], x0_y0: Tuple[int, int], action0, action1, problem):
    """
    Given:
        - x1_y1 = (x1, y1), a potential location at time t = 1
        - x0_y0 = (x0, y0), Pacman's location at time t = 0
        - action0 = one of the four items in DIRECTIONS, Pacman's action at time t = 0
        - action1 = to ensure match with autograder solution
        - problem = an instance of logicAgents.LocMapProblem
    Note:
        - there's no sensorModel because we know everything about the world
        - the successorAxioms should be allLegalSuccessorAxioms where needed
    Return:
        - a model where Pacman is at (x1, y1) at time t = 1
        - a model where Pacman is not at (x1, y1) at time t = 1
    """
    walls_grid: game.Grid = problem.walls
    walls_list: List[Tuple[int, int]] = walls_grid.asList()
    all_coords: List[Tuple[int, int]] = list(
        itertools.product(range(problem.getWidth() + 2), range(problem.getHeight() + 2)))
    non_outer_wall_coords: List[Tuple[int, int]] = list(
        itertools.product(range(1, problem.getWidth() + 1), range(1, problem.getHeight() + 1)))
    KB: List[Expr] = []
    x0, y0 = x0_y0
    x1, y1 = x1_y1

    # We know which coords are walls:
    map_sent: List[PropSymbolExpr] = [PropSymbolExpr(wall_str, x, y) for x, y in walls_list]
    KB.append(conjoin(map_sent))

    "*** BEGIN YOUR CODE HERE ***"

    # Add the following to KB first:

    # 1. pac_physics_axioms(...) with appropriate time steps.
    #    No sensorModel here.
    KB.append(pacphysicsAxioms(1, all_coords, non_outer_wall_coords, walls_grid,
                               sensorModel=None, successorAxioms=allLegalSuccessorAxioms))
    KB.append(pacphysicsAxioms(0, all_coords, non_outer_wall_coords, walls_grid,
                               sensorModel=None, successorAxioms=None))

    # 2. pacman's current location (x0, y0)
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time=0))

    # 3. pacman takes action 0
    KB.append(PropSymbolExpr(action0, time=0))

    # 4. pacman takes action 1
    KB.append(PropSymbolExpr(action1, time=1))

    # In model_1, pacman is at (x1, y1) at time t=1 given (x0_y0, action0, action1)
    # if model1 is False, we know Pacman is guaranteed NOT to be there.
    model1: Expr = conjoin(KB + [PropSymbolExpr(pacman_str, x1, y1, time=1)])

    # In model_2, pacman is not at (x1, y1) ...... (same as above)
    # if model2 is False, we know Pacman is guaranteed to be there.
    model2: Expr = conjoin(KB + [(~PropSymbolExpr(pacman_str, x1, y1, time=1))])

    # query the SAT solver with findModel(...) for the two models
    return findModel(model1), findModel(model2)
    "*** END YOUR CODE HERE ***"


# ______________________________________________________________________________
# QUESTION 4

def positionLogicPlan(problem) -> List[str]:
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    Overview: add knowledge incrementally, and query for a model each timestep. Do NOT use pacphysicsAxioms.
    """
    walls_grid: game.Grid = problem.walls
    width: int = problem.getWidth()
    height: int = problem.getHeight()
    walls_list: List[Tuple[int, int]] = walls_grid.asList()
    x0, y0 = problem.startState
    xg, yg = problem.goal

    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords: List[Tuple[int, int]] = list(itertools.product(range(width + 2), range(height + 2)))
    non_wall_coords: List[Tuple[int, int]] = [loc for loc in all_coords if loc not in walls_list]
    actions: List[str] = ['North', 'South', 'East', 'West']
    KB: List[Expr] = []

    "*** BEGIN YOUR CODE HERE ***"

    # We need to return a sequence of actions for pacman to execute

    # Add to KB: initial knowledge: pacman's initial location at time 0
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time=0))

    for t in range(50):  # '50' is said in the description

        # Print time step to see code running
        print("timestep: {}".format(t))
        # t starts at 0, cuz we do have actions and transition to 1 in 0th timestep!!!

        # Add to KB: Pacman can only be at exactly one of the locations in non_wall_coords at time t
        KB.append(exactlyOne([PropSymbolExpr(pacman_str, x, y, time=t) for x, y in non_wall_coords]))

        # Now we use findModel to check if ...
        # there exist a satisfying assignment for the knowledge base so far?
        curr_model_sol: Union[Dict[Expr, bool], bool] = findModel(
            conjoin(KB + [PropSymbolExpr(pacman_str, xg, yg, time=t)]))
        # If there is, return a sequence of actions from start to goal using extractActionSequence.
        # Here, Goal Assertion is the expression asserting that Pacman is at the goal at timestep t.
        if curr_model_sol:
            return extractActionSequence(curr_model_sol, actions=actions)

        # Add to KB: Pacman takes exactly one action per timestep.
        KB.append(exactlyOne([PropSymbolExpr(action, time=t) for action in actions]))

        # Add to KB: Transition Model sentences:
        #            call pacmanSuccessorAxiomSingle(...) for all possible pacman positions in non_wall_coords.
        # Here we don't use exactlyOne to choose only one possible transition.
        # Instead, we choose all possible transitions because all transitions are valid everytime in transition model
        for x, y in non_wall_coords:
            KB.append(pacmanSuccessorAxiomSingle(x, y, t + 1, walls_grid=walls_grid))

    "*** END YOUR CODE HERE ***"


# ______________________________________________________________________________
# QUESTION 5

# logicAgents.FoodPlanningProblem
def foodLogicPlan(problem) -> List:
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all the food.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    Overview: add knowledge incrementally, and query for a model each timestep. Do NOT use pacphysicsAxioms.
    """
    walls: game.Grid = problem.walls
    width: int = problem.getWidth()
    height: int = problem.getHeight()
    walls_list: List[Tuple[int, int]] = walls.asList()
    x0, y0 = problem.start[0]
    food: List[Tuple[int, int]] = problem.start[1].asList()

    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords: List[Tuple[int, int]] = list(itertools.product(range(width + 2), range(height + 2)))

    non_wall_coords: List[Tuple[int, int]] = [loc for loc in all_coords if loc not in walls_list]
    actions: List[str] = ['North', 'South', 'East', 'West']

    KB: List[Expr] = []

    "*** BEGIN YOUR CODE HERE ***"

    # Same as previous question,
    # Add to KB: initial knowledge: pacman's initial location at time 0
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time=0))

    # Init Food[x,y]_t variables with PropSymbolExpr(food_str, x, y, time=0),
    # where each variable is true IFF there's food at (x,y) at time t
    for x, y in food:
        KB.append(PropSymbolExpr(food_str, x, y, time=0))

    for t in range(50):
        print("timestep: {}".format(t))

        # Goal assertion is True IFF all food have been eaten.
        # This happens when all Food[x,y]_t are false.

        food_expr: List[PropSymbolExpr] = []  # create a list of food expressions Food[x,y]_t
        for x, y in food:
            food_expr.append(PropSymbolExpr(food_str, x, y, time=t))

            # Food[x,y]_t+1  <=>  (Food[x,y]_t and not pacman[x,y]_t
            KB.append(PropSymbolExpr(food_str, x, y, time=t + 1) %
                      (PropSymbolExpr(food_str, x, y, time=t) & (~PropSymbolExpr(pacman_str, x, y, time=t))))

        # (same as the above function)
        curr_model_sol: Union[Dict[Expr, bool], bool] = findModel(conjoin(KB + [~ disjoin(food_expr)]))

        if curr_model_sol:  # If there is an assignment for the model, return
            # return a sequence of actions from start to goal using extractActionSequence.
            return extractActionSequence(curr_model_sol, actions=actions)

        # Add to KB: Pacman takes exactly one action per timestep.
        KB.append(exactlyOne([PropSymbolExpr(pacman_str, x, y, time=t) for x, y in non_wall_coords]))

        # Add to KB: Transition Model sentences:
        #            call pacmanSuccessorAxiomSingle(...) for all possible pacman positions in non_wall_coords.
        # Here we don't use exactlyOne to choose only one possible transition.
        # Instead, we choose all possible transitions because all transitions are valid everytime in transition model
        KB.append(exactlyOne([PropSymbolExpr(action, time=t) for action in actions]))
        for x, y in non_wall_coords:
            KB.append(pacmanSuccessorAxiomSingle(x, y, t + 1, walls_grid=walls))

    "*** END YOUR CODE HERE ***"


# ______________________________________________________________________________
# QUESTION 6

# problem: logicAgents.LocalizationProblem, agent: logicAgents.LocalizationLogicAgent
def localization(problem, agent) -> Generator:
    """
    problem: a LocalizationProblem instance
    agent: a LocalizationLogicAgent instance
    """
    walls_grid: game.Grid = problem.walls
    walls_list: List[Tuple[int, int]] = walls_grid.asList()
    all_coords: List[Tuple[int, int]] = list(
        itertools.product(range(problem.getWidth() + 2), range(problem.getHeight() + 2)))
    non_outer_wall_coords: List[Tuple[int, int]] = list(
        itertools.product(range(1, problem.getWidth() + 1), range(1, problem.getHeight() + 1)))

    KB: List[Expr] = []

    "*** BEGIN YOUR CODE HERE ***"
    # findModel means that it's possible that pacman is/ isn't at (x, y). Entails means that it's guaranteed that
    # pacman is/ isn't at (x,y). We gradually find probable information. A simple example is that if we do findModel
    # on A, and do it again on not A, we can get true both times, and can't get two falses. On entails we can get two
    # falses but can't get two trues.

    # you should use ENTAILS because ENTAILS proves something.
    # Why the other one is wrong: >> creates a logical statement, that by itself doesn't do anything

    # Use entails(a,b), if return true, from a know KB, then it's guaranteed to be true If use '>>', you are only
    # writing an Expr expression, 'KB >> need_to_be_proved', this could be true by setting KB false which is not what
    # we wanted, but ENTAILS() helps us guarantee if 'need_to_be_proved' is true, knowing KB is true.

    # Add to KB: where the walls are (walls_list) and aren't in (walls_list)
    for x, y in all_coords:
        if (x, y) in walls_list:
            KB.append(PropSymbolExpr(wall_str, x, y))
        else:
            KB.append(~PropSymbolExpr(wall_str, x, y))

    for t in range(agent.num_timesteps):

        # Add pacPhysics to KB
        KB.append(pacphysicsAxioms(t, all_coords=all_coords, non_outer_wall_coords=non_outer_wall_coords,
                                   walls_grid=walls_grid, sensorModel=sensorAxioms,
                                   successorAxioms=allLegalSuccessorAxioms))

        # Add action to KB
        KB.append(PropSymbolExpr(agent.actions[t], time=t))

        # Add percept information to KB
        percepts: Tuple[bool, bool, bool, bool] = agent.getPercepts()  # tells whether there is a wall on 4 directions
        KB.append(fourBitPerceptRules(t, percepts))

        # find possible pacman locations with updated KB
        possible_locations: List[Tuple[int, int]] = []
        for x, y in non_outer_wall_coords:
            # 'entails' proves something for sure is true

            # guaranteed pacman is at x, y at time t?
            at_x_y: bool = entails(conjoin(KB), PropSymbolExpr(pacman_str, x, y, time=t))
            # guaranteed pacman is not at x, y at time t?
            not_at_x_y: bool = entails(conjoin(KB), (~PropSymbolExpr(pacman_str, x, y, time=t)))
            # if at_x_y is false, we can't prove there's a guaranteed pacman at x,y,t
            # if not_x_y is false, we can't prove there's guaranteed no pacman
            # if at_x_y and not_x_y are both false:
            #     if we can't prove that pacman is not there,
            #     then it's possible for him to be there,
            #     and we can't prove it's always there

            # if it's not unsatisfiable, where it has a satisfying assignment. Add it to possible position of pacman
            if not not_at_x_y:
                possible_locations.append((x, y))

            if at_x_y:
                KB.append(PropSymbolExpr(pacman_str, x, y, time=t))  # add that pacman is valid at there at time t
            if not_at_x_y:
                KB.append(~PropSymbolExpr(pacman_str, x, y, time=t))  # add that pacman unsatisfiable at there at time t

        # call agent.moveToNextState(action_t) on the current agent action at timestep t
        agent.moveToNextState(agent.actions[t])
        "*** END YOUR CODE HERE ***"

        # yield the possible locations
        yield possible_locations


# ______________________________________________________________________________
# QUESTION 7

def mapping(problem, agent) -> Generator:
    """
    problem: a MappingProblem instance
    agent: a MappingLogicAgent instance
    """
    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth() + 2), range(problem.getHeight() + 2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth() + 1), range(1, problem.getHeight() + 1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight() + 2)] for x in range(problem.getWidth() + 2)]

    # Pacman knows that the outer border of squares are all walls
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    "*** BEGIN YOUR CODE HERE ***"

    # Given 'problem' and 'logic agent', yields for knowledge about the map at t.

    # Get initial location (pac_x_0, pac_y_0) and add to KB.
    KB.append(PropSymbolExpr(pacman_str, pac_x_0, pac_y_0, time=0))
    # Also add whether there is a wall (there isn't a wall) at that location. Starting location is sure to be no walls.
    KB.append(~PropSymbolExpr(wall_str, pac_x_0, pac_y_0))

    # We know the starting location
    known_map[pac_x_0][pac_y_0] = 0

    for t in range(agent.num_timesteps):

        # (same as the above function)
        # Add pacPhysics information to KB
        KB.append(pacphysicsAxioms(t, all_coords=all_coords, non_outer_wall_coords=non_outer_wall_coords,
                                   walls_grid=known_map, sensorModel=sensorAxioms,
                                   successorAxioms=allLegalSuccessorAxioms))

        # add action information to KB
        KB.append(PropSymbolExpr(agent.actions[t], time=t))

        # add percept information to KB
        percepts: Tuple[bool, bool, bool, bool] = agent.getPercepts()  # tells whether there is a wall on 4 directions
        KB.append(fourBitPerceptRules(t, percepts))

        # find provable wall locations with updated KB
        for x, y in non_outer_wall_coords:

            # guarantee that there is a wall at x,y ?
            wall_at_x_y: bool = entails(conjoin(KB), PropSymbolExpr(wall_str, x, y))

            # guarantee that there isn't a wall at x,y ?
            wall_not_x_y: bool = entails(conjoin(KB), (~PropSymbolExpr(wall_str, x, y)))

            if wall_at_x_y:
                # If we are sure there is a wall at x, y,
                # it is a 'provable wall location', so add it to KB and known_map
                KB.append(PropSymbolExpr(wall_str, x, y))
                known_map[x][y] = 1
            if wall_not_x_y:
                # If we are sure there isn't a wall at x, y
                # it is also 'provable'.
                KB.append(~PropSymbolExpr(wall_str, x, y))
                known_map[x][y] = 0
            # In other case, we can't prove there's a wall, and we can't prove there isn't a wall.
            # we don't do anything

        # call agent.moveToNextState(action_t) on the current agent action at time t.
        agent.moveToNextState(agent.actions[t])

        "*** END YOUR CODE HERE ***"
        yield known_map


# ______________________________________________________________________________
# QUESTION 8

def slam(problem, agent) -> Generator:
    """
    problem: a SLAMProblem instance
    agent: a SLAMLogicAgent instance
    """
    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth() + 2), range(problem.getHeight() + 2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth() + 1), range(1, problem.getHeight() + 1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight() + 2)] for x in range(problem.getWidth() + 2)]

    # We know that the outer_coords are all walls.
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    "*** BEGIN YOUR CODE HERE ***"

    # get initial location of pacman and add this to KB.
    # the starting position is sure to have a pacman and have no walls
    KB.append(PropSymbolExpr(pacman_str, pac_x_0, pac_y_0, time=0))
    KB.append(~PropSymbolExpr(wall_str, pac_x_0, pac_y_0))
    known_map[pac_x_0][pac_y_0] = 0

    for t in range(agent.num_timesteps):

        # add pacPhysics to KB
        KB.append(pacphysicsAxioms(t, all_coords=all_coords, non_outer_wall_coords=non_outer_wall_coords,
                                   walls_grid=known_map, sensorModel=SLAMSensorAxioms,
                                   successorAxioms=SLAMSuccessorAxioms))

        # add actions to KB
        KB.append(PropSymbolExpr(agent.actions[t], time=t))

        # add percept information to KB
        percepts: Tuple[bool, bool, bool, bool] = agent.getPercepts()  # tells whether there is a wall on 4 directions
        KB.append(numAdjWallsPerceptRules(t, percepts))

        # (same as Question 7): Find provable wall locations with updated KB.
        for x, y in non_outer_wall_coords:

            # guarantee there's a wall at x,y?
            wall_at_x_y: bool = entails(conjoin(KB), PropSymbolExpr(wall_str, x, y))

            # guarantee there isn't a wall at x,y?
            wall_not_x_y: bool = entails(conjoin(KB), (~PropSymbolExpr(wall_str, x, y)))

            if wall_at_x_y:  # there must be a wall at x,y. Provable
                KB.append(PropSymbolExpr(wall_str, x, y))
                known_map[x][y] = 1

            if wall_not_x_y:  # there must be no wall at x,y. Also provable
                KB.append(~PropSymbolExpr(wall_str, x, y))
                known_map[x][y] = 0

        # (same as Question 6): Find possible pacman locations with updated KB.
        possible_locations: List[Tuple[int, int]] = []

        for x, y in non_outer_wall_coords:

            # guaranteed pacman is at x, y at time t?
            at_x_y: bool = entails(conjoin(KB), PropSymbolExpr(pacman_str, x, y, time=t))

            # guaranteed pacman is not at x, y at time t?
            not_at_x_y: bool = entails(conjoin(KB), (~PropSymbolExpr(pacman_str, x, y, time=t)))

            if not not_at_x_y:
                possible_locations.append((x, y))

            if at_x_y:
                KB.append(PropSymbolExpr(pacman_str, x, y, time=t))

            if not_at_x_y:
                KB.append(~PropSymbolExpr(pacman_str, x, y, time=t))

        # Call agent.moveToNextState(action_t) on the current agent action at timestep t.
        agent.moveToNextState(agent.actions[t])

        "*** END YOUR CODE HERE ***"
        yield known_map, possible_locations


# Abbreviations
plp = positionLogicPlan
loc = localization
mp = mapping
flp = foodLogicPlan
# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)


# ______________________________________________________________________________
# Important expression generating functions, useful to read for understanding of this project.


def sensorAxioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, x + dx, y + dy, time=t)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (
                    PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], time=t)
        all_percept_exprs.append(percept_unit_clause % disjoin(percept_exprs))

    return conjoin(all_percept_exprs + combo_var_def_exprs)


def fourBitPerceptRules(t: int, percepts: List) -> Expr:
    """
    Localization and Mapping both use the 4 bit sensor, which tells us True/False whether
    a wall is to pacman's north, south, east, and west.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 4, "Percepts must be a length 4 list."

    percept_unit_clauses = []
    for wall_present, direction in zip(percepts, DIRECTIONS):
        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], time=t)
        if not wall_present:
            percept_unit_clause = ~PropSymbolExpr(blocked_str_map[direction], time=t)
        percept_unit_clauses.append(percept_unit_clause)  # The actual sensor readings
    return conjoin(percept_unit_clauses)


def numAdjWallsPerceptRules(t: int, percepts: List) -> Expr:
    """
    SLAM uses a weaker numAdjWallsPerceptRules sensor, which tells us how many walls pacman is adjacent to
    in its four directions.
        000 = 0 adj walls.
        100 = 1 adj wall.
        110 = 2 adj walls.
        111 = 3 adj walls.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 3, "Percepts must be a length 3 list."

    percept_unit_clauses = []
    for i, percept in enumerate(percepts):
        n = i + 1
        percept_literal_n = PropSymbolExpr(geq_num_adj_wall_str_map[n], time=t)
        if not percept:
            percept_literal_n = ~percept_literal_n
        percept_unit_clauses.append(percept_literal_n)
    return conjoin(percept_unit_clauses)


def SLAMSensorAxioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, x + dx, y + dy, time=t)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(
                combo_var % (PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        blocked_dir_clause = PropSymbolExpr(blocked_str_map[direction], time=t)
        all_percept_exprs.append(blocked_dir_clause % disjoin(percept_exprs))

    percept_to_blocked_sent = []
    for n in range(1, 4):
        wall_combos_size_n = itertools.combinations(blocked_str_map.values(), n)
        n_walls_blocked_sent = disjoin([
            conjoin([PropSymbolExpr(blocked_str, time=t) for blocked_str in wall_combo])
            for wall_combo in wall_combos_size_n])
        # n_walls_blocked_sent is of form: (N & S) | (N & E) | ...
        percept_to_blocked_sent.append(
            PropSymbolExpr(geq_num_adj_wall_str_map[n], time=t) % n_walls_blocked_sent)

    return conjoin(all_percept_exprs + combo_var_def_exprs + percept_to_blocked_sent)


def allLegalSuccessorAxioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    """walls_grid can be a 2D array of ints or bools."""
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = pacmanSuccessorAxiomSingle(
            x, y, t, walls_grid)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)


def SLAMSuccessorAxioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    """walls_grid can be a 2D array of ints or bools."""
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = SLAMSuccessorAxiomSingle(
            x, y, t, walls_grid)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)


# ______________________________________________________________________________
# Various useful functions, are not needed for completing the project but may be useful for debugging


def modelToString(model: Dict[Expr, bool]) -> str:
    """Converts the model to a string for printing purposes. The keys of a model are 
    sorted before converting the model to a string.
    
    model: Either a boolean False or a dictionary of Expr symbols (keys) 
    and a corresponding assignment of True or False (values). This model is the output of 
    a call to pycoSAT.
    """
    if not model:
        return "False"
    else:
        # Dictionary
        model_list = sorted(model.items(), key=lambda item: str(item[0]))
        return str(model_list)


def extractActionSequence(model: Dict[Expr, bool], actions: List) -> List:
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[2]":True, "P[3,4,0]":True, "P[3,3,0]":False, "West[0]":True, "GhostScary":True, "West[2]":False, "South[1]":True, "East[0]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print(plan)
    ['West', 'South', 'North']
    """
    plan = [None for _ in range(len(model))]
    for sym, val in model.items():
        parsed = parseExpr(sym)
        if type(parsed) == tuple and parsed[0] in actions and val:
            action, _, time = parsed
            plan[time] = action
    # return list(filter(lambda x: x is not None, plan))
    return [x for x in plan if x is not None]


# Helpful Debug Method
def visualizeCoords(coords_list, problem) -> None:
    wall_grid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    for (x, y) in itertools.product(range(problem.getWidth() + 2), range(problem.getHeight() + 2)):
        if (x, y) in coords_list:
            wall_grid.data[x][y] = True
    print(wall_grid)


# Helpful Debug Method
def visualizeBoolArray(bool_arr, problem) -> None:
    wall_grid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    wall_grid.data = copy.deepcopy(bool_arr)
    print(wall_grid)


class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()

    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()
