
#Frankencode from search4e.ipynb and various othr sources 


import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import sys
import random

class Problem(object):
    """The abstract class for a search problem."""

    def __init__(self, initial=None, goals=(), **additional_keywords):
        """Provide an initial state and optional goal states.
        A subclass can have additional keyword arguments."""
        self.initial = initial  # The initial state of the problem.
        self.goals = goals      # A collection of possible goal states.
        self.__dict__.update(**additional_keywords)

    def actions(self, state):
        "Return a list of actions executable in this state."
        raise NotImplementedError # Override this!

    def result(self, state, action):
        "The state that results from executing this action in this state."
        raise NotImplementedError # Override this!

    def is_goal(self, state):
        "True if the state is a goal." 
        return state in self.goals # Optionally override this!

    def step_cost(self, state, action, result=None):
        "The cost of taking this action from this state."
        return 1 # Override this if actions have different costs        

    
class TSP_problem(Problem):

    '''
    subclass of Problem to define various functions 
    '''

    def two_opt(self, state):
        '''
        Neighbour generating function for Traveling Salesman Problem
        '''
        state2 = state[:]
        l = random.randint(0, len(state2) - 1)
        r = random.randint(0, len(state2) - 1)
        if l > r:
            l, r = r,l
        state2[l : r + 1] = reversed(state2[l : r + 1])
        return state2

    def actions(self, state):
        '''
        action that can be excuted in given state
        '''
        return [self.two_opt] #this is a strange call, but it looks back to the Node 
    
    def result(self, state, action):
        '''
        result after applying the given action on the given state
        '''
        return action(state)

    def path_cost(self, c, state1, action, state2):
        '''
        total distance for the Traveling Salesman to be covered if in state2
        '''
        cost = 0
        for i in range(len(state2) - 1):
            cost += distances[state2[i]][state2[i + 1]]
        cost += distances[state2[0]][state2[-1]]
        return cost
 
    def value(self, state):
        '''
        value of path cost given negative for the given state
        '''
        return -1 * self.path_cost(None, None, None, state)

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

def probability(p):
    """Return true with probability p."""
    return p > random.uniform(0.0, 1.0)

def action_sequence(node):
    "The sequence of actions to get to this node."
    actions = []
    while node.previous:
        actions.append(node.action)
        node = node.previous
    return actions[::-1]

def state_sequence(node):
    "The sequence of states to get to this node."
    states = [node.state]
    while node.previous:
        node = node.previous
        states.append(node.state)
    return states[::-1]

def exp_schedule(k=20, lam=0.005, limit=100):
    """One possible schedule function for simulated annealing"""
    return lambda t: (k * 2.718281**(-lam * t) if t < limit else 0)


def simulated_annealing(problem, schedule=exp_schedule()):
    """[Figure 4.5] CAUTION: This differs from the pseudocode as it
    returns a state instead of a Node."""
    current = Node(problem.initial)
    for t in range(sys.maxsize):
        T = schedule(t)
        if T == 0:
            return current.state
        neighbors = current.expand(problem)
        if not neighbors:
            return current.state
        next_choice = random.choice(neighbors)
        delta_e = problem.value(next_choice.state) - problem.value(current.state)
        if delta_e > 0 or probability(2.718281**(delta_e / T)):
            current = next_choice

def simulated_annealing_full(problem, schedule=exp_schedule()):
    """ This version returns all the states encountered in reaching 
    the goal state."""
    states = []
    current = Node(problem.initial)
    for t in range(sys.maxsize):
        states.append(current.state)
        T = schedule(t)
        if T == 0:
            return states
        neighbors = current.expand(problem)
        if not neighbors:
            return current.state
        next_choice = random.choice(neighbors)
        delta_e = problem.value(next_choice.state) - problem.value(current.state)
        if delta_e > 0 or probability(2.718281**(delta_e / T)):
            current = next_choice

class Graph:
    """A graph connects nodes (vertices) by edges (links). Each edge can also
    have a length associated with it. The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2})
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C. You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added. You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B. 'Lengths' can actually be any object at
    all, and nodes can be any hashable object."""

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)


def UndirectedGraph(graph_dict=None):
    """Build a Graph where every edge (including future ones) goes both ways."""
    return Graph(graph_dict=graph_dict, directed=False)

romania_map = UndirectedGraph(dict(
    Arad=dict(Zerind=75, Sibiu=140, Timisoara=118),
    Bucharest=dict(Urziceni=85, Pitesti=101, Giurgiu=90, Fagaras=211),
    Craiova=dict(Drobeta=120, Rimnicu=146, Pitesti=138),
    Drobeta=dict(Mehadia=75),
    Eforie=dict(Hirsova=86),
    Fagaras=dict(Sibiu=99),
    Hirsova=dict(Urziceni=98),
    Iasi=dict(Vaslui=92, Neamt=87),
    Lugoj=dict(Timisoara=111, Mehadia=70),
    Oradea=dict(Zerind=71, Sibiu=151),
    Pitesti=dict(Rimnicu=97),
    Rimnicu=dict(Sibiu=80),
    Urziceni=dict(Vaslui=142)))
romania_map.locations = dict(
    Arad=(91, 492), Bucharest=(400, 327), Craiova=(253, 288),
    Drobeta=(165, 299), Eforie=(562, 293), Fagaras=(305, 449),
    Giurgiu=(375, 270), Hirsova=(534, 350), Iasi=(473, 506),
    Lugoj=(165, 379), Mehadia=(168, 339), Neamt=(406, 537),
    Oradea=(131, 571), Pitesti=(320, 368), Rimnicu=(233, 410),
    Sibiu=(207, 457), Timisoara=(94, 410), Urziceni=(456, 350),
    Vaslui=(509, 444), Zerind=(108, 531))

cities = []
distances ={}
states = []

# creating initial path
for name in romania_map.locations.keys():    
    distances[name] = {}
    cities.append(name)

# distances['city1']['city2'] contains euclidean distance between their coordinates
for name_1,coordinates_1 in romania_map.locations.items():
    for name_2,coordinates_2 in romania_map.locations.items():
        distances[name_1][name_2] = np.linalg.norm([coordinates_1[0] - coordinates_2[0], coordinates_1[1] - coordinates_2[1]])
        distances[name_2][name_1] = np.linalg.norm([coordinates_1[0] - coordinates_2[0], coordinates_1[1] - coordinates_2[1]])

# creating the problem        
tsp_problem = TSP_problem(cities)

# all the states as a 2-D list of paths
states = simulated_annealing_full(tsp_problem)

# print (states)

for state in states: 
    print (tsp_problem.value(state))
# 🤔 How is it evaluating the path cost? 
# A: it is done in thev problem class, but references an external variable called distance. 
# I will include this into the problem class itself as a lookup.....

# Iteration 
next_state = cities
states = []
# to plot only the final states of every simulated annealing iteration
for iterations in range(100):
    tsp_problem = TSP_problem(next_state)  
    states.append(simulated_annealing(tsp_problem))
    next_state = states[-1]
