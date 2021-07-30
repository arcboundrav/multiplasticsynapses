import networkx as nx
import numpy as np

#############
# CONSTANTS #
#############
P1 = 0.3
Q1 = 0.4

P2 = P1**2.0
P3 = P1**3.0

Q2 = Q1**((2.0*3.0)/3.0)
Q3 = Q1**((2.0*4.0)/3.0)
Q4 = Q1**((2.0*5.0)/3.0)

STARTING_N = 100
PROPERTY_DICT_IDX = -1
SRC_IDX = 0
DST_IDX = 1

POTENTIATION_EDGE_LIST = [
    "w1->s1",
    "w2->s1",
    "w3->s1",
    "w4->s1",
    "s1->s2",
    "w2->w1",
    "s2->s3",
    "w3->w2",
    "s3->s4",
    "w4->w3"
]

DEPRESSION_EDGE_LIST = [
    "s1->w1",
    "s2->w1",
    "s3->w1",
    "s4->w1",
    "w1->w2",
    "s2->s1",
    "w2->w3",
    "s3->s2",
    "w3->w4",
    "s4->s3"
]

INDICES_AND_LABELS = [
    (0, 'w4'),
    (1, 'w3'),
    (2, 'w2'),
    (3, 'w1'),
    (4, 's1'),
    (5, 's2'),
    (6, 's3'),
    (7, 's4')
]

l2i = {}
i2l = {}
for t in INDICES_AND_LABELS:
    l2i[t[1]] = t[0]
    i2l[t[0]] = t[1]
######################################################################################################

def multiplastic_synapse_factory():
    G = nx.DiGraph()
    for t in INDICES_AND_LABELS:
        G.add_node(t[0], label=t[1], N=STARTING_N)

    def add_weighted_edge(graph, source_name, target_name, weight, symmetric=False):
        label = "{}->{}".format(source_name, target_name)
        graph.add_edge(l2i[source_name], l2i[target_name], weight=weight, label=label)
        if symmetric:
            label = "{}->{}".format(target_name, source_name)
            graph.add_edge(l2i[target_name], l2i[source_name], weight=weight, label=label)

    add_weighted_edge(G, 'w1', 's1', Q1, symmetric=True)
    add_weighted_edge(G, 'w1', 'w2', P1, symmetric=True)
    add_weighted_edge(G, 's1', 's2', P1, symmetric=True)
    add_weighted_edge(G, 'w2', 'w3', P2, symmetric=True)
    add_weighted_edge(G, 's2', 's3', P2, symmetric=True)
    add_weighted_edge(G, 'w3', 'w4', P3, symmetric=True)
    add_weighted_edge(G, 's3', 's4', P3, symmetric=True)
    add_weighted_edge(G, 'w4', 's1', Q4)
    add_weighted_edge(G, 's4', 'w1', Q4)
    add_weighted_edge(G, 'w3', 's1', Q3)
    add_weighted_edge(G, 's3', 'w1', Q3)
    add_weighted_edge(G, 'w2', 's1', Q2)
    add_weighted_edge(G, 's2', 'w1', Q2)
    return G


def get_edge_by_prop(graph, property, value):
    return [e for e in graph.edges(data=True) if (e[PROPERTY_DICT_IDX][property] == value)][0]


def get_node_by_prop(graph, property, value):
    return [n for n in graph.nodes(data=True) if (n[PROPERTY_DICT_IDX][property] == value)][0]


def get_graph_weak_synapses(graph):
    return [n for n in graph.nodes(data=True) if (n[PROPERTY_DICT_IDX]['label'].startswith('w'))]


def get_graph_strong_synapses(graph):
    return [n for n in graph.nodes(data=True) if (n[PROPERTY_DICT_IDX]['label'].startswith('s'))]


def sum_graph_strong_synapses(graph):
    return sum(synapse[PROPERTY_DICT_IDX]['N'] for synapse in get_graph_strong_synapses(graph))


def sum_graph_weak_synapses(graph):
    return sum(synapse[PROPERTY_DICT_IDX]['N'] for synapse in get_graph_weak_synapses(graph))


def strong_synapse_fraction(graph):
    N_strong = sum_graph_strong_synapses(graph)
    N_weak = sum_graph_weak_synapses(graph)
    denominator = N_strong + N_weak
    return N_strong / denominator


def roll(probability):
    if (np.random.random() < probability):
        return True
    return False


def change_metastate_of_synapses(graph, edge_label):
    edge = get_edge_by_prop(graph, 'label', edge_label)
    weight = edge[PROPERTY_DICT_IDX]['weight']
    source_node = get_node_by_prop(graph, 'label', i2l[edge[SRC_IDX]])
    target_node = get_node_by_prop(graph, 'label', i2l[edge[DST_IDX]])

    source_node_N = source_node[PROPERTY_DICT_IDX]['N']
    n_to_donate = 0
    for i in range(source_node_N):
        if roll(weight):
            n_to_donate += 1

    source_node[PROPERTY_DICT_IDX]['N'] -= n_to_donate
    target_node[PROPERTY_DICT_IDX]['N'] += n_to_donate


def update_graph(graph, edge_labels):
    for edge_label in edge_labels:
        change_metastate_of_synapses(graph, edge_label)


def potentiate_graph(graph, edge_labels=POTENTIATION_EDGE_LIST):
    update_graph(graph, edge_labels)


def depress_graph(graph, edge_labels=DEPRESSION_EDGE_LIST):
    update_graph(graph, edge_labels)


class Model:
    def __init__(self, sigma=0.1):
        self.sigma = 0.1
        self.left = multiplastic_synapse_factory()
        self.right = multiplastic_synapse_factory()
        self.left_idx = 0
        self.right_idx = 1

    def react_to_outcome(self, side_idx):
        if (side_idx == self.left_idx):
            potentiate_graph(self.left)
            depress_graph(self.right)
        elif (side_idx == self.right_idx):
            potentiate_graph(self.right)
            depress_graph(self.left)

    def choose_side(self):
        strong_synapse_frac_diff = strong_synapse_fraction(self.left) - strong_synapse_fraction(self.right)
        strong_synapse_frac_diff /= self.sigma
        chance_left_chosen = (1 + np.exp(-strong_synapse_fraction_difference))**-1
        if roll(chance_left_chosen):
            return self.left_idx
        return self.right_idx

    def handle_trials(self, list_of_trials):
        guesses = []
        results = []
        for trial in list_of_trials:
            guess = self.choose_side()
            guesses.append(guess)
            self.react_to_outcome(trial)
            if (guess == trial):
                results.append(1)
            else:
                results.append(0)
        return guesses, results


def generate_trials(N=100):
    return np.random.randint(2, size=N)


def test(trials=None):
    if (trials is None):
        trials = generate_trials()
    g, r = M.handle_trials(trials)
    print("{}% correct.".format(np.round((np.mean(r)*100), 2)))


M = Model()
T = generate_trials()
T2 = [0] * 10
T2+= [1] * 10
T2+= [0] * 10
