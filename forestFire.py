import random
import numpy as np
import networkx as nx
from collections import deque
import shortestDistances as sd

def start_a_fire(graph, number_of_nodes, seed=42):
        """
        Starting a forest fire from a single node.
        """
        sampled_nodes = set()
        set_of_nodes = set(list(graph.nodes()))
        visited_nodes = deque(maxlen=100)
        while len(sampled_nodes) < number_of_nodes:
            remaining_nodes = list(set_of_nodes.difference(sampled_nodes))
            seed_node = random.choice(remaining_nodes)
            sampled_nodes.add(seed_node)
            node_queue = deque([seed_node])
            while len(sampled_nodes) < number_of_nodes:
                if len(node_queue) == 0:
                    node_queue = deque(
                        [
                            visited_nodes.popleft()
                            for k in range(
                                min(10, len(visited_nodes))
                            )
                        ]
                    )
                    if len(node_queue) == 0:
                        print(
                            "Warning: could not collect the required number of nodes. The fire could not find enough nodes to burn."
                        )
                        break
                top_node = node_queue.popleft()
                sampled_nodes.add(top_node)
                node = top_node
                neighbors = set(graph.neighbors(node))
                unvisited_neighbors = neighbors.difference(sampled_nodes)
                score = np.random.geometric(.4)
                count = min(len(unvisited_neighbors), score)
                burned_neighbors = random.sample(unvisited_neighbors, count)
                visited_nodes.extendleft(
                    unvisited_neighbors.difference(set(burned_neighbors))
                )
                for neighbor in burned_neighbors:
                    if len(sampled_nodes) >= number_of_nodes:
                        break
                    node_queue.extend([neighbor])
        return nx.subgraph(graph, sampled_nodes).copy()


class MultipleSourceContinuingFF:
    def __init__(self, G, sources, nodesPerIteration, shortestDistances):
        self.originalGraph = G.to_directed()
        self.sources = sources
        self.nodesPerIteration = nodesPerIteration
        self.forestFires = {}
        self.terminatedForestFires = []
        self.firstIteration = True
        self.shortestDistances = shortestDistances
        self.sampled_nodes = {}
        self.subgraph = nx.DiGraph()

        for source in sources:
            forestFire = {}
            forestFire['sampled_nodes'] = set()
            forestFire['set_of_nodes'] = set(list(self.originalGraph.nodes()))
            forestFire['seed_node'] = source
            forestFire['node_queue_dict'] = {}
            forestFire['node_queue_dict'][source] = True
            forestFire['node_queue'] = deque([forestFire['seed_node']])
            forestFire['visited_nodes'] = deque()
            forestFire['visited_nodes_dict'] = {}
            forestFire['visited_nodes'] = deque()
            forestFire['nodesLastIteration'] = 0
            self.forestFires[source] = forestFire

    def updateShortestDistances(shortestDistances):
        self.shortestDistances = shortestDistances

    def alternatingForestFire(self):
        iterationNodes = []
        if self.firstIteration:
            nodesThisIteration = len(self.sources)
            self.firstIteration = False
        else:
            nodesThisIteration = 0

        while len(self.terminatedForestFires) < len(self.sources) and nodesThisIteration < self.nodesPerIteration :
            for source in [x for x in self.sources if x not in self.terminatedForestFires]:
                while True:
                    if len(self.forestFires[source]['node_queue']) == 0:
                        self.forestFires[source]['node_queue'] = deque(
                            [
                                self.forestFires[source]['visited_nodes'].popleft()
                                for k in range(
                                    min(10, len(self.forestFires[source]['visited_nodes']))
                                )
                            ]
                        )
                        if len(self.forestFires[source]['node_queue']) == 0:
                            print('Forest fire for ' + str(source) + ' finished with ' + str(len(self.forestFires[source]['sampled_nodes'])) + ' nodes found!')
                            self.terminatedForestFires.append(source)
                            break
                    top_node = self.forestFires[source]['node_queue'].popleft()
                    if top_node not in self.sampled_nodes or top_node == source:
                        break
                if source in self.terminatedForestFires:
                    continue

                if top_node not in self.sampled_nodes:
                    nodesThisIteration = nodesThisIteration + 1
                    self.forestFires[source]['sampled_nodes'].add(top_node)
                    self.sampled_nodes[top_node] = True
                    iterationNodes.append(top_node)
                node = top_node

                neighbors = set(self.originalGraph.neighbors(node))

                unvisited_neighbors = neighbors
                for o in self.sources:
                    unvisited_neighbors = unvisited_neighbors.difference(self.forestFires[o]['sampled_nodes'])
                
                score = np.random.geometric(.4)
                count = min(len(unvisited_neighbors), score)
                burned_neighbors = random.sample(unvisited_neighbors, count)
                
                for x in unvisited_neighbors.difference(set(burned_neighbors)):
                    if x not in self.forestFires[source]['visited_nodes_dict'] and x not in self.sampled_nodes:
                        self.forestFires[source]['visited_nodes_dict'][x] = True
                        self.forestFires[source]['visited_nodes'].extendleft([x])
                for neighbor in burned_neighbors:
                    if neighbor not in self.forestFires[source]['node_queue_dict'] and neighbor not in self.sampled_nodes:
                        self.forestFires[source]['node_queue'].extend([neighbor])
                        self.forestFires[source]['node_queue_dict'][neighbor] = True
        
        for source in self.sources:
            self.forestFires[source]['nodesLastIteration'] = len(self.forestFires[source]['sampled_nodes'])
        subgraphTemp = self.originalGraph.subgraph(self.sampled_nodes.keys())

        for node in iterationNodes:
            self.subgraph.add_node(node)
            for edge in subgraphTemp.in_edges(node):
                self.subgraph.add_edge(edge[0], edge[1])
            for edge in subgraphTemp.out_edges(node):
                self.subgraph.add_edge(edge[0], edge[1])

        return (self.subgraph, len(self.terminatedForestFires) == len(self.sources), iterationNodes) 
        


def test():
    G = nx.random_regular_graph(6, 20)
    nx.draw(G, with_labels=True)

    sources = [random.sample(list(G.nodes()), 1 ), None]
    print(sources)

    shortestDistances = sd.multipleSourceShortestDistances(G, sources[0])

    mscff = MultipleSourceContinuingFF(G, sources[0], 3, shortestDistances)

    finished = False
    while not finished:
        subgraph, finished, _ = mscff.alternatingForestFire()
        nx.draw(subgraph, with_labels=True)

# test()
