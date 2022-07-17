from typing import List

import networkx
from collections import deque

def initIR(Graph, p):
    sigma = [1 for x in range(Graph.number_of_nodes())]
    adj = networkx.adjacency_matrix(Graph) * p
    sum = sigma
    currAdj = adj
    for x in range(0, max([max(j.values()) for (i,j) in networkx.shortest_path_length(Graph)])
):
        sum = sum + currAdj*sigma
        currAdj = currAdj*adj
    d = {}
    for node in list(Graph.nodes()):
        d[node] = sum[list(Graph.nodes()).index(node)]
    return d

def updateIR(sigma, edge, Graph: networkx.DiGraph, p, topologicalSort: List):
    updates = {edge[0]: [sigma[edge[1]] * p]}
    for node in reversed(topologicalSort):
        if node not in updates:
            continue
        for update in updates[node]:
            sigma[node] = sigma[node] + update
        for pred in Graph.predecessors(node):
            if pred not in updates:
                updates[pred] = []
            updates[pred].append(sum([update*p for update in updates[node]]))
    return sigma

def initIC(Graph: networkx.DiGraph, seeds, p, topologicalSort):
    ic = {}
    for x in list(Graph.nodes()):
        ic[x] = 0
    q = deque()
    updates = {}
    for seed in seeds:
        ic[seed] = 1
        for neighbor in list(Graph.neighbors(seed)):
            if neighbor not in updates:
                updates[neighbor] = []
            updates[neighbor].append((0, 1))
    for node in topologicalSort:
        if node not in updates or node in seeds:
            continue
        for update in updates[node]:
            ic[node] = ic[node] + (1 - ic[node]) / (1 - p * update[0]) * p * (update[1] - update[0])
        for neighbor in list(Graph.neighbors(node)):
            if neighbor not in updates:
                updates[neighbor] = []
            updates[neighbor].append((0, ic[node]))
    return ic

def updateIC(ic, edge, Graph, p, topologicalSort):
    prevQ = ic[edge[1]]
    ic[edge[1]] = ic[edge[1]] + (1 - ic[edge[1]])*p*ic[edge[0]]
    updates = {}
    for neighbor in Graph.neighbors(edge[1]):
        if neighbor not in updates:
            updates[neighbor] = []
        updates[neighbor].append((prevQ, ic[edge[1]]))
    for node in topologicalSort:
        if node not in updates:
            continue
        prevQ = ic[node]
        for update in updates[node]:
            ic[node] = ic[node] + (1 - ic[node]) / (1 - p * update[0]) * p * (update[1] - update[0])
        for neighbor in list(Graph.neighbors(node)):
            if neighbor not in updates:
                updates[neighbor] = []
            updates[neighbor].append((prevQ, ic[node]))
    return ic

def IRFA(Graph: networkx.DiGraph, seeds, k, candidateEdges, p):
    sigma = initIR(Graph, p)
    topSort = list(networkx.topological_sort(Graph))
    q = initIC(Graph, seeds, p, topSort)
    kPerNode = {}
    selectedEdges = []
    for node in list(Graph.nodes()):
        kPerNode[node] = 0
    while len(kPerNode) > 0 and len(candidateEdges) > 0:
        maxVal = 0
        maxEdge = None
        for node in candidateEdges:
            for newNeighbor in candidateEdges[node].copy():
                if newNeighbor not in kPerNode:
                    candidateEdges[node].remove(newNeighbor)
                    continue
                curr = (1 - q[newNeighbor])*sigma[newNeighbor]
                if curr > maxVal:
                    maxVal = curr
                    maxEdge = (node, newNeighbor)
        if maxEdge == None:
            break
        selectedEdges.append(maxEdge)
        sigma = updateIR(sigma, maxEdge, Graph, p, topSort)
        Graph.add_edge(maxEdge[0], maxEdge[1])
        q = updateIC(q, maxEdge, Graph, p, topSort)
        candidateEdges[maxEdge[0]].remove(maxEdge[1])
        if len(candidateEdges[maxEdge[0]]) == 0:
            del candidateEdges[maxEdge[0]]
        kPerNode[maxEdge[0]] = kPerNode[maxEdge[0]] + 1
        if kPerNode[maxEdge[0]] == k:
            if maxEdge[0] in candidateEdges:
                del candidateEdges[maxEdge[0]]
            del kPerNode[maxEdge[0]]
        kPerNode[maxEdge[1]] = kPerNode[maxEdge[1]] + 1
        if kPerNode[maxEdge[1]] == k:
            if maxEdge[1] in candidateEdges:
                del candidateEdges[maxEdge[1]]
            del kPerNode[maxEdge[1]]
    return selectedEdges


def removeCylces(Graph, sources):
    reachableFromSource = {}


def testInitIR():
    Graph = networkx.DiGraph()
    Graph.add_edges_from([(3,1),(3,2),(4,3),(4,5),(6,5),(7,6),(7,4)])
    print(initIR(Graph, .5))

def testInitIC():
    Graph = networkx.DiGraph()
    Graph.add_edges_from([(3,1),(3,2),(4,3),(4,5),(6,5),(7,6),(7,4)])
    print(initIC(Graph, [7, 3], .5, networkx.topological_sort(Graph)))


def testUpdateIR():
    Graph = networkx.DiGraph()
    Graph.add_edges_from([(3,2),(4,3),(4,5),(6,5),(7,6),(7,4)])
    Graph.add_node(1)
    sigma = initIR(Graph, .5)
    print(sigma)
    print(updateIR(sigma, (3,1), Graph, .5, list(networkx.topological_sort(Graph))))

def testUpdateIC():
    Graph = networkx.DiGraph()
    Graph.add_edges_from([(3,2),(3,1),(4,5),(6,5),(7,6),(7,4)])
    q = initIC(Graph, [7], .5, networkx.topological_sort(Graph))
    print(q)
    print(updateIC(q, (4,3), Graph, .5, networkx.topological_sort(Graph)))

def testIRFA():
    Graph = networkx.DiGraph()
    Graph.add_edges_from([(3,1),(3,2),(4,3),(4,5),(6,5),(7,6),(7,4)])
    candidateEdges = {}
    candidateEdges[7] = set((5, 3))
    print(IRFA(Graph, [7], 1, candidateEdges, .5))

testInitIR()
testInitIC()
testUpdateIR()
testUpdateIC()
testIRFA()