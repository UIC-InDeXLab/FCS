import logging
from typing import List

import numpy
import networkx
import numpy as np
import pickle


import generateLPParams

def f(e, omega, q):
    return (q[e[0], 0] - q[e[1], 0])**2/(1 + omega[e[0], e[0]] + omega[e[1], e[1]] - 2*omega[e[0], e[1]])

def computeOmega(G :networkx.Graph, p):
    G = G.copy()
    networkx.set_edge_attributes(G, p, 'weight')
    laplacian_matrix = networkx.laplacian_matrix(G)
    return np.linalg.inv(laplacian_matrix + np.identity(G.number_of_nodes()))

def computeS(g1, g2):
    s = [[0] for x in range(len(g1) + len(g2))]
    for node in g1:
        s[node][0] = 1
    for node in g2:
        s[node][0] = -1
    return s

def updateOmega(omega, e):
    c = omega[:,e[0]] - omega[:,e[1]]
    fac = 1/(1 + c[e[0], 0] - c[e[1], 0])
    return omega - (fac * np.dot(c, c.transpose()))

def spgreedy(Graph: networkx.Graph, candidateEdges, g1: List[int], g2: List[int], p, k):
    results = set()
    omega = computeOmega(Graph, p)
    s = computeS(g1, g2)
    sTranspose = np.transpose(s)
    kPerNode = {}
    for node in range(len(list(Graph.nodes()))):
        kPerNode[node] = 0
    while len(candidateEdges) > 0 and len(kPerNode) > 0:
        maxScore = -1
        chosenEdge = None
        i = 0
        q = np.dot(omega, s)
        for source in candidateEdges:
            for dest in candidateEdges[source].copy():
                if dest not in kPerNode:
                    candidateEdges[source].remove(dest)
                    continue
                score = f([source, dest], omega, q)
                i = i + 1
                if score > maxScore:
                    chosenEdge = (source, dest)
                    maxScore = score
        if chosenEdge == None:
            return results
        results.add(chosenEdge)
        omega = updateOmega(omega, chosenEdge)
        candidateEdges[chosenEdge[0]].remove(chosenEdge[1])
        if len(candidateEdges[chosenEdge[0]]) == 0:
            del candidateEdges[chosenEdge[0]]
        kPerNode[chosenEdge[0]] = kPerNode[chosenEdge[0]] + 1
        if kPerNode[chosenEdge[0]] == k:
            del kPerNode[chosenEdge[0]]
            if chosenEdge[0] in candidateEdges:
                del candidateEdges[chosenEdge[0]]
        kPerNode[chosenEdge[1]] = kPerNode[chosenEdge[1]] + 1
        if kPerNode[chosenEdge[1]] == k:
            del kPerNode[chosenEdge[1]]
            if chosenEdge[1] in candidateEdges:
                del candidateEdges[chosenEdge[1]]
    return results

def testComputeOmega():
    Graph = networkx.Graph()
    Graph.add_edges_from([(0,1),(0,2),(0,3),(1,2)])
    omega = (computeOmega(Graph, .5))
    print(omega)
    print(updateOmega(omega, [1,3]))

def testGenerateS():
    _, G, sources, nodeToGender = pickle.load(open('tiny-graphs-with-sources-3.pickle', 'rb'))
    g1, g2 = generateLPParams.getGroupsNX(G)
    g1up = [list(G.nodes()).index(n) for n in g1]
    g2up = [list(G.nodes()).index(n) for n in g2]
    print(computeS(g1up, g2up))
    print(nodeToGender)


def test_f():
    Graph = networkx.Graph()
    Graph.add_edges_from([(0,1),(0,2),(0,3),(1,2)])
    omega = (computeOmega(Graph, .5))
    s = computeS([list(Graph.nodes()).index(0), list(Graph.nodes()).index(1)], [list(Graph.nodes()).index(2),list(Graph.nodes()).index(3)])
    result = f([1, 3], omega, s, np.transpose(s))

testComputeOmega()
# testGenerateS()
# test_f()