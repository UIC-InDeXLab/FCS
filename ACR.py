import networkx
import networkx as nx
import numpy as np

def getL2(graph: networkx.Graph):
    return nx.laplacian_matrix(graph).toarray()

def getv2(mtrx):
    eigenValues, eigenVectors = np.linalg.eig(mtrx)
    idx = eigenValues.argsort()
    v2 = eigenVectors[:, idx][:,1]
    return v2

def selectGreedy(edges, k):
    kPerNode = {}
    for edge in edges:
        kPerNode[edge[0]] = 0
        kPerNode[edge[1]] = 0

    curEdge = 0
    res = []
    while len(kPerNode) > 0 and curEdge < len(edges):
        if edges[curEdge][0] in kPerNode and edges[curEdge][1] in kPerNode:
            res.append(edges[curEdge])
            kPerNode[edges[curEdge][0]] = kPerNode[edges[curEdge][0]] + 1
            kPerNode[edges[curEdge][1]] = kPerNode[edges[curEdge][1]] + 1
            if kPerNode[edges[curEdge][0]] == k:
                del kPerNode[edges[curEdge][0]]
            if kPerNode[edges[curEdge][1]] == k:
                del kPerNode[edges[curEdge][1]]
        curEdge = curEdge + 1

    return res

def ACRFoF(originalGraph: nx.Graph, candidateEdgesGraph: nx.Graph, alpha):
    L2 = getL2(candidateEdgesGraph)
    v2 = getv2(L2)
    S = [x for x in range(candidateEdgesGraph.number_of_edges())]
    candidateEdges = list(candidateEdgesGraph.edges())
    candidateNodes = list(candidateEdgesGraph.nodes())
    for edge in candidateEdges:
        ScoreFoF = len(set(originalGraph.neighbors(edge[0])).intersection(set(originalGraph.neighbors(edge[1]))))
        ScoreACR = (v2[candidateNodes.index(edge[0])] - v2[candidateNodes.index(edge[1])])**2
        S[candidateEdges.index(edge)] = ScoreFoF + alpha*np.log(ScoreACR)
    return sorted([candidateEdges[x] for x in range(candidateEdgesGraph.number_of_edges())], key=lambda e: S[candidateEdges.index(e)], reverse=True)

def testACR():
    G = nx.from_edgelist([(0,1), (1,2), (2,3), (3,4), (5,6)])
    candidateEdgeGraph = nx.from_edgelist([(0,2),(1,3),(2,4),(3,5),(4,6)])
    print(selectGreedy(ACRFoF(G, candidateEdgeGraph, .5), 1))

testACR()
