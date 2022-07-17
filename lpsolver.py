from scipy.optimize import linprog
import numpy as np
import pickle
import buildNetworkX as bnx
import fof
import generateLPParams as gen
import shortestDistances as sd
import sys

def solveOptimized2(edgeNodeDistances, Uie, shortestDistances, nodePositionsToXijPosition, numXij, nodes, candidateEdges, lengths, kPerNode, g1, g2, p):
    #objective function
    objective = np.zeros(numXij+candidateEdges+2)

    for node in range(nodes):
        if node in nodePositionsToXijPosition:
            positionInfo = nodePositionsToXijPosition[node]
            for j in range(positionInfo[1]):
                objective[positionInfo[0]+j] = -1*(p**(j+positionInfo[2]+1) if j+positionInfo[2] < lengths-1 else p**(j+positionInfo[2]))

    base = [0 for i in range(numXij+candidateEdges+2)]
    left = None
    right = None

    for node in range(nodes):
        if node in nodePositionsToXijPosition:
            positionInfo = nodePositionsToXijPosition[node]
            for j in range(positionInfo[1]):
                tmp = base[:]
                tmp[positionInfo[0] + j] = 1
                for e in range(candidateEdges):
                    if node in edgeNodeDistances[e] and edgeNodeDistances[e][node] <= positionInfo[2]+j:
                        tmp[numXij + e] = -1
                if left is None:
                    left = np.array([tmp])
                    right = np.array(0)
                else:
                    left = np.append(left, [tmp], axis=0)
                    right = np.append(right, 0)

    for i in Uie:
        tmp = base[:]
        for e in Uie[i]:
            tmp[numXij + e] = 1
        left = np.append(left, [tmp], axis=0)
        right = np.append(right,  kPerNode[i]) 
    
    
    tmp = base[:]


    for node in range(nodes):
        if node in nodePositionsToXijPosition:
            positionInfo = nodePositionsToXijPosition[node]
            for j in range(positionInfo[1]):
                if node in g1:
                    tmp[positionInfo[0]+j] = (1*((p**(j+positionInfo[2]))-(p**(j+positionInfo[2]+1)) if j+positionInfo[2] < lengths-1 else p**(j+positionInfo[2])))/len(g1)
                if node in g2:
                    tmp[positionInfo[0]+j] = (-1*((p**(j+positionInfo[2]))-(p**(j+positionInfo[2]+1)) if j+positionInfo[2] < lengths-1 else p**(j+positionInfo[2])))/len(g2)
    
    for node in shortestDistances:
        if node in g1:
            tmp[-2] = tmp[-2] + (p**shortestDistances[node])/len(g1)
        if node in g2:
            tmp[-1] = tmp[-1] - (p**shortestDistances[node])/len(g2)
    #print([tmp])
    left_eq = np.array([tmp])
    right_eq = np.array(0)

    bnds = [(0,1) for i in range(numXij+candidateEdges+2)]

    bnds[-2] = (1,1)
    bnds[-1] = (1,1)

    res = linprog(objective, A_ub=left, b_ub=right, A_eq=left_eq, b_eq=right_eq, bounds=bnds)

    return base if res.status == 2 else res.x

def forestFireOptimized(edgeNodeDistances, Uie, shortestDistances, nodePositionsToXijPosition, numXij, nodes, candidateEdges, lengths, kPerNode, g1, g2, g1s, g2s, p):
    #objective function
    objective = np.zeros(numXij+candidateEdges+2)

    for node in nodePositionsToXijPosition:
        positionInfo = nodePositionsToXijPosition[node]
        for j in range(positionInfo[1]):
            objective[positionInfo[0]+j] = -1*((p**(j+positionInfo[2]) - p**(j+positionInfo[2]+1)) if j+positionInfo[2] < lengths-1 else p**(j+positionInfo[2]))

    base = [0 for i in range(numXij+candidateEdges+2)]
    left = None
    right = None

    for node in nodePositionsToXijPosition:
        positionInfo = nodePositionsToXijPosition[node]
        for j in range(positionInfo[1]):
            tmp = base[:]
            tmp[positionInfo[0] + j] = 1
            for e in range(candidateEdges):
                if node in edgeNodeDistances[e] and edgeNodeDistances[e][node] <= positionInfo[2]+j:
                    tmp[numXij + e] = -1
            if left is None:
                left = np.array([tmp])
                right = np.array(0)
            else:
                left = np.append(left, [tmp], axis=0)
                right = np.append(right, 0)

    for i in Uie:
        tmp = base[:]
        for e in Uie[i]:
            tmp[numXij + e] = 1
        left = np.append(left, [tmp], axis=0)
        right = np.append(right,  kPerNode[i]) 
    
    
    tmp = base[:]

    for node in nodePositionsToXijPosition:
        positionInfo = nodePositionsToXijPosition[node]
        for j in range(positionInfo[1]):
            if node in g1:
                tmp[positionInfo[0]+j] = (1*((p**(j+positionInfo[2]))-(p**(j+positionInfo[2]+1)) if j+positionInfo[2] < lengths-1 else p**(j+positionInfo[2])))/len(g1)
            if node in g2:
                tmp[positionInfo[0]+j] = (-1*((p**(j+positionInfo[2]))-(p**(j+positionInfo[2]+1)) if j+positionInfo[2] < lengths-1 else p**(j+positionInfo[2])))/len(g2)
    
    tmp[-2] = g1s
    tmp[-1] = -1*g2s
    
    np.set_printoptions(threshold=sys.maxsize)

    left_eq = np.array([tmp])
    right_eq = np.array(0)


    bnds = [(0,1) for i in range(numXij+candidateEdges+2)]

    bnds[-2] = (1,1)
    bnds[-1] = (1,1)

    res = linprog(objective, A_ub=left, b_ub=right, A_eq=left_eq, b_eq=right_eq, bounds = bnds)

    return res.x





def test():
    adjacencyLists = {}
    adjacencyLists['1'] = ['2','3','11','12']
    adjacencyLists['2'] = ['4','5','6','7','8','9','10']

    G = bnx.buildNetworkXFromAM(adjacencyLists)

    friendsOfFriends = fof.friendsOfFriendsNX(G)

    nodes = list(G.nodes())
    edges = []
    for node in friendsOfFriends:
        for newNeighbor in friendsOfFriends[node]:
            edges.append((node, newNeighbor))

    nodeLengthEdges = gen.edgeToNodeDistances(G, edges, ['1'])
    nodeNeighbors = gen.getNodeNeighbors(nodes, edges)
    shortestDistances = sd.shortestDistancesPositions(G, ['1'])
    #print(shortestDistances)

    # print(edges)
    # print(nodes)

    g1 = [nodes.index("1"), nodes.index("3"), nodes.index("7"), nodes.index("8"), nodes.index("9"), nodes.index("10")]
    g2 = [nodes.index("2"), nodes.index("4"), nodes.index("5"), nodes.index("6"), nodes.index("11"), nodes.index("12")]

    #res = solve(nodeLengthEdges, nodeNeighbors, len(nodes), len(edges), 3, 1, g1, g2)

    #print(res[:len(nodes)*3])
    #print(res[len(nodes)*3:])

    kPerNode = {}
    for x in range(12):
        kPerNode[x] = 3

    # res2 = solveOptimized(nodeLengthEdges, nodeNeighbors, shortestDistances, len(nodes), len(edges), 3, kPerNode, g1, g2)

    # print(res2[:len(nodes)*3])
    # print(res2[len(nodes)*3:])

    edgeNodeDistancesInit = gen.edgeToNodeDistances(G, edges, ['1'])
    tempEdges = edges.copy()
    for edge in tempEdges:
        if gen.shouldRemoveEdge(edgeNodeDistancesInit[tempEdges.index(edge)], shortestDistances):
            edges.remove(edge)
    edgeNodeDistances = gen.edgeToNodeDistances(G, edges, ['1'])
    nodeNeighbors = gen.getNodeNeighbors(nodes, edges)
    shortestDistanceOverAnEdge = gen.shortestDistanceOverAnEdge(edgeNodeDistances)
    nodePositionsToXijPosition, numXij = gen.nodePositionToXijPosition(shortestDistanceOverAnEdge, shortestDistances, nodes)

    print(edges)

    # res2 = solveOptimized2(nodeLengthEdges, nodeNeighbors, shortestDistances, nodePositionsToXijPosition, numXij, len(nodes), len(edges), 3, kPerNode, g1, g2)
    #print(solveGurobi(nodeLengthEdges, nodeNeighbors, shortestDistances, nodePositionsToXijPosition, numXij, len(nodes), len(edges), 3, kPerNode, g1, g2))

    # print(res2[:numXij])
    # print(res2[numXij:numXij+len(edges)])

test()


def solveOptimizedUnfair(edgeNodeDistances, Uie, shortestDistances, nodePositionsToXijPosition, numXij, nodes, candidateEdges, lengths, kPerNode, p):
    objective = np.zeros(numXij + candidateEdges + 2)

    for node in range(nodes):
        if node in nodePositionsToXijPosition:
            positionInfo = nodePositionsToXijPosition[node]
            for j in range(positionInfo[1]):
                objective[positionInfo[0] + j] = -1 * (
                    p ** (j + positionInfo[2] + 1) if j + positionInfo[2] < lengths - 1 else p ** (
                                j + positionInfo[2]))

    base = [0 for i in range(numXij + candidateEdges + 2)]
    left = None
    right = None

    for node in range(nodes):
        if node in nodePositionsToXijPosition:
            positionInfo = nodePositionsToXijPosition[node]
            for j in range(positionInfo[1]):
                tmp = base[:]
                tmp[positionInfo[0] + j] = 1
                for e in range(candidateEdges):
                    if node in edgeNodeDistances[e] and edgeNodeDistances[e][node] <= positionInfo[2] + j:
                        tmp[numXij + e] = -1
                if left is None:
                    left = np.array([tmp])
                    right = np.array(0)
                else:
                    left = np.append(left, [tmp], axis=0)
                    right = np.append(right, 0)

    for i in Uie:
        tmp = base[:]
        for e in Uie[i]:
            tmp[numXij + e] = 1
        left = np.append(left, [tmp], axis=0)
        right = np.append(right, kPerNode[i])


    bnds = [(0, 1) for i in range(numXij + candidateEdges + 2)]

    bnds[-2] = (1, 1)
    bnds[-1] = (1, 1)

    res = linprog(objective, A_ub=left, b_ub=right, bounds=bnds)

    return res.x