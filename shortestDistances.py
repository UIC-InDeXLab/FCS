import networkx as nx
import math
import buildNetworkX as bnx
from collections import deque

#safe
def updatableShortestPaths(G, queues, current):
    i = 0
    updated = {}
    while i < len(queues):
        queue = queues[i]
        while not len(queue) == 0:
            node = queue.popleft()
            if node not in updated and i < current[node]:
                updated[node] = i
                for neighbor in G.neighbors(node):
                    if i + 1 < current[neighbor]:
                        if i+1 == len(queues):
                            queues.append(deque())
                        queues[i+1].append(neighbor)
        i = i + 1
    return updated


#safe
def multipleSourceShortestDistances(G, sources):
    sourceQueue = deque()
    for source in sources:
        sourceQueue.append(source)
    current = {}
    for node in G.nodes():
        current[node] = float('inf')
    updated = updatableShortestPaths(G, [sourceQueue], current)
    for node in updated:
        current[node] = updated[node]
    return current

#safe
def shortestDistancesNewEdges(G, newEdges, current):
    queues = []
    for edge in newEdges:
        while len(queues) <= current[edge[0]] + 1:
            queues.append(deque())
        queues[current[edge[0]]+1].append(edge[1])
    return updatableShortestPaths(G, queues, current)

def shortestDistancesNewNodes(G, newNodes, current):
    queues = []
    for node in newNodes:
        if newNodes[node] == float('inf'):
            continue
        while len(queues) <= newNodes[node]:
            queues.append(deque())
        queues[newNodes[node]].append(node)
    return updatableShortestPaths(G, queues, current)

#safe
def shortestDistances(G, sources):
    result = {}
    for node in G.nodes():
        result[node] = math.inf
    for source in sources:
        length = nx.single_source_shortest_path_length(G, source)
        for node in G.nodes():
            if (node in length):
                result[node] = min(result[node], length[node])
    return result

#safe
def shortestDistancesPositions(G, sources):
    result = {}
    nodes = list(G.nodes())
    for node in range(len(nodes)):
        result[node] = math.inf
    for source in sources:
        length = nx.single_source_shortest_path_length(G, source)
        for node in range(len(nodes)):
            if (nodes[node] in length):
                result[node] = min(result[node], length[nodes[node]])
    return result

#safe
def FxForMPPFair(X, sources, g1, g2, p):
    resultG1 = {}
    resultG2 = {}
    g1Dict = {}
    for node in g1:
        g1Dict[node] = True
    length = multipleSourceShortestDistances(X, sources)
    for node in length:
        if node in g1Dict:
            resultG1[node] = max(resultG1[node] if node in resultG1 else 0, p**(length[node]))
        else:
            resultG2[node] = max(resultG2[node] if node in resultG2 else 0, p**(length[node]))
    resg1 = 0
    resg2 = 0
    for x in resultG1:
        resg1 = resg1 + resultG1[x]/len(g1)
    for x in resultG2:
        resg2 = resg2 + resultG2[x]/len(g2)
    return resg1, resg2

#safe
def FxForMPPFairNewEdges(X, shortestDistances, newEdges, g1, g2, resg1, resg2, p):
    updatedLength = shortestDistancesNewEdges(X, newEdges, shortestDistances)
    g1Dict = {}
    for node in g1:
        g1Dict[node] = True
    for node in updatedLength:
        if node in g1Dict:
            resg1 = resg1 + ((p**(updatedLength[node]) - (p**(shortestDistances[node]) if node in shortestDistances else 0))/len(g1))
        else:
            resg2 = resg2 + ((p**(updatedLength[node]) - (p**(shortestDistances[node]) if node in shortestDistances else 0))/len(g2))
    return resg1, resg2

#safe
def FxForMPP(X, sources, p):
    result = {}
    for node in X.nodes():
        result[node] = 0
    for source in sources:
        length = nx.single_source_shortest_path_length(X, source)
        for node in X.nodes():
            if (node in length):
                result[node] = max(result[node], p**(length[node]))
    res = 0
    for x in result:
        res = res + result[x]
    return res

#safe
def FxForMPPNewEdges(X, newEdges, shortestDistances, res, p):
    updatedLength = shortestDistancesNewEdges(X, newEdges, shortestDistances)
    for node in updatedLength:
        res = res + p**(updatedLength[node]) - (p**(shortestDistances[node]) if node in shortestDistances else 0)
    return res

def FxForMPPFairUpdateValues(updateValues, shortestDistances, g1Dict, g2Dict, resg1, resg2, p, previousG1Size=None, previousG2Size=None):
    if previousG1Size != None and previousG2Size != None:
        resg1 = resg1 * previousG1Size / len(g1Dict)
        resg2 = resg2 * previousG2Size / len(g2Dict)
    for node in updateValues:
        if node in g1Dict:
            resg1 = resg1 + (p**(updateValues[node]))/len(g1Dict) - ((p**(shortestDistances[node]) if node in shortestDistances else 0)/len(g1Dict))
        else:
            resg2 = resg2 + (p**(updateValues[node]))/len(g2Dict) - ((p**(shortestDistances[node]) if node in shortestDistances else 0)/len(g2Dict))
    return resg1, resg2

def FxForMPPUpdateValues(updateValues, shortestDistances, res, p):
    for node in updateValues:
        res = res + p**(updateValues[node]) - (p**(shortestDistances[node]) if node in shortestDistances else 0)
    return res

#safe
def FxForRMPPNewEdges(X, newEdges, updatedEdgeNodeDistances, shortestDistances, res, p):
    updatedLength = {}
    for edge in range(len(newEdges)):
        for node in updatedEdgeNodeDistances[edge]:
            if node not in updatedLength:
                updatedLength[node] = updatedEdgeNodeDistances[edge][node]
            else:
                updatedLength[node] = min(updatedLength[node], updatedEdgeNodeDistances[edge][node])
    for node in updatedLength:
        res = res + p**(updatedLength[node]) - (p**(shortestDistances[node]) if node in shortestDistances else 0)
    return res, updatedLength

#safe
def FxForRMPPNewEdgesFair(X, newEdges, updatedEdgeNodeDistances, shortestDistances, g1, g2, resg1, resg2, p):
    updatedLength = {}
    for edge in range(len(newEdges)):
        for node in updatedEdgeNodeDistances[edge]:
            if node not in updatedLength:
                updatedLength[node] = updatedEdgeNodeDistances[edge][node]
            else:
                updatedLength[node] = min(updatedLength[node], updatedEdgeNodeDistances[edge][node])
    for node in updatedLength:
        if node in g1:
            resg1 = resg1 + (p**(updatedLength[node]) - (p**(shortestDistances[node]) if node in shortestDistances else 0))/len(g1)
        elif node in g2:
            resg2 = resg2 + (p**(updatedLength[node]) - (p**(shortestDistances[node]) if node in shortestDistances else 0))/len(g2)
    return resg1, resg2, updatedLength


def FxForRMPPUpdateEdge(updatedEdgeNodeDistance, shortestDistances, updatedLength, res, p):
    for node in updatedEdgeNodeDistance:
        if node in updatedLength:
            res = res + p**(updatedEdgeNodeDistance[node]) - p**(updatedLength[node])
        else:
            res = res + p**(updatedEdgeNodeDistance[node]) - (p**(shortestDistances[node]) if node in shortestDistances else 0)
    return res

def test():
    adjacencyLists = {}
    adjacencyLists['1'] = {'2', '3'}
    adjacencyLists['2'] = {'1', '5'}
    adjacencyLists['3'] = {'2', '4'}

    G = bnx.buildNetworkXFromAM(adjacencyLists)

    res = shortestDistancesPositions(G, ['1', '5'])

    nodes = list(G.nodes())

    if res[nodes.index('1')] != 0:
        print('tests failed, distance to 1 not 0')
    elif res[nodes.index('5')] != 0:
        print('tests failed, distance to 5 not 0')
    elif res[nodes.index('2')] != 1:
        print('tests failed, distance to 2 not 1')
    elif res[nodes.index('3')] != 1:
        print('tests failed, distance to 3 not 1')
    elif res[nodes.index('4')] != 2:
        print('tests failed, distance to 4 not 2')
    else:
        print('tests passed!')

def testBFS():
    adjacencyLists = {}
    adjacencyLists['1'] = {'2','3'}
    adjacencyLists['2'] = {'4'}
    adjacencyLists['3'] = {'2', '6'}
    adjacencyLists['4'] = {'5'}
    adjacencyLists['5'] = {'6'}
    adjacencyLists['6'] = {'1'}
    adjacencyLists['7'] = {'8'}
    adjacencyLists['8'] = {'9'}
    adjacencyLists['9'] = {'6'}


    G = bnx.buildNetworkXFromAM(adjacencyLists)

    # myQueue = Queue()
    # myQueue.put('4')
    # myQueue.put('9')
    # print(updatableShortestPaths(G, [Queue(), myQueue], {'1': 0, '2': 1, '3': 1, '4': 2, '5': 3, '6': 4, '7': 1, '8': 2, '9': 3}))
    print(multipleSourceShortestDistances(G, ['1', '8']))
    #print(shortestDistancesNewEdges(G, [('1','4'),('1','9')], {'1': 0, '2': 1, '3': 1, '4': 2, '5': 3, '6': 4, '7': 1, '8': 2, '9': 3}))

# testBFS()
