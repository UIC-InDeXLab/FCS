import shortestDistances as sd
import buildNetworkX as bnx

def getNodeNeighbors(nodes, edges):
    nodeNeighbors = {}
    for node in range(len(nodes)):
        nodeNeighbors[node] = []
        for edge in range(len(edges)):
            if edges[edge][0] == nodes[node]:
                nodeNeighbors[node].append(edge)
            if edges[edge][1] == nodes[node]:
                nodeNeighbors[node].append(edge)
    return nodeNeighbors

def getNodeNeighborsOptimized(nodes, edges):
    nodeNeighbors = {}
    for edge in range(len(edges)):
        if nodes[edges[edge][0]] not in nodeNeighbors:
            nodeNeighbors[nodes[edges[edge][0]]] = []
        nodeNeighbors[nodes[edges[edge][0]]].append(edge)
        if nodes[edges[edge][1]] not in nodeNeighbors:
            nodeNeighbors[nodes[edges[edge][1]]] = []
        nodeNeighbors[nodes[edges[edge][1]]].append(edge)
    return nodeNeighbors

def getGroups(nodeToGender, nodes):
    g1 = []
    g2 = []
    for node in range(len(nodes)):
        if nodeToGender[nodes[node]] == '1':
            g1.append(node)
        else:
            g2.append(node)
    return g1, g2

def getGroupsEnhanced(nodeToGender, nodes, numSubgraphNodes):
    g1 = []
    g2 = []
    for node in range(len(nodes)):
        if nodeToGender[nodes[node]] == '1':
            g1.append(node + numSubgraphNodes)
        else:
            g2.append(node + numSubgraphNodes)
    return g1, g2

def getGroupsNames(nodeToGender, nodes):
    g1 = []
    g2 = []
    for node in nodes:
        if nodeToGender[node] == '1':
            g1.append(node)
        else:
            g2.append(node)
    return g1, g2


def getGroupsNX(G):
    g1 = []
    g2 = []
    for node in list(G.nodes()):
        if G.nodes()[node]['gender'] == 'male':
            g1.append(node)
        else:
            g2.append(node)
    return g1, g2

def getNoteToGenderNX(G):
    nodeToGender = {}
    for node in list(G.nodes()):
        if G.nodes()[node]['gender'] == 'male':
            nodeToGender[node] = "1"
        else:
            nodeToGender[node] = "2"
    return nodeToGender

def edgeToNodeDistances(G, edges, sources):
    res = {}
    X = G.copy()
    for edge in range(len(edges)):
        X.add_edge(edges[edge][0], edges[edge][1])
        res[edge] = sd.shortestDistancesPositions(X, sources)
        X.remove_edge(edges[edge][0], edges[edge][1])
    return res

def edgeToNodeDistancesUpdated(G, edges, shortestDistances):
    res = {}
    for edge in range(len(edges)):
        if shortestDistances[edges[edge][0]] + 1 < shortestDistances[edges[edge][1]]:
            G.add_edge(edges[edge][0], edges[edge][1])
            res[edge] = sd.shortestDistancesNewEdges(G, [edges[edge]], shortestDistances)
            G.remove_edge(edges[edge][0], edges[edge][1])
        else:
            res[edge] = {}
    return res

def shouldRemoveEdge(edgeDistances, shortestDistances):
    for node in edgeDistances:
        if edgeDistances[node] < shortestDistances[node]:
            return False
    return True

def shortestDistanceOverAnEdge(edgeNodeDistances):
    res = {}
    for edge in edgeNodeDistances:
        for node in edgeNodeDistances[edge]:
            if node not in res:
                res[node] = edgeNodeDistances[edge][node]
            elif edgeNodeDistances[edge][node] < res[node]:
                res[node] = edgeNodeDistances[edge][node]
    return res

##Same as other method but duplicated for clear separation from previous methods of calculations
#safe
def updatedShortestDistanceOverAnEdge(edgeNodeDistances):
    res = {}
    for edge in edgeNodeDistances:
        for node in edgeNodeDistances[edge]:
            if node not in res:
                res[node] = edgeNodeDistances[edge][node]
            elif edgeNodeDistances[edge][node] < res[node]:
                res[node] = edgeNodeDistances[edge][node]
    return res

def nodePositionToXijPosition(shortestDistanceOverAnEdge, shortestDistances, nodes):
    res = {}
    current = 0
    for node in range(len(nodes)):
        if shortestDistanceOverAnEdge[node] < shortestDistances[node]:
            res[node] = (current, shortestDistances[node] - shortestDistanceOverAnEdge[node], shortestDistanceOverAnEdge[node])
            current = current + shortestDistances[node] - shortestDistanceOverAnEdge[node]
    return res, current

#safe
def updatedNodePositionToXijPosition(updatedShortestDistanceOverAnEdge, shortestDistances, nodes):
    res = {}
    current = 0
    for node in updatedShortestDistanceOverAnEdge:
        res[nodes.index(node)] = (current, shortestDistances[node] - updatedShortestDistanceOverAnEdge[node], updatedShortestDistanceOverAnEdge[node])
        current = current + shortestDistances[node] - updatedShortestDistanceOverAnEdge[node]
    return res, current

def updatedNodePositionToXijPositionOptimized(updatedShortestDistanceOverAnEdge, shortestDistances, nodes):
    res = {}
    current = 0
    for node in updatedShortestDistanceOverAnEdge:
        res[nodes[node]] = (current, shortestDistances[node] - updatedShortestDistanceOverAnEdge[node], updatedShortestDistanceOverAnEdge[node])
        current = current + shortestDistances[node] - updatedShortestDistanceOverAnEdge[node]
    return res, current

def test():
    adjacencyLists = {}
    adjacencyLists['1'] = {'2', '3'}
    adjacencyLists['2'] = {'1', '5'}
    adjacencyLists['3'] = {'2', '4'}

    G = bnx.buildNetworkXFromAM(adjacencyLists)

    nodes = ['1','2','3','4','5']
    edges = [('1', '4'), ('1', '5'), ('3', '5')]
    
    sources = ['1']

    result = edgeToNodeDistancesUpdated(G, edges, {'1': 0, '2': 1, '3': 1, '4': 2, '5': 2})

    print(result)

    # result2 = shortestDistanceOverAnEdge(result)

    # print(result2)

    # result3 = nodePositionToXijPosition(result2, sd.shortestDistancesPositions(G, sources), nodes)

    # print(result3)

test()