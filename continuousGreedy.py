import buildNetworkX as bnx
import generateLPParams as gen
import random
import shortestDistances as sd
import networkx as nx

def edgeMatching(candidateEdges, w, k):
    kPerNode = {}
    res = []
    for edge in candidateEdges:
        kPerNode[edge[0]] = k
        kPerNode[edge[1]] = k
    pairing = []
    for i in range(len(candidateEdges)):
        pairing.append((candidateEdges[i], w[i], i))
    pairing.sort(reverse=True, key=lambda e: e[1])
    for pair in pairing:
        if kPerNode[pair[0][0]] > 0 and kPerNode[pair[0][1]] > 0:
            kPerNode[pair[0][0]] = kPerNode[pair[0][0]] - 1
            kPerNode[pair[0][1]] = kPerNode[pair[0][1]] - 1
            res.append(pair[2])
    return res

def contiuous_greedy_cs_optimized(candidateEdges, updatedEdgeNodeDistances, r, d, sources, k, shortestDistances,
                                  nodeNeighbors, csInit, G, p=0.5):
    y = [0 for x in range(len(candidateEdges))]
    l = 0
    #cs = sd.FxForMPP(G, sources, p)
    while l < d:
        w = [0 for x in range(len(candidateEdges))]
        for j in range(r):
            Xedges = set()
            for i in range(len(candidateEdges)):
                val = random.uniform(0,1)
                if val < y[i]:
                    Xedges.add(candidateEdges[i])
            csX, updatedLength = sd.FxForRMPPNewEdges(G, Xedges, updatedEdgeNodeDistances, shortestDistances, csInit, p)
            for m in range(len(candidateEdges)):
                Xedges.add(candidateEdges[m])
                csM = sd.FxForRMPPUpdateEdge(updatedEdgeNodeDistances[m], shortestDistances, updatedLength, csX, p)
                Xedges.remove(candidateEdges[m])
                w[m] = w[m] + (csM - csX)/r
        edges = edgeMatching(candidateEdges, w, k)
        for edge in edges:
            y[edge] = y[edge] + 1/d
        l = l + 1
    return y

def continuous_greedy_rounding(G, y, candidateEdges, k, shortestDistances, csInit, g1names, g2names, g1sInit, g2sInit,
                               p=0.5):
    nodeToEdges = {}
    for i in range(len(candidateEdges)):
        if not candidateEdges[i][0] in nodeToEdges:
            nodeToEdges[candidateEdges[i][0]] = [candidateEdges[i]]
        else:
            nodeToEdges[candidateEdges[i][0]].append(candidateEdges[i])
    
    selectedEdges = []
    for i in range(len(candidateEdges)):
        val = random.uniform(0,1)
        if val < y[i]:
            selectedEdges.append(candidateEdges[i])
    edgeSets = []
    while len(selectedEdges) != 0:
        runoffSet = []
        edgeSet = []
        for node in nodeToEdges:
            nodeEdges = [x for x in nodeToEdges[node] if x in selectedEdges]
            edgeSet.extend(nodeEdges[:k])
            runoffSet.extend(nodeEdges[k:])
        edgeSets.append(edgeSet)
        selectedEdges = runoffSet

    maxCs = 0
    maxCsDisparity = 0
    maxEdgeSet = []

    for edgeSet in edgeSets:
        cs = sd.FxForMPPNewEdges(G, edgeSet, shortestDistances, csInit, p)
        if cs > maxCs:
            maxCs = cs
            g1s, g2s = sd.FxForMPPFairNewEdges(G, shortestDistances, edgeSet, g1names, g2names, g1sInit, g2sInit, p)
            maxCsDisparity = g1s/g2s
            if maxCsDisparity < 1:
                maxCsDisparity = 1/maxCsDisparity
            maxEdgeSet = edgeSet
    
    return maxCs, maxCsDisparity, maxEdgeSet

sudo apt-get install python3-pip
def testCG():
    adjacencyLists = {}
    adjacencyLists['1'] = ['2']
    adjacencyLists['2'] = ['3', '4', '5']
    adjacencyLists['3'] = ['6', '7', '8']
    adjacencyLists['4'] = ['9','10','11']
    adjacencyLists['5'] = ['2']
    G = bnx.buildNetworkXFromAM(adjacencyLists)
    # g1, g2 = gen.getGroups(nodeToGender, list(G.nodes()))
    candidateEdges = [('1','3'),('1','4'),('1','5')]
    sources = ['1']
    nodeNeighbors = gen.getNodeNeighbors(list(G.nodes()), candidateEdges)
    shortestDistances = sd.multipleSourceShortestDistances(G, sources)
    edgeNodeDistances = gen.edgeToNodeDistancesUpdated(G, candidateEdges, shortestDistances)
    csInit = sd.FxForMPP(G, sources)
    res = contiuous_greedy_cs_optimized(candidateEdges, edgeNodeDistances, 5, 10, sources, 2, shortestDistances,
                                        nodeNeighbors, csInit, G)

    print(res)

    if res[2] >= res[0]:
        print('tests failed! wrong edge selected')
    elif res[2] >= res[1]:
        print('tests failed! wrong edge selected')
    else:
        print('tests passed!')

def testRounding():
    adjacencyLists = {}
    adjacencyLists['1'] = ['2']
    adjacencyLists['2'] = ['3', '4', '5']
    adjacencyLists['3'] = ['6', '7', '8']
    adjacencyLists['4'] = ['9','10','11']
    adjacencyLists['5'] = ['2']
    G = bnx.buildNetworkXFromAM(adjacencyLists)
    candidateEdges = [('1','3'),('1','4'),('1','5')]
    nodeToGender = {}
    for x in list(G.nodes())[:-3]:
        nodeToGender[x] = '1'
    for x in list(G.nodes())[-3:]:
        nodeToGender[x] = '2'
    g1, g2 = gen.getGroupsNames(nodeToGender, list(G.nodes()))
    y = [1,1,1]
    sources = ['1']
    shortestDistances = sd.multipleSourceShortestDistances(G, sources)
    csInit = sd.FxForMPP(G, sources)
    g1s, g2s = sd.FxForMPPFair(G, sources, g1)

    print(continuous_greedy_rounding(G, y, candidateEdges, 1, shortestDistances, csInit, g1, g2, g1s, g2s))


# testCG()
