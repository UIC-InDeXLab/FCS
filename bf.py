import itertools
import generateLPParams as gen
import shortestDistances as sd
import pickle
import fof

#safe
class BruteForce:
    def __init__(self, G, sources, nodeToGender, nodeNeighbors, k, candidateEdges):
        self.G = G
        self.shortestDistances = sd.multipleSourceShortestDistances(G, sources)
        self.g1names, self.g2names = gen.getGroupsNames(nodeToGender, list(G.nodes()))
        self.g1sInit, self.g2sInit = sd.FxForMPPFair(G, sources, self.g1names, self.g2names)
        self.csInit = sd.FxForMPP(G, sources)
        self.nodeNeighbors = nodeNeighbors
        self.k = k
        self.numNodes = len(list(G.nodes()))
        self.candidateEdges = candidateEdges

        self.bestDisparity = float('inf')
        self.bestContentSpread = 0
        self.bestEdgeSelection = None

        kPerNode = {}
        for node in list(self.G.nodes()):
            kPerNode[node] = self.k

        self.findBestAllEdgeCombos(0, kPerNode, [])

    def getResults(self):
        return self.bestDisparity, self.bestContentSpread, self.bestEdgeSelection

    def findBestRecursive(self, node, edgeSelection=[[]]):
        if node == self.numNodes:
            for selection in edgeSelection:
                for e in selection:
                    edge = self.candidateEdges[e]
                    self.G.add_edge(edge[0], edge[1])
                g1s, g2s = sd.FxForMPPFairNewEdges(self.G, self.shortestDistances, [self.candidateEdges[e] for e in selection], self.g1names, self.g2names, self.g1sInit, self.g2sInit)
                if g1s == 0 or g2s == 0:
                    for e in selection:
                        edge = self.candidateEdges[e]
                        self.G.remove_edge(edge[0], edge[1])
                    continue
                disparity = g1s/g2s
                if disparity < 1:
                    disparity = 1/disparity
                if disparity < self.bestDisparity:
                    print(disparity)
                    cs = sd.FxForMPPNewEdges(self.G, [self.candidateEdges[e] for e in selection], self.shortestDistances, self.csInit)
                    self.bestDisparity = disparity
                    self.bestContentSpread = cs
                    self.bestEdgeSelection = [self.candidateEdges[e] for e in selection]
                elif abs(disparity - self.bestDisparity) <= 1e-10:
                    cs = sd.FxForMPPNewEdges(self.G, [self.candidateEdges[e] for e in selection], self.shortestDistances, self.csInit)
                    if cs > self.bestContentSpread:
                        self.bestContentSpread = cs
                        self.bestEdgeSelection = [self.candidateEdges[e] for e in selection]
                for e in selection:
                    edge = self.candidateEdges[e]
                    self.G.remove_edge(edge[0], edge[1])
            return
        
        for comb in itertools.combinations(self.nodeNeighbors[node], min(self.k, len(self.nodeNeighbors[node]))):
            self.findBestRecursive(node+1, [e + list(comb) for e in edgeSelection])

    def findBestAllEdgeCombos(self, edge, kPerNode, edgesSelected):
        if edge == len(self.candidateEdges):
            for e in edgesSelected:
                self.G.add_edge(e[0], e[1])
            g1s, g2s = sd.FxForMPPFairNewEdges(self.G, self.shortestDistances, edgesSelected, self.g1names, self.g2names, self.g1sInit, self.g2sInit)
            if g1s == 0 or g2s == 0:
                for e in edgesSelected:
                    self.G.remove_edge(e[0], e[1])
                return
            disparity = g1s/g2s
            if disparity < 1:
                disparity = 1/disparity
            if disparity < self.bestDisparity:
                print(disparity)
                cs = sd.FxForMPPNewEdges(self.G, edgesSelected, self.shortestDistances, self.csInit)
                self.bestDisparity = disparity
                self.bestContentSpread = cs
                self.bestEdgeSelection = edgesSelected
            elif abs(disparity - self.bestDisparity) <= 1e-10:
                cs = sd.FxForMPPNewEdges(self.G, edgesSelected, self.shortestDistances, self.csInit)
                if cs > self.bestContentSpread:
                    self.bestContentSpread = cs
                    self.bestEdgeSelection = edgesSelected
            for e in edgesSelected:
                self.G.remove_edge(e[0], e[1])
            return

        if kPerNode[self.candidateEdges[edge][0]] > 0 and kPerNode[self.candidateEdges[edge][1]] > 0:
            kPerNode[self.candidateEdges[edge][0]] = kPerNode[self.candidateEdges[edge][0]] - 1
            kPerNode[self.candidateEdges[edge][1]] = kPerNode[self.candidateEdges[edge][1]] - 1
            self.findBestAllEdgeCombos(edge + 1, kPerNode, edgesSelected + [self.candidateEdges[edge]])
            kPerNode[self.candidateEdges[edge][0]] = kPerNode[self.candidateEdges[edge][0]] + 1
            kPerNode[self.candidateEdges[edge][1]] = kPerNode[self.candidateEdges[edge][1]] + 1
        self.findBestAllEdgeCombos(edge + 1, kPerNode, edgesSelected)



def testBruteForce():
    _, G, sources, nodeToGender = pickle.load(open('tiny-graphs-with-sources-3.pickle', 'rb'))
    shortestDistancesNames = sd.shortestDistances(G, sources)
    shortestDistances = sd.shortestDistancesPositions(G, sources)
    friendsOfFriends = fof.friendsOfFriendsNX(G)
    candidateEdges = []
    for node in friendsOfFriends:
        for newNeighbor in friendsOfFriends[node]:
            if node in shortestDistancesNames and newNeighbor in shortestDistancesNames and shortestDistancesNames[node] + 1 < shortestDistancesNames[newNeighbor]:
                candidateEdges.append((node, newNeighbor))
    nodes = list(G.nodes())
    print(candidateEdges)
    nodeNeighbors = gen.getNodeNeighbors(nodes, candidateEdges)

    myBf = BruteForce(G, sources, nodeToGender, nodeNeighbors, 2, candidateEdges)
    print(myBf.getResults())

    # print(optimalBF(G, sources, nodeToGender, candidateEdges, nodeNeighbors, 2))


#testBruteForce()
