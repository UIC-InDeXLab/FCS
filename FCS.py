import networkx

import ACR
import fof
import buildNetworkX as bnx
import generateLPParams as gen
import lpsolver
import pickle
import time
import forestFire as ff
import spgreedy
from randomizedRounding import round
import random
import numpy as np
import time
import continuousGreedy
import shortestDistances as sd
import threading
from multiprocessing import Process, Pool
from math import floor
import cProfile
import os.path
import networkx as nx
import bf
import RemoveEdges
import IRFA
import sys

def optimizedIterativeForestFCS(graph, sources, k, g1names, g2names, initialContentSpread, nodeToGender, alpha,
                                nodesPerIteration=500, friendsOfFriends=None, p=0.5):
    #print('.')
    if friendsOfFriends == None:
        friendsOfFriends = fof.friendsOfFriendsNX(graph)
    #print('.')
    inverseFriendsOfFriends = {}
    for node in friendsOfFriends:
       for opposite in friendsOfFriends[node]:
           if opposite not in inverseFriendsOfFriends:
               inverseFriendsOfFriends[opposite] = set()
           inverseFriendsOfFriends[opposite].add(node)
    #print('.')

    nodes = list(graph.nodes())

    kPerNodeTotal = {}
    for node in nodes:
        kPerNodeTotal[node] = k

    start_time = time.time()

    shortestDistances = sd.multipleSourceShortestDistances(graph, sources[0])

    mscff = ff.MultipleSourceContinuingFF(graph, sources[0], nodesPerIteration, shortestDistances)

    edgesCalculated = set()
    finished = False

    iterationResults = []

    previousIterationEdges = []

    runningShortestDistance = {}
    maxLengthInverse = {}

    firstIteration = True

    g1Dict = {}
    g2Dict = {}
    sg1namesDict = {}
    sg2namesDict = {}
    g1sInit = None
    g2sInit = None
    csInit = None
    subgraphNodes = []
    subgraphIndices = {}

    while not finished:
        print('.')
        subgraph, finished, iterationNodes = mscff.alternatingForestFire()

        originalEdgesSet = set()

        for node in iterationNodes:
            if node in inverseFriendsOfFriends:
                for origin in inverseFriendsOfFriends[node]:
                    if origin in subgraphIndices or origin in iterationNodes:
                        originalEdgesSet.add((origin, node))
            if node in friendsOfFriends:
                for newNeighbor in friendsOfFriends[node]:
                    if newNeighbor in subgraphIndices or newNeighbor in iterationNodes:
                        originalEdgesSet.add((node, newNeighbor))

        originalEdges = list(originalEdgesSet)
        
        for edge in previousIterationEdges:
            subgraph.add_edge(edge[0], edge[1])

        g1, g2 = gen.getGroupsEnhanced(nodeToGender, iterationNodes, len(subgraphNodes))
        sg1names, sg2names = gen.getGroupsNames(nodeToGender, iterationNodes)

        currentIterationNodes = {}

        for node in iterationNodes:
            subgraphIndices[node] = len(subgraphNodes)
            subgraphNodes.append(node)
            currentIterationNodes[node] = True

        previousG1Size = len(sg1namesDict)
        previousG2Size = len(sg2namesDict)
        for node in g1:
            g1Dict[node] = True
        for node in g2:
            g2Dict[node] = True
        for node in sg1names:
            sg1namesDict[node] = True
        for node in sg2names:
            sg2namesDict[node] = True

        #print(g1)
        #print(g2)
        


        if not firstIteration:
            newNodes = {}
            for node in iterationNodes:
                shortestDistances[node] = float('inf')
            for node in iterationNodes:
                newNodes[node] = min([shortestDistances[x] + 1 for x in subgraph.predecessors(node)])
            updatedNodes = sd.shortestDistancesNewNodes(subgraph, newNodes, shortestDistances)
            g1sInit, g2sInit = sd.FxForMPPFairUpdateValues(updatedNodes, shortestDistances, sg1namesDict, sg2namesDict, g1sInit, g2sInit, p, previousG1Size=previousG1Size, previousG2Size=previousG2Size)
            csInit = sd.FxForMPPUpdateValues(updatedNodes, shortestDistances, csInit, p)
            for node in updatedNodes:
                if shortestDistances[node] in maxLengthInverse and node in maxLengthInverse[shortestDistances[node]]:
                    del maxLengthInverse[shortestDistances[node]][node]
                    if not maxLengthInverse[shortestDistances[node]]:
                        del maxLengthInverse[shortestDistances[node]]
                shortestDistances[node] = updatedNodes[node]
                if not shortestDistances[node] in maxLengthInverse:
                    maxLengthInverse[shortestDistances[node]] = {}
                maxLengthInverse[shortestDistances[node]][node] = True
                shortestDistancePositions[subgraphIndices[node]] = shortestDistances[node]

        edgeNodeDistancesInit = gen.edgeToNodeDistancesUpdated(subgraph, originalEdges, shortestDistances)
        originalEdges = [originalEdges[edge] for edge in range(len(originalEdges)) if len(edgeNodeDistancesInit[edge]) != 0]

        edgesAdded = 0

        chosenEdges = []
        

        previousEdgesAdded = -1

        while True:
            #print('hey')
            edges = originalEdges.copy()
            edges = [edge for edge in edges if edge not in chosenEdges and kPerNodeTotal[edge[0]] > 0 and kPerNodeTotal[edge[1]] > 0]
            startPreprocess = time.process_time()

            if firstIteration:
                shortestDistancePositions = {}
                for x in subgraphIndices:
                    if not shortestDistances[x] in maxLengthInverse:
                        maxLengthInverse[shortestDistances[x]] = {}
                    maxLengthInverse[shortestDistances[x]][x] = True
                    shortestDistancePositions[subgraphIndices[x]] = shortestDistances[x]
                g1sInit, g2sInit = sd.FxForMPPFair(subgraph, sources[0], sg1namesDict.keys(), sg2namesDict.keys(), p)
                csInit = sd.FxForMPP(subgraph, sources[0], p)

            elif chosenEdges:
                # print(maxLengthInverse)
                updatedNodes = sd.shortestDistancesNewEdges(subgraph, chosenEdges, shortestDistances)
                g1sInit, g2sInit = sd.FxForMPPFairUpdateValues(updatedNodes, shortestDistances, sg1namesDict, sg2namesDict, g1sInit, g2sInit, p)
                csInit = sd.FxForMPPUpdateValues(updatedNodes, shortestDistances, csInit, p)
                for node in updatedNodes:
                    # print(node)
                    # print(shortestDistances[node])
                    del maxLengthInverse[shortestDistances[node]][node]
                    if not maxLengthInverse[shortestDistances[node]]:
                        del maxLengthInverse[shortestDistances[node]]
                    shortestDistances[node] = updatedNodes[node]
                    if not shortestDistances[node] in maxLengthInverse:
                        maxLengthInverse[shortestDistances[node]] = {}
                    maxLengthInverse[shortestDistances[node]][node] = True
                    shortestDistancePositions[subgraphIndices[node]] = shortestDistances[node]

            firstIteration = False

            edgeNodeDistancesInit = gen.edgeToNodeDistancesUpdated(subgraph, edges, shortestDistances)
            edges = [edges[edge] for edge in range(len(edges)) if len(edgeNodeDistancesInit[edge]) != 0]

            # print(shortestDistances)
            edgeNodeDistances = gen.edgeToNodeDistancesUpdated(subgraph, edges, shortestDistances)
            nodeNeighbors = gen.getNodeNeighborsOptimized(subgraphIndices, edges)
            # print(nodeNeighbors)
            # print(edges)
            shortestDistanceOverAnEdge = gen.updatedShortestDistanceOverAnEdge(edgeNodeDistances)
            nodePositionsToXijPosition, numXij = gen.updatedNodePositionToXijPositionOptimized(shortestDistanceOverAnEdge, shortestDistances, subgraphIndices)
            #print(nodePositionsToXijPosition)
            #print(g1Dict)
            # print(subgraphIndices)
            # print(edge  NodeDistances)
            

            feasibleNodes = sum([1 for x in [edge[0] for edge in edges if kPerNodeTotal[edge[1]] > 0] if kPerNodeTotal[x] > 0 and len(nodeNeighbors[subgraphIndices[x]]) > 0])

            if previousEdgesAdded == edgesAdded or feasibleNodes < len(set([edge[0] for edge in originalEdges]))*alpha:
                previousIterationEdges.extend(chosenEdges)
                break

            previousEdgesAdded = edgesAdded

            startLP = time.process_time()

            kPerNode = {}
            for node in [edge[0] for edge in edges]:
                kPerNode[subgraphIndices[node]] = kPerNodeTotal[node]
            for node in [edge[1] for edge in edges]:
                kPerNode[subgraphIndices[node]] = kPerNodeTotal[node]

            edgeNodeDistancesNodePos = {}
            for edge in edgeNodeDistances:
                edgeNodeDistancesNodePos[edge] = {}
                for node in edgeNodeDistances[edge]:
                    edgeNodeDistancesNodePos[edge][subgraphIndices[node]] = edgeNodeDistances[edge][node]


            print(len(edges))
            if not nodePositionsToXijPosition:
                break

            try:
                res = lpsolver.forestFireOptimized(edgeNodeDistancesNodePos, nodeNeighbors, shortestDistancePositions, nodePositionsToXijPosition, numXij, len(subgraphNodes), len(edges), max(maxLengthInverse.keys()), kPerNode, g1Dict, g2Dict, g1sInit, g2sInit, p)
            except:
                break

            icRoundings = []


            # print(numXij)
            # print(res)

            # g1stuff = g1sInit
            # g2stuff = -1*g2sInit
            # for node in nodePositionsToXijPosition:
            #     positionInfo = nodePositionsToXijPosition[node]
            #     for j in range(positionInfo[1]):
            #         if node in g1:
            #             g1stuff = g1stuff + (-1*(.5**(j+positionInfo[2]+1) if j+positionInfo[2] < max(maxLengthInverse.keys())-1 else .5**(j+positionInfo[2]))*res[positionInfo[0]+j])/len(g1Dict)
            #         else:
            #             g2stuff = g2stuff + (-1*(.5**(j+positionInfo[2]+1) if j+positionInfo[2] < max(maxLengthInverse.keys())-1 else .5**(j+positionInfo[2]))*res[positionInfo[0]+j])/len(g2Dict)

            # print(g1stuff + g2stuff)

            icRoundings = []
            for i in range(200):
                nodeToEdge = round(edges, res[numXij:-2])
                newEdges = []
                for src in nodeToEdge:
                    for dest in nodeToEdge[src]:
                        subgraph.add_edge(src, dest)
                        newEdges.append((src, dest))
                updatedNodes = sd.shortestDistancesNewEdges(subgraph, newEdges, shortestDistances)

                # for node in updatedNodes:
                    # print(node, updatedNodes[node], shortestDistances[node])




                g1s, g2s = sd.FxForMPPFairUpdateValues(updatedNodes, shortestDistances, sg1namesDict, sg2namesDict, g1sInit, g2sInit, p)
                if g1s == 0 or g2s == 0:
                    continue
                disparity = g1s/g2s
                if disparity < 1:
                    disparity = 1/disparity


                # print(disparity)
                
                # g1s2, g2s2 = sd.FxForMPPFairNewEdges(subgraph, shortestDistances, newEdges, g1, g2, g1sInit, g2sInit)
                # if g1s2 == 0 or g2s2 == 0:
                #     continue
                # disparity2 = g1s2/g2s2
                # if disparity2 < 1:
                #     disparity2 = 1/disparity2
                # print(disparity2) 

                cs = sd.FxForMPPUpdateValues(updatedNodes, shortestDistances, csInit, p)

                icRoundings.append((None, disparity, cs, nodeToEdge))
                for src in nodeToEdge:
                    for dest in nodeToEdge[src]:
                        subgraph.remove_edge(src, dest)
            # print('evaluations')
            # print([rounding[1] for rounding in icRoundings])

            if (len(icRoundings)==0):
                return False, None

            icRoundings.sort(key=lambda e: e[1])
            disparityCutoff = 1.0 + ((icRoundings[0][1] - 1.0) * 1.2)
            icRoundings = [rounding for rounding in icRoundings if rounding[1] <= disparityCutoff]
            icRoundings.sort(key=lambda e: e[2], reverse=True)
            bestICRounding = icRoundings[0]

            for src in bestICRounding[3]:
                kPerNodeTotal[src] = max(kPerNodeTotal[src] - len(bestICRounding[3][src]), 0)
                for dest in bestICRounding[3][src]:
                    kPerNodeTotal[dest] = max(kPerNodeTotal[dest] - 1, 0)

            # print(bestICRounding[1])

            thisIterationEdges = []

            #print('hey1')
            #print(bestICRounding)
            for src in bestICRounding[3]:
                for dest in bestICRounding[3][src]:
                    subgraph.add_edge(src, dest)
                    chosenEdges.append((src, dest))
                    thisIterationEdges.append((src, dest))
                    edgesAdded = edgesAdded + 1 
            
                # g1s, g2s = sd.FxForMPPFair(subgraph, sources[0], sg1names, sg2names)
                # disparity = g1s/g2s
                # if disparity < 1:
                #     disparity = 1/disparity
                # cs = sd.FxForMPP(subgraph, sources[0])

                # print('Disparity')
                # print((disparity-1)*100)
                # print('Lift')
                # print((cs-initialContentSpread)/initialContentSpread*100)

            #print(len(chosenEdges))
            updatedNodes = sd.shortestDistancesNewEdges(subgraph, thisIterationEdges, shortestDistances)
            g1sInit, g2sInit = sd.FxForMPPFairUpdateValues(updatedNodes, shortestDistances, sg1namesDict, sg2namesDict, g1sInit, g2sInit, p)
            csInit = sd.FxForMPPUpdateValues(updatedNodes, shortestDistances, csInit, p)
            for node in updatedNodes:
                del maxLengthInverse[shortestDistances[node]][node]
                if not maxLengthInverse[shortestDistances[node]]:
                    del maxLengthInverse[shortestDistances[node]]
                shortestDistances[node] = updatedNodes[node]
                if not shortestDistances[node] in maxLengthInverse:
                    maxLengthInverse[shortestDistances[node]] = {}
                maxLengthInverse[shortestDistances[node]][node] = True
                shortestDistancePositions[subgraphIndices[node]] = shortestDistances[node]

        current_time = time.time()

        results = {}
        results['Nodes'] = len(subgraphNodes)
        results['Time'] = current_time - start_time
        results['Chosen Edges'] = chosenEdges

        #print('hey2')
        iterationResults.append(results)

    print('Done in:', iterationResults[-1]['Time'])
    return processIterationResults(iterationResults, graph, sources, g1names, g2names, initialContentSpread, 100, p)
    # print(iterationResults)



def processIterationResults(iterationResults, graph, sources, g1names, g2names, initialContentSpread,collapseSize,p=0.5):
    finalGraph = graph.copy()

    iterationPrevious = 0
    iterationNext = iterationPrevious + collapseSize

    allEdges = [edge for iteration in iterationResults for edge in iteration['Chosen Edges']]

    for edge in allEdges:
        finalGraph.add_edge(edge[0], edge[1])

    g1s, g2s = sd.FxForMPPFair(finalGraph, sources[0], g1names, g2names, p)
    disparity = g1s/g2s
    if disparity < 1:
        disparity = 1/disparity
    cs = sd.FxForMPP(finalGraph, sources[0], p)

    print('Final Dispairty')
    print((disparity-1)*100)
    print((cs-initialContentSpread)/initialContentSpread*100)

    iterations = []
    while (iterationPrevious < len(iterationResults)):
        iteration = {}
        itEdges = [edge for iteration in iterationResults[iterationPrevious:iterationNext] for edge in iteration['Chosen Edges']]
        iteration['Chosen Edges'] = itEdges
        iteration['Nodes'] = max([iter['Nodes'] for iter in iterationResults[iterationPrevious:iterationNext]])
        iteration['Time'] = max([iter['Time'] for iter in iterationResults[iterationPrevious:iterationNext]])
        iterations.append(iteration)
        iterationPrevious = iterationNext
        iterationNext = iterationNext + collapseSize

    finalGraph = graph.copy()

    for iteration in iterations:
        for edge in iteration['Chosen Edges']:
            finalGraph.add_edge(edge[0], edge[1])

        g1s, g2s = sd.FxForMPPFair(finalGraph, sources[0], g1names, g2names, p)
        disparity = g1s/g2s
        if disparity < 1:
            disparity = 1/disparity
        cs = sd.FxForMPP(finalGraph, sources[0], p)

        iteration['Disparity'] = (disparity-1)*100
        iteration['Lift'] = (cs-initialContentSpread)/initialContentSpread*100
    return iterations

#safe
#move graph.copy and fof generation to inputs?
def iterFCS(graph, sources, g1, g2, g1names, g2names, alpha, k, p, initialContentSpread, friends='fof', fairnessOff=False):
    graph = graph.copy()
    ce = fof.CommunityMembership(graph) if friends == 'community' else fof.friendsOfFriendsNX(graph)
    start_time = time.time()

    shortestDistances = sd.multipleSourceShortestDistances(graph, sources[0])
    for node in ce:
        ce[node] = [x for x in ce[node] if shortestDistances[x] != float('inf')]

    nodes = list(graph.nodes())

    kPerNode = {}
    for node in range(len(nodes)):
        kPerNode[node] = k

    originalEdges = []
    for node in ce:
        for newNeighbor in ce[node]:
            originalEdges.append((node, newNeighbor))

    edgesAdded = 0

    chosenEdges = []

    length = max(shortestDistances.values())

    shortestDistancePositions = {}
    for x in shortestDistances:
        shortestDistancePositions[nodes.index(x)] = shortestDistances[x]

    edgeNodeDistancesInit = gen.edgeToNodeDistancesUpdated(graph, originalEdges, shortestDistances)
    
    originalEdges = [originalEdges[edge] for edge in range(len(originalEdges)) if len(edgeNodeDistancesInit[edge]) != 0]
    nodeNeighbors = gen.getNodeNeighbors(nodes, originalEdges)
    
    edges = originalEdges.copy()

    feasibleNodes = sum([1 for x in nodes if kPerNode[nodes.index(x)] > 0 and len(nodeNeighbors[nodes.index(x)]) > 0])

    iterationResults = []

    while feasibleNodes > len(set([edge[0] for edge in originalEdges]))*alpha:
        print(feasibleNodes)
        shortestDistances = sd.multipleSourceShortestDistances(graph, sources[0])
        for x in shortestDistances:
            shortestDistancePositions[nodes.index(x)] = shortestDistances[x]
            
        edgeNodeDistancesInit = gen.edgeToNodeDistancesUpdated(graph, edges, shortestDistances)

        edges = [edges[edge] for edge in range(len(edges)) if len(edgeNodeDistancesInit[edge]) != 0]

        edgeNodeDistances = gen.edgeToNodeDistancesUpdated(graph, edges, shortestDistances)
        nodeNeighbors = gen.getNodeNeighbors(nodes, edges)
        shortestDistanceOverAnEdge = gen.updatedShortestDistanceOverAnEdge(edgeNodeDistances)
        nodePositionsToXijPosition, numXij = gen.updatedNodePositionToXijPosition(shortestDistanceOverAnEdge, shortestDistances, nodes)

        if numXij == 0:
            break
        
        edgeNodeDistancesNodePos = {}
        for edge in edgeNodeDistances:
            edgeNodeDistancesNodePos[edge] = {}
            for node in edgeNodeDistances[edge]:
                edgeNodeDistancesNodePos[edge][nodes.index(node)] = edgeNodeDistances[edge][node]

        if not edgeNodeDistancesNodePos:
            break

        try:
            if fairnessOff:
                res = lpsolver.solveOptimizedUnfair(edgeNodeDistancesNodePos, nodeNeighbors, shortestDistancePositions, nodePositionsToXijPosition, numXij, len(nodes), len(edges), length, kPerNode, p)
            else:
                res = lpsolver.solveOptimized2(edgeNodeDistancesNodePos, nodeNeighbors, shortestDistancePositions, nodePositionsToXijPosition, numXij, len(nodes), len(edges), length, kPerNode, g1, g2, p)
        except:
            break

        roundings = []
        
        g1sInit, g2sInit = sd.FxForMPPFair(graph, sources[0], g1names, g2names, p)
        csInit = sd.FxForMPP(graph, sources[0], p)
        workingGraph = graph.copy()

        for i in range(200):
            nodeToEdge = round(edges, res[numXij:-2])
            newEdges = []
            for src in nodeToEdge:
                for dest in nodeToEdge[src]:
                    workingGraph.add_edge(src, dest)
                    newEdges.append((src, dest))
            g1s, g2s = sd.FxForMPPFairNewEdges(workingGraph, shortestDistances, newEdges, g1names, g2names, g1sInit, g2sInit, p)
            if g1s == 0 or g2s == 0:
                continue
            disparity = g1s/g2s
            if disparity < 1:
                disparity = 1/disparity
            cs = sd.FxForMPPNewEdges(workingGraph, newEdges, shortestDistances, csInit,p)

            roundings.append((disparity, cs, nodeToEdge))
            for src in nodeToEdge:
                for dest in nodeToEdge[src]:
                    workingGraph.remove_edge(src, dest)

        if not fairnessOff:
            roundings.sort(key=lambda e: e[0])
            disparityCutoff = 1.0 + ((roundings[0][0] - 1.0) * 1.2)
            roundings = [rounding for rounding in roundings if rounding[0] <= disparityCutoff]
        roundings.sort(key=lambda e: e[1], reverse=True)
        bestRounding = roundings[0]


        if len(bestRounding[2]) == 0:
            break


        for src in bestRounding[2]:
            kPerNode[nodes.index(src)] = max(kPerNode[nodes.index(src)] - len(bestRounding[2][src]), 0)
            for dest in bestRounding[2][src]:
                kPerNode[nodes.index(dest)] = max(kPerNode[nodes.index(dest)] - 1, 0)

        rmppGraph = graph.copy()
        newEdges = []
        for src in bestRounding[2]:
            for dest in bestRounding[2][src]:
                graph.add_edge(src, dest)
                chosenEdges.append((src, dest))
                edgesAdded = edgesAdded + 1
                newEdges.append((src, dest))

        results = {}

        g1s, g2s = sd.FxForMPPFair(graph, sources[0], g1names, g2names, p)
        cs = sd.FxForMPP(graph, sources[0], p)
        disparity = g1s/g2s
        if disparity < 1:
            disparity = 1/disparity

        current_time = time.time()
        results['Lift'] = (cs-initialContentSpread)/initialContentSpread * 100
        results['Disparity'] = (disparity - 1) * 100
        results['Time'] = (current_time - start_time)
        results['Chosen Edges'] = chosenEdges

        iterationResults.append(results)

        edges = [edge for edge in edges if edge not in chosenEdges and kPerNode[nodes.index(edge[0])] > 0 and kPerNode[nodes.index(edge[1])] > 0]
        feasibleNodes = sum([1 for x in nodes if kPerNode[nodes.index(x)] > 0 and len(nodeNeighbors[nodes.index(x)]) > 0])

    #Returns false if finding a fair solution was impossbile.
    return (iterationResults, time.time() - start_time) if len(iterationResults) > 0 else (False, time.time() - start_time)


#safe
def generateLargeGraphs(G, nodeToGender, i, p=0.5):
    trimmedGraph = ff.start_a_fire(G, 1000000)
    print('.')
    g1names, g2names = gen.getGroupsNames(nodeToGender, list(trimmedGraph.nodes()))
    g1, g2 = gen.getGroups(nodeToGender, list(trimmedGraph.nodes()))

    worstDisparity = 1
    finalSources = None
    for i in range(75):
        print(i)
        sources = [random.sample(list(trimmedGraph.nodes()), 9), None]
        g1s, g2s = sd.FxForMPPFair(trimmedGraph, sources[0], g1names, g2names, p)
        disparity = g1s/g2s if g1s/g2s > 1 else g2s/g1s
        if disparity > worstDisparity:
            worstDisparity = disparity
            finalSources = sources

    pickle.dump((trimmedGraph, sources, nodeToGender), open('1000000-graph-'+str(i)+'.pickle', 'wb'))
    shortestDistances = sd.multipleSourceShortestDistances(trimmedGraph, sources[0])
    mscff100000 = ff.MultipleSourceContinuingFF(trimmedGraph, sources[0], 100000, shortestDistances, 1)
    print('.')
    subgraph100000, _, _ = mscff100000.alternatingForestFire()
    pickle.dump((subgraph100000, sources, nodeToGender), open('100000-graph'+str(i)+'.pickle', 'wb'))
    shortestDistances = sd.multipleSourceShortestDistances(subgraph100000, sources[0    ])
    mscff50000 = ff.MultipleSourceContinuingFF(subgraph100000, sources[0], 50000, shortestDistances, 1)
    print('.')
    subgraph50000, _, _ = mscff50000.alternatingForestFire()  
    pickle.dump((subgraph50000, sources, nodeToGender), open('50000-graph'+str(i)+'.pickle', 'wb'))


#safe
def generateTinyGraphs(i, p=0.5):
    G = pickle.load(open('graph_spa_500_0.pickle', 'rb'))

    while True:
        print('.')
        trimmedGraph = ff.start_a_fire(G, 40)
        nodeToGender = gen.getNoteToGenderNX(trimmedGraph)
        g1names, g2names = gen.getGroupsNames(nodeToGender, list(trimmedGraph.nodes()))
        g1, g2 = gen.getGroups(nodeToGender, list(trimmedGraph.nodes()))
        sources = [random.sample(list(trimmedGraph.nodes()), 1), None]
        g1s, g2s = sd.FxForMPPFair(trimmedGraph, sources[0], g1names, g2names, p)
        if (g1s == 0 or g2s == 0):
            continue 
        initialDisparity = g1s/g2s if g1s/g2s > 1 else g2s/g1s
        if initialDisparity > 2 or initialDisparity < 1.95:
            continue
        shortestDistancesNames = sd.shortestDistances(trimmedGraph, sources[0])
        friendsOfFriends = fof.friendsOfFriendsNX(trimmedGraph)
        candidateEdges = []
        for node in friendsOfFriends:
            for newNeighbor in friendsOfFriends[node]:
                if node in shortestDistancesNames and newNeighbor in shortestDistancesNames and shortestDistancesNames[node] + 1 < shortestDistancesNames[newNeighbor]:
                    candidateEdges.append((node, newNeighbor))
        nodeNeighbors = gen.getNodeNeighbors(list(trimmedGraph.nodes()), candidateEdges)

        myBf = bf.BruteForce(trimmedGraph, sources[0], nodeToGender, nodeNeighbors, 2, candidateEdges)
        bestDisparity, bestContentSpread, _ = myBf.getResults()
        if bestDisparity < 1.0000001:
            pickle.dump((initialDisparity, trimmedGraph, sources[0], nodeToGender), open('tiny-graphs-with-sources-'+str(i)+'.pickle', 'wb'))
            return

#safe
def OptimalityExperiment(i, p=0.5):
    result = {}
    initialDisparity, G, sources, nodeToGender = pickle.load(open('tiny-graphs-with-sources-'+str(i)+'.pickle', 'rb'))
    g1names, g2names = gen.getGroupsNames(nodeToGender, list(G.nodes()))
    g1, g2 = gen.getGroups(nodeToGender, list(G.nodes()))

    shortestDistancesNames = sd.shortestDistances(G, sources)
    shortestDistances = sd.shortestDistancesPositions(G, sources)
    friendsOfFriends = fof.friendsOfFriendsNX(G)
    candidateEdges = []

    #calculates "good" edges, potentially move into methods
    for node in friendsOfFriends:
        for newNeighbor in friendsOfFriends[node]:
            if node in shortestDistancesNames and newNeighbor in shortestDistancesNames and shortestDistancesNames[node] + 1 < shortestDistancesNames[newNeighbor]:
                candidateEdges.append((node, newNeighbor))
    nodes = list(G.nodes())
    nodeNeighbors = gen.getNodeNeighbors(nodes, candidateEdges)

    result['Initial Disparity'] = initialDisparity
    result['Initial Content Spread'] = sd.FxForMPP(G, sources, p)

    bfStarTime = time.time()
    myBf = bf.BruteForce(G, sources, nodeToGender, nodeNeighbors, 2, candidateEdges)
    bestDisparity, bestContentSpread, _ = myBf.getResults()

    optResults = {}
    optResults['Time'] = time.time() - bfStarTime
    optResults['Disparity'] = (bestDisparity-1)*100
    optResults['Content Spread'] = (bestContentSpread - result['Initial Content Spread'])/result['Initial Content Spread']*100
    result['Optimum'] = optResults

    #result = pickle.load(open('Optimality-Experiment-'+str(i)+'.pickle', 'rb'))

    result['IterFCS'] = []
    for j in range(10):
        result['IterFCS'].append(
            iterFCS(G, [sources, None], g1, g2, g1names, g2names, .05, p, result['Initial Content Spread']))
    
    pickle.dump(result, open('Optimality-Experiment-Updated-'+str(i)+'.pickle', 'wb'))

    print('Done!')

#safe
def VaryingKExperiment(k, p=0.5, friends='fof', oneThousand=False):
    if oneThousand:
        Graphs = pickle.load(open('1000-graphs-and-sources.pickle', 'rb'))
        sourcesGraph = Graphs[3]
    else:
        Graphs = pickle.load(open('synthetic-graphs-and-sources.pickle', 'rb'))
        sourcesGraph = Graphs['sources-3']

    averageInitialDisparity = 0
    graphNum = 0
    for contents in sourcesGraph:
        G, sources, nodeToGender = ((contents[0], contents[1][0], contents[2])
                                    if oneThousand else (contents[0], contents[1], gen.getNoteToGenderNX(contents[0])))
        friendsOfFriends = fof.CommunityMembership(G) if friends == 'community' else fof.friendsOfFriendsNX(G)
        g1, g2 = gen.getGroups(nodeToGender, list(G.nodes()))
        g1names, g2names = gen.getGroupsNames(nodeToGender, list(G.nodes()))

        nodes = list(G.nodes())
        edges = []
        for node in friendsOfFriends:
            for newNeighbor in friendsOfFriends[node]:
                edges.append((node, newNeighbor))

        shortestDistances = sd.multipleSourceShortestDistances(G, sources)

        edgeNodeDistancesInit = gen.edgeToNodeDistancesUpdated(G, edges, shortestDistances)
        edges = [edges[edge] for edge in range(len(edges)) if len(edgeNodeDistancesInit[edge]) != 0]
        edgeNodeDistances = gen.edgeToNodeDistancesUpdated(G, edges, shortestDistances)
        nodeNeighbors = gen.getNodeNeighbors(nodes, edges)

        g1s, g2s = sd.FxForMPPFair(G, sources, g1names, g2names, p)

        initialDisparity = ((g1s/g2s if g1s/g2s > 1 else g2s/g1s) - 1) * 100
        initialContentSpread = sd.FxForMPP(G, sources, p)
        averageInitialDisparity = averageInitialDisparity + (initialDisparity/len(sourcesGraph))

        iterFCSResults = iterFCS(G, [sources, None], g1, g2, g1names, g2names, .05, k, p, initialContentSpread,
                                 friends=friends)
        
        results = {}

        results['Initial Disparity'] = initialDisparity

        if iterFCSResults[0] != False:
            results['Lift'] = iterFCSResults[0][-1]['Lift']
            results['Final Disparity'] = iterFCSResults[0][-1]['Disparity']
            results['Time'] = iterFCSResults[0][-1]['Time']
            results['Chosen Edges'] = len(iterFCSResults[0][-1]['Chosen Edges'])
        else:
            results['Lift'] = 0
            results['FInal Disparity'] = results['Initial Disaprity']
            results['Time'] = results[1]
            results['Chosen Edges'] = 0

        start_time = time.time()
        y = continuousGreedy.contiuous_greedy_cs_optimized(edges, edgeNodeDistances, 10, 500, sources, k,
                                                           shortestDistances, nodeNeighbors, initialContentSpread, G, p)
        maxCs, maxCsDisparity, maxEdgeSet = continuousGreedy.continuous_greedy_rounding(G, y, edges, k,
                                                                                        shortestDistances,
                                                                                        initialContentSpread, g1names,
                                                                                        g2names, g1s, g2s, p)
        end_time = time.time()
        
        results['CG Lift'] = (maxCs-initialContentSpread)/initialContentSpread*100
        results['CG Final Disparity'] = (maxCsDisparity-1)*100
        results['CG Time'] = (end_time-start_time)
        results['Max Edge Set'] = len(maxEdgeSet)   


        dag = G.copy()
        dag.remove_edges_from(RemoveEdges.dfs_remove_back_edges(G))
        topologicalSort = list(networkx.topological_sort(dag))
        irfaFoF = {}
        for source in friendsOfFriends:
            irfaFoF[source] = [dest for dest in friendsOfFriends[source] if topologicalSort.index(source) < topologicalSort.index(dest)]
        start_time = time.time()
        edges = IRFA.IRFA(dag, sources, k, irfaFoF, p)
        end_time = time.time()
        RG = G.copy()
        RG.add_edges_from(edges)
        g1s, g2s = sd.FxForMPPFair(RG, sources, g1names, g2names, p)
        results['IRFA Final Disparity'] = (g1s / g2s - 1 if g1s / g2s > 1 else g2s / g1s - 1) * 100
        results['IRFA Lift'] = (sd.FxForMPP(RG, sources,p) - initialContentSpread)/initialContentSpread * 100,
        results['IRFA Time'] = end_time - start_time

        ud = G.to_undirected()
        nds = list(ud.nodes())
        indexedEdges = {}
        for source in friendsOfFriends:
            indexedEdges[nds.index(source)] = set()
            for dest in friendsOfFriends[source]:
                indexedEdges[nds.index(source)].add(nds.index(dest))
        start_time = time.time()
        edges = spgreedy.spgreedy(ud, indexedEdges, g1, g2, p, k)
        end_time = time.time()
        RG = G.copy()
        RG.add_edges_from([(nds[edge[0]], nds[edge[1]]) for edge in edges])
        g1s, g2s = sd.FxForMPPFair(RG, sources, g1names, g2names, p)
        results['SPGREEDY Final Disparity'] = (g1s / g2s - 1 if g1s / g2s > 1 else g2s / g1s - 1) * 100
        results['SPGREEDY Lift'] = (sd.FxForMPP(RG, sources, p) - initialContentSpread)/initialContentSpread * 100
        results['SPGREEDY Time'] = end_time - start_time

        originalGraph = G.to_undirected()
        candidateEdgesGraph =  nx.create_empty_copy(originalGraph)
        for source in friendsOfFriends:
            for dest in friendsOfFriends[source]:
                candidateEdgesGraph.add_edge(source, dest)
        start_time = time.time()
        edges = ACR.selectGreedy(ACR.ACRFoF(originalGraph, candidateEdgesGraph, .5), k)
        end_time = time.time()
        RG = G.copy()
        RG.add_edges_from(edges)
        g1s, g2s = sd.FxForMPPFair(RG, sources, g1names, g2names, p)
        results['ACR Final Disparity'] = (g1s / g2s - 1 if g1s / g2s > 1 else g2s / g1s - 1) * 100
        results['ACR Lift'] = (sd.FxForMPP(RG, sources, p) - initialContentSpread)/initialContentSpread * 100
        results['ACR Time'] = end_time - start_time

        pickle.dump(results, open('varying-'+('1000-' if oneThousand else '')+'k-'+str(k)+'-trial-'+str(graphNum)+'-'+friends+'.pickle', 'wb'))

        graphNum = graphNum + 1

#safe
def VaryingSourcesExperient(sources, p=0.5, friends='fof', oneThousand=False):
    if oneThousand:
        Graphs = pickle.load(open('1000-graphs-and-sources.pickle', 'rb'))
        sourcesGraph = Graphs[sources]
    else:
        Graphs = pickle.load(open('synthetic-graphs-and-sources.pickle', 'rb'))
        sourcesGraph = Graphs['sources-'+str(sources)]


    graphNum = 0
    for contents in sourcesGraph:
        G, sources, nodeToGender = ((contents[0], contents[1][0], contents[2])
                                    if oneThousand else (contents[0], contents[1], gen.getNoteToGenderNX(contents[0])))
        friendsOfFriends = fof.CommunityMembership(G) if friends == 'community' else fof.friendsOfFriendsNX(G)

        g1, g2 = gen.getGroups(nodeToGender, list(G.nodes()))
        g1names, g2names = gen.getGroupsNames(nodeToGender, list(G.nodes()))

        nodes = list(G.nodes())
        edges = []
        for node in friendsOfFriends:
            for newNeighbor in friendsOfFriends[node]:
                edges.append((node, newNeighbor))

        shortestDistances = sd.multipleSourceShortestDistances(G, sources)

        edgeNodeDistancesInit = gen.edgeToNodeDistancesUpdated(G, edges, shortestDistances)
        edges = [edges[edge] for edge in range(len(edges)) if len(edgeNodeDistancesInit[edge]) != 0]
        edgeNodeDistances = gen.edgeToNodeDistancesUpdated(G, edges, shortestDistances)
        nodeNeighbors = gen.getNodeNeighbors(nodes, edges)

        g1s, g2s = sd.FxForMPPFair(G, sources, g1names, g2names, p)

        initialDisparity = ((g1s/g2s if g1s/g2s > 1 else g2s/g1s) - 1) * 100
        initialContentSpread = sd.FxForMPP(G, sources, p)

        iterFCSResults = iterFCS(G, [sources, None], g1, g2, g1names, g2names, .05, 3, p, initialContentSpread,
                                 friends=friends)
        
        results = {}

        results['Initial Disparity'] = initialDisparity

        if iterFCSResults[0] != False:
            results['Lift'] = iterFCSResults[0][-1]['Lift']
            results['Final Disparity'] = iterFCSResults[0][-1]['Disparity']
            results['Time'] = iterFCSResults[0][-1]['Time']
            results['Chosen Edges'] = len(iterFCSResults[0][-1]['Chosen Edges'])
        else:
            results['Lift'] = 0
            results['Final Disparity'] = results['Initial Disparity']
            results['Time'] = results[1]
            results['Chosen Edges'] = 0

        unfairResults = iterFCS(G, [sources, None], g1, g2, g1names, g2names, .05, 3, initialContentSpread, fairnessOff=True)

        results['UF-Lift'] = unfairResults[-1]['Lift']
        results['UF-Final Disparity'] = unfairResults[-1]['Disparity']
        results['UF-sTime'] = unfairResults[-1]['Time']
        #
        #
        # start_time = time.time()
        # y = continuousGreedy.contiuous_greedy_cs_optimized(edges, edgeNodeDistances, 10, 500, sources, 3,
        #                                                    shortestDistances, nodeNeighbors, initialContentSpread, G)
        # maxCs, maxCsDisparity, maxEdgeSet = continuousGreedy.continuous_greedy_rounding(G, y, edges, 3,
        #                                                                                 shortestDistances,
        #                                                                                 initialContentSpread, g1names,
        #                                                                                 g2names, g1s, g2s)
        # end_time = time.time()
        #
        # results['CG Lift'] = (maxCs-initialContentSpread)/initialContentSpread*100
        # results['CG Final Disparity'] = (maxCsDisparity-1)*100
        # results['CG Time'] = (end_time-start_time)
        #
        # dag = G.copy()
        # dag.remove_edges_from(RemoveEdges.dfs_remove_back_edges(G))
        # topologicalSort = list(networkx.topological_sort(dag))
        # irfaFoF = {}
        # for source in friendsOfFriends:
        #     irfaFoF[source] = [dest for dest in friendsOfFriends[source] if topologicalSort.index(source) < topologicalSort.index(dest)]
        # start_time = time.time()
        # edges = IRFA.IRFA(dag, sources, 3, irfaFoF, p)
        # end_time = time.time()
        # RG = G.copy()
        # RG.add_edges_from(edges)
        # g1s, g2s = sd.FxForMPPFair(RG, sources, g1names, g2names, p)
        # results['IRFA Final Disparity'] = (g1s / g2s - 1 if g1s / g2s > 1 else g2s / g1s - 1) * 100
        # results['IRFA Lift'] = (sd.FxForMPP(RG, sources,p) - initialContentSpread)/initialContentSpread * 100,
        # results['IRFA Time'] = end_time - start_time
        #
        # ud = G.to_undirected()
        # nds = list(ud.nodes())
        # indexedEdges = {}
        # for source in friendsOfFriends:
        #     indexedEdges[nds.index(source)] = set()
        #     for dest in friendsOfFriends[source]:
        #         indexedEdges[nds.index(source)].add(nds.index(dest))
        # start_time = time.time()
        # edges = spgreedy.spgreedy(ud, indexedEdges, g1, g2, p, 3)
        # end_time = time.time()
        # RG = G.copy()
        # RG.add_edges_from([(nds[edge[0]], nds[edge[1]]) for edge in edges])
        # g1s, g2s = sd.FxForMPPFair(RG, sources, g1names, g2names, p)
        # results['SPGREEDY Final Disparity'] = (g1s / g2s - 1 if g1s / g2s > 1 else g2s / g1s - 1) * 100
        # results['SPGREEDY Lift'] = (sd.FxForMPP(RG, sources, p) - initialContentSpread)/initialContentSpread * 100
        # results['SPGREEDY Time'] = end_time - start_time
        #
        # originalGraph = G.to_undirected()
        # candidateEdgesGraph =  nx.create_empty_copy(originalGraph)
        # for source in friendsOfFriends:
        #     for dest in friendsOfFriends[source]:
        #         candidateEdgesGraph.add_edge(source, dest)
        # start_time = time.time()
        # edges = ACR.selectGreedy(ACR.ACRFoF(originalGraph, candidateEdgesGraph, .5), 3)
        # end_time = time.time()
        # RG = G.copy()
        # RG.add_edges_from(edges)
        # g1s, g2s = sd.FxForMPPFair(RG, sources, g1names, g2names, p)
        # results['ACR Final Disparity'] = (g1s / g2s - 1 if g1s / g2s > 1 else g2s / g1s - 1) * 100
        # results['ACR Lift'] = (sd.FxForMPP(RG, sources, p) - initialContentSpread)/initialContentSpread * 100
        # results['ACR Time'] = end_time - start_time

        # pickle.dump(results, open('varying-'+('1000-' if oneThousand else '')+'sources-'+str(len(sources))+'-trial-'+str(graphNum)+'-'+friends+'.pickle', 'wb'))
        pickle.dump(results, open('varying-fairness-sources-'+str(len(sources))+'-trial-'+str(graphNum)+'-'+friends+'.pickle', 'wb'))


        graphNum = graphNum + 1


def VaryingPExperient(p, friends='fof', oneThousand=False):
    if oneThousand:
        Graphs = pickle.load(open('1000-graphs-and-sources.pickle', 'rb'))
        sourcesGraph = Graphs[3]
    else:
        Graphs = pickle.load(open('synthetic-graphs-and-sources.pickle', 'rb'))
        sourcesGraph = Graphs['sources-3']

    graphNum = 0
    for contents in sourcesGraph:
        G, sources, nodeToGender = ((contents[0], contents[1][0], contents[2])
                                    if oneThousand else (contents[0], contents[1], gen.getNoteToGenderNX(contents[0])))

        friendsOfFriends = fof.CommunityMembership(G) if friends == 'community' else fof.friendsOfFriendsNX(G)

        g1, g2 = gen.getGroups(nodeToGender, list(G.nodes()))
        g1names, g2names = gen.getGroupsNames(nodeToGender, list(G.nodes()))

        nodes = list(G.nodes())
        edges = []
        for node in friendsOfFriends:
            for newNeighbor in friendsOfFriends[node]:
                edges.append((node, newNeighbor))

        shortestDistances = sd.multipleSourceShortestDistances(G, sources)

        edgeNodeDistancesInit = gen.edgeToNodeDistancesUpdated(G, edges, shortestDistances)
        edges = [edges[edge] for edge in range(len(edges)) if len(edgeNodeDistancesInit[edge]) != 0]
        edgeNodeDistances = gen.edgeToNodeDistancesUpdated(G, edges, shortestDistances)
        nodeNeighbors = gen.getNodeNeighbors(nodes, edges)

        g1s, g2s = sd.FxForMPPFair(G, sources, g1names, g2names, p)

        initialDisparity = ((g1s / g2s if g1s / g2s > 1 else g2s / g1s) - 1) * 100
        initialContentSpread = sd.FxForMPP(G, sources, p)

        iterFCSResults = iterFCS(G, [sources, None], g1, g2, g1names, g2names, .05, 3, p, initialContentSpread)

        results = {}

        results['Initial Disparity'] = initialDisparity

        if iterFCSResults[0] != False:
            results['Lift'] = iterFCSResults[0][-1]['Lift']
            results['Final Disparity'] = iterFCSResults[0][-1]['Disparity']
            results['Time'] = iterFCSResults[0][-1]['Time']
            results['Chosen Edges'] = len(iterFCSResults[0][-1]['Chosen Edges'])
        else:
            results['Lift'] = 0
            results['Final Disparity'] = results['Initial Disparity']
            results['Time'] = results[1]
            results['Chosen Edges'] = 0

        start_time = time.time()
        y = continuousGreedy.contiuous_greedy_cs_optimized(edges, edgeNodeDistances, 10, 500, sources, 3,
                                                           shortestDistances, nodeNeighbors, initialContentSpread, G, p)
        maxCs, maxCsDisparity, maxEdgeSet = continuousGreedy.continuous_greedy_rounding(G, y, edges, 3,
                                                                                        shortestDistances,
                                                                                        initialContentSpread, g1names,
                                                                                        g2names, g1s, g2s, p)
        end_time = time.time()

        results['CG Lift'] = (maxCs - initialContentSpread) / initialContentSpread * 100
        results['CG Final Disparity'] = (maxCsDisparity - 1) * 100
        results['CG Time'] = (end_time - start_time)

        dag = G.copy()
        dag.remove_edges_from(RemoveEdges.dfs_remove_back_edges(G))
        topologicalSort = list(networkx.topological_sort(dag))
        irfaFoF = {}
        for source in friendsOfFriends:
            irfaFoF[source] = [dest for dest in friendsOfFriends[source] if
                               topologicalSort.index(source) < topologicalSort.index(dest)]
        start_time = time.time()
        edges = IRFA.IRFA(dag, sources, 3, irfaFoF, p)
        end_time = time.time()
        RG = G.copy()
        RG.add_edges_from(edges)
        g1s, g2s = sd.FxForMPPFair(RG, sources, g1names, g2names, p)
        results['IRFA Final Disparity'] = (g1s / g2s - 1 if g1s / g2s > 1 else g2s / g1s - 1) * 100
        results['IRFA Lift'] = (sd.FxForMPP(RG, sources, p) - initialContentSpread) / initialContentSpread * 100,
        results['IRFA Time'] = end_time - start_time

        ud = G.to_undirected()
        nds = list(ud.nodes())
        indexedEdges = {}
        for source in friendsOfFriends:
            indexedEdges[nds.index(source)] = set()
            for dest in friendsOfFriends[source]:
                indexedEdges[nds.index(source)].add(nds.index(dest))
        start_time = time.time()
        edges = spgreedy.spgreedy(ud, indexedEdges, g1, g2, p, 3)
        end_time = time.time()
        RG = G.copy()
        RG.add_edges_from([(nds[edge[0]], nds[edge[1]]) for edge in edges])
        g1s, g2s = sd.FxForMPPFair(RG, sources, g1names, g2names, p)
        results['SPGREEDY Final Disparity'] = (g1s / g2s - 1 if g1s / g2s > 1 else g2s / g1s - 1) * 100
        results['SPGREEDY Lift'] = (sd.FxForMPP(RG, sources, p) - initialContentSpread) / initialContentSpread * 100
        results['SPGREEDY Time'] = end_time - start_time

        originalGraph = G.to_undirected()
        candidateEdgesGraph = nx.create_empty_copy(originalGraph)
        for source in friendsOfFriends:
            for dest in friendsOfFriends[source]:
                candidateEdgesGraph.add_edge(source, dest)
        start_time = time.time()
        edges = ACR.selectGreedy(ACR.ACRFoF(originalGraph, candidateEdgesGraph, .5), 3)
        end_time = time.time()
        RG = G.copy()
        RG.add_edges_from(edges)
        g1s, g2s = sd.FxForMPPFair(RG, sources, g1names, g2names, p)
        results['ACR Final Disparity'] = (g1s / g2s - 1 if g1s / g2s > 1 else g2s / g1s - 1) * 100
        results['ACR Lift'] = (sd.FxForMPP(RG, sources, p) - initialContentSpread) / initialContentSpread * 100
        results['ACR Time'] = end_time - start_time

        pickle.dump(results,
                    open('varying-'+('1000-' if oneThousand else '')+'p-' + str(p) + '-trial-' + str(graphNum) + '-' + friends + '.pickle',
                         'wb'))

        graphNum = graphNum + 1


#safe
def MediumGraphsExperiment(algorithm, sizePos, trial, friends, p):
    if (sizePos < 4):
        G, sources = pickle.load(open('500-to-10000-graphs.pickle', 'rb'))[sizePos][trial]
        _, _ , nodeToGender = pickle.load(open('100000-graph74.pickle', 'rb'))
    elif (sizePos == 4):
        G, sources, _ = pickle.load(open('500-to-10000-graphs.pickle', 'rb'))[sizePos][trial]
        _, _, nodeToGender = pickle.load(open('100000-graph74.pickle', 'rb'))
    else:
        G, sources, nodeToGender = pickle.load(open('20000-graph-and-sources', 'rb'))

    g1names, g2names = gen.getGroupsNames(nodeToGender, list(G.nodes()))
    g1, g2 = gen.getGroups(nodeToGender, list(G.nodes()))
    results = {}

    sources = sources[0]

    friendsOfFriends = fof.CommunityMembership(G) if friends == 'community' else fof.friendsOfFriendsNX(G)

    nodes = list(G.nodes())
    edges = []
    for node in friendsOfFriends:
        for newNeighbor in friendsOfFriends[node]:
            edges.append((node, newNeighbor))

    shortestDistances = sd.multipleSourceShortestDistances(G, sources)

    edgeNodeDistancesInit = gen.edgeToNodeDistancesUpdated(G, edges, shortestDistances)
    edges = [edges[edge] for edge in range(len(edges)) if len(edgeNodeDistancesInit[edge]) != 0]
    edgeNodeDistances = gen.edgeToNodeDistancesUpdated(G, edges, shortestDistances)
    nodeNeighbors = gen.getNodeNeighbors(nodes, edges)

    g1s, g2s = sd.FxForMPPFair(G, sources, g1names, g2names, p)
    results['Initial Disparity'] = g1s/g2s if g1s/g2s > 1 else g2s/g1s
    results['Initial Content Spread'] = sd.FxForMPP(G, sources, p)
    if algorithm == 'IterFCS':
        results['IterFCS'] = iterFCS(G, [sources, None], g1, g2, g1names, g2names, .05, 3, p, results['Initial Content Spread'])
    elif algorithm == 'ForestFire':
        results['ForestFire'] = optimizedIterativeForestFCS(G, [sources, None], 3, g1names, g2names,
                                                            results['Initial Content Spread'], nodeToGender, .05,
                                                            nodesPerIteration=500)
    elif algorithm == 'CG':
        start_time = time.time()
        y = continuousGreedy.contiuous_greedy_cs_optimized(edges, edgeNodeDistances, 10, 500, sources, 3,
                                                           shortestDistances, nodeNeighbors,
                                                           results['Initial Content Spread'], G)
        maxCs, maxCsDisparity, maxEdgeSet = continuousGreedy.continuous_greedy_rounding(G, y, edges, 3,
                                                                                        shortestDistances, results[
                                                                                            'Initial Content Spread'],
                                                                                        g1names, g2names, g1s, g2s)
        end_time = time.time()
        cgResult = {}
        cgResult['CS'] = (maxCs-results['Initial Content Spread'])/results['Initial Content Spread']*100
        cgResult['Disparity'] = (maxCsDisparity-1)*100
        cgResult['Time'] = end_time-start_time
        results['CG'] = cgResult
    elif algorithm == 'IRFA':
        dag = G.copy()
        dag.remove_edges_from(RemoveEdges.remove_cycle_edges_by_mfas(G.copy()))
        topologicalSort = list(networkx.topological_sort(dag))
        for source in friendsOfFriends:
            friendsOfFriends[source] = [dest for dest in friendsOfFriends[source] if topologicalSort.index(source) < topologicalSort.index(dest)]
        start_time = time.time()
        edges = IRFA.IRFA(dag, sources, 3, friendsOfFriends, .5)
        end_time = time.time()
        G.add_edges_from(edges)
        g1s, g2s = sd.FxForMPPFair(G, sources, g1names, g2names, p)
        irfaResults = {'Disparity': g1s / g2s if g1s / g2s > 1 else g2s / g1s,
            'Content Spread': sd.FxForMPP(G, sources,p),
            'Time': end_time - start_time}
        results['IRFA'] = irfaResults
    elif algorithm == 'SPGREEDY':
        ud = G.to_undirected()
        nds = list(ud.nodes())
        indexedEdges = {}
        for source in friendsOfFriends:
            indexedEdges[nds.index(source)] = set()
            for dest in friendsOfFriends[source]:
                indexedEdges[nds.index(source)].add(nds.index(dest))
        start_time = time.time()
        edges = spgreedy.spgreedy(ud, indexedEdges, g1, g2, .5, 3)
        end_time = time.time()
        G.add_edges_from([(nds[edge[0]], nds[edge[1]]) for edge in edges])
        g1s, g2s = sd.FxForMPPFair(G, sources, g1names, g2names, p)
        spgreedyResults = {'Disparity': g1s / g2s if g1s / g2s > 1 else g2s / g1s,
                       'Content Spread': sd.FxForMPP(G, sources, p), 'Time': end_time - start_time,
                        'Edges': edges}
        results['SPGREEDY'] = spgreedyResults
    elif algorithm == 'ACR':
        print('a')
        originalGraph = G.to_undirected()
        print('b')
        candidateEdgesGraph =  nx.create_empty_copy(originalGraph)
        for source in friendsOfFriends:
            for dest in friendsOfFriends[source]:
                candidateEdgesGraph.add_edge(source, dest)
        print('c')
        start_time = time.time()
        edges = ACR.selectGreedy(ACR.ACRFoF(originalGraph, candidateEdgesGraph, .5), 3)
        end_time = time.time()
        RG = G.copy()
        RG.add_edges_from(edges)
        g1s, g2s = sd.FxForMPPFair(RG, sources, g1names, g2names, p)
        acrResults = {'Disparity': g1s / g2s if g1s / g2s > 1 else g2s / g1s,
                       'Content Spread': sd.FxForMPP(RG, sources, p), 'Time': end_time - start_time,
                        'Edges': edges}
        results['ACR'] = acrResults


        
    pickle.dump(results, open('medium-graph-'+algorithm+'-size-'+str(sizePos)+'-trial-'+str(trial)+'.pickle', 'wb'))

#safe
def ForestFireOptions(trial, k, nodesPerIteration, nodeToGender, p=0.5):
    G, sources, _ = pickle.load(open('500-to-10000-graphs.pickle', 'rb'))[4][trial]
    g1names, g2names = gen.getGroupsNames(nodeToGender, list(G.nodes()))
    g1, g2 = gen.getGroups(nodeToGender, list(G.nodes()))
    results = {}

    sources = sources[0]

    friendsOfFriends = fof.friendsOfFriendsNX(G)

    nodes = list(G.nodes())

    graphIndices = {}

    for node in range(len(nodes)):
        graphIndices[nodes[node]] = node

    edges = []
    for node in friendsOfFriends:
        for newNeighbor in friendsOfFriends[node]:
            edges.append((node, newNeighbor))

    shortestDistances = sd.multipleSourceShortestDistances(G, sources)

    edgeNodeDistancesInit = gen.edgeToNodeDistancesUpdated(G, edges, shortestDistances)
    nodeNeighbors = gen.getNodeNeighborsOptimized(graphIndices, edges)
    edges = [edges[edge] for edge in range(len(edges)) if len(edgeNodeDistancesInit[edge]) != 0]
    edgeNodeDistances = gen.edgeToNodeDistancesUpdated(G, edges, shortestDistances)
    nodeNeighbors = gen.getNodeNeighborsOptimized(graphIndices, edges)

    g1s, g2s = sd.FxForMPPFair(G, sources, g1names, g2names, p)

    results['Initial Disparity'] = g1s/g2s if g1s/g2s > 1 else g2s/g1s
    results['Initial Content Spread'] = sd.FxForMPP(G, sources, p)
    results['ForestFire'] = optimizedIterativeForestFCS(G, [sources, None], k, g1names, g2names,
                                                        results['Initial Content Spread'], nodeToGender, .05,
                                                        nodesPerIteration=nodesPerIteration)
        
    pickle.dump(results, open('forest-fire9-k'+str(k)+'-npi-'+str(nodesPerIteration)+'-trial-'+str(trial)+'.pickle', 'wb'))

#safe
def ForestFireScale(scale, p=0.5):
    trimmedGraph, sources, nodeToGender = pickle.load(open(str(scale)+'-graph.pickle', 'rb'))
    g1names, g2names = gen.getGroupsNames(nodeToGender, list(trimmedGraph.nodes()))
    g1, g2 = gen.getGroups(nodeToGender, list(trimmedGraph.nodes()))
    results = {}

    sources = sources[0]

    g1s, g2s = sd.FxForMPPFair(trimmedGraph, sources, g1names, g2names, p)

    friendsOfFriends = fof.friendsOfFriendsNX(trimmedGraph)
    edges = []
    for node in friendsOfFriends:
        for newNeighbor in friendsOfFriends[node]:
            edges.append((node, newNeighbor))

    shortestDistances = sd.multipleSourceShortestDistances(trimmedGraph, sources)

    edgeNodeDistancesInit = gen.edgeToNodeDistancesUpdated(trimmedGraph, edges, shortestDistances)
    # edges = [edges[edge] for edge in range(len(edges)) if len(edgeNodeDistancesInit[edge]) != 0]
    #edgeNodeDistances = gen.edgeToNodeDistancesUpdated(trimmedGraph, edges, shortestDistances)
    #nodeNeighbors = gen.getNodeNeighbors(list(trimmedGraph.nodes()), edges)



    results['Initial Disparity'] = g1s/g2s if g1s/g2s > 1 else g2s/g1s
    results['Initial Content Spread'] = sd.FxForMPP(trimmedGraph, sources, p)


    # results['ForestFire'] = optimizedIterativeForestFCS(trimmedGraph, [sources, None], 5, g1names, g2names,
    #                                                     results['Initial Content Spread'], nodeToGender, .05,
    #                                                     nodesPerIteration=800)

    # start_time = time.time()
    # y = continuousGreedy.contiuous_greedy_cs_optimized(edges, edgeNodeDistances, 10, 500, sources, 5,
    #                                                    shortestDistances, nodeNeighbors,
    #                                                    results['Initial Content Spread'], trimmedGraph, p)
    # maxCs, maxCsDisparity, maxEdgeSet = continuousGreedy.continuous_greedy_rounding(trimmedGraph, y, edges, 5,
    #                                                                                 shortestDistances,
    #                                                                                 results['Initial Content Spread'],
    #                                                                                 g1names, g2names, g1s, g2s, p)
    # end_time = time.time()
    #
    # results['CG Lift'] = (maxCs - results['Initial Content Spread']) / results['Initial Content Spread'] * 100
    # results['CG Final Disparity'] = (maxCsDisparity - 1) * 100
    # results['CG Time'] = (end_time - start_time)

    dag = trimmedGraph.copy()
    dag.remove_edges_from(RemoveEdges.remove_cycle_edges_by_mfas(trimmedGraph))
    topologicalSort = list(networkx.topological_sort(dag))
    for source in friendsOfFriends:
        friendsOfFriends[source] = [dest for dest in friendsOfFriends[source] if
                                    topologicalSort.index(source) < topologicalSort.index(dest)]
    start_time = time.time()
    print('.')
    edges = IRFA.IRFA(dag, sources, 3, friendsOfFriends, .5)
    end_time = time.time()
    trimmedGraph.add_edges_from(edges)
    g1s, g2s = sd.FxForMPPFair(trimmedGraph, sources, g1names, g2names, p)
    irfaResults = {'Disparity': ((g1s / g2s if g1s / g2s > 1 else g2s / g1s)-1) * 100,
                   'Content Spread': (sd.FxForMPP(trimmedGraph, sources, p) - results['Initial Content Spread']) / results['Initial Content Spread'] * 100,
                   'Time': end_time - start_time}
    results['IRFA'] = irfaResults

    #ud = trimmedGraph.to_undirected()
    #nds = list(ud.nodes())
    #indexedEdges = {}
    #for source in friendsOfFriends:
    #   indexedEdges[nds.index(source)] = set()
    #   for dest in friendsOfFriends[source]:
    #       indexedEdges[nds.index(source)].add(nds.index(dest))
    #start_time = time.time()
    #edges = spgreedy.spgreedy(ud, indexedEdges, g1, g2, p, 5)
    #end_time = time.time()
    #RG = trimmedGraph.copy()
    #RG.add_edges_from([(nds[edge[0]], nds[edge[1]]) for edge in edges])
    #g1s, g2s = sd.FxForMPPFair(RG, sources, g1names, g2names, p)
    #results['SPGREEDY Final Disparity'] = (g1s / g2s - 1 if g1s / g2s > 1 else g2s / g1s - 1) * 100
    #results['SPGREEDY Lift'] = (sd.FxForMPP(RG, sources, p) - results['Initial Content Spread'])/results['Initial Content Spread'] * 100
    #results['SPGREEDY Time'] = end_time - start_time

    #originalGraph = trimmedGraph.to_undirected()
    #candidateEdgesGraph =  nx.create_empty_copy(originalGraph)
    #for source in friendsOfFriends:
    #    for dest in friendsOfFriends[source]:
    #        candidateEdgesGraph.add_edge(source, dest)
    #start_time = time.time()
    #edges = ACR.selectGreedy(ACR.ACRFoF(originalGraph, candidateEdgesGraph, .5), k)
    #end_time = time.time()
    #RG = trimmedGraph.copy()
    #RG.add_edges_from(edges)
    #g1s, g2s = sd.FxForMPPFair(RG, sources, g1names, g2names, p)
    #results['ACR Final Disparity'] = (g1s / g2s - 1 if g1s / g2s > 1 else g2s / g1s - 1) * 100
    #results['ACR Lift'] = (sd.FxForMPP(RG, sources, p) - initialContentSpread)/initialContentSpread * 100
    #results['ACR Time'] = end_time - start_time


    pickle.dump(results, open('irfa-scale-results-'+str(scale)+'.pickle', 'wb'))

def generateMediumLargeGrahps(p=0.5):
    G, sources, nodeToGender = pickle.load(open('50000-graph74.ls', 'rb'))

    graphSamples = []
    for i in range(20):
        print(i)
        trimmedGraph = ff.start_a_fire(G, 10000)
        g1names, g2names = gen.getGroupsNames(nodeToGender, list(trimmedGraph.nodes()))
        g1, g2 = gen.getGroups(nodeToGender, list(trimmedGraph.nodes()))

        for j in range(20):
            sources = [random.sample(list(trimmedGraph.nodes()), 6), None]
            g1s, g2s = sd.FxForMPPFair(trimmedGraph, sources[0], g1names, g2names, p)
            disparity = g1s/g2s if g1s/g2s > 1 else g2s/g1s
            graphSamples.append((disparity, trimmedGraph, sources))
    
    graphSamples.sort(key=lambda e: e[0], reverse=True)
    pickle.dump(graphSamples[:5], open('10000-node-samples.pickle', 'wb'))

    for i in range(5):
        _, tenKGraph, tenKSources = graphSamples[i]
        shortestDistances = sd.multipleSourceShortestDistances(tenKGraph, tenKSources[0])
        mscff4000 = ff.MultipleSourceContinuingFF(tenKGraph, tenKSources[0], 4000, shortestDistances, 1)
        subgraph4000, _, _ = mscff4000.alternatingForestFire()
        shortestDistances = sd.multipleSourceShortestDistances(subgraph4000, tenKSources[0])
        mscff2000 = ff.MultipleSourceContinuingFF(subgraph4000, tenKSources[0], 2000, shortestDistances, 1)
        subgraph2000, _, _ = mscff2000.alternatingForestFire()
        shortestDistances = sd.multipleSourceShortestDistances(subgraph2000, tenKSources[0])
        mscff1000 = ff.MultipleSourceContinuingFF(subgraph2000, tenKSources[0], 1000, shortestDistances, 1)
        subgraph1000, _, _ = mscff1000.alternatingForestFire()
        shortestDistances = sd.multipleSourceShortestDistances(subgraph1000, tenKSources[0])
        mscff500 = ff.MultipleSourceContinuingFF(subgraph1000, tenKSources[0], 500, shortestDistances, 1)
        subgraph500, _, _ = mscff500.alternatingForestFire()

        pickle.dump((tenKSources[0], tenKGraph, subgraph4000, subgraph2000, subgraph1000, subgraph500), open('medium-scaling-samples-'+str(i)+'.pickle', 'wb'))

def generate1000NodeGraphs():
    G, _, _ = pickle.load(open('100000-graph74.pickle', 'rb'))

    graphs = []
    for x in range(20):
        print('.')
        source = random.sample(list(G.nodes()), 1)
        res = ff.start_a_fire(G, 1000, source)
        while res.number_of_nodes() != 1000:
            res = ff.start_a_fire(G, 1000, source)
        graphs.append(res)

    pickle.dump(graphs, open('1000-node-graphs.pickle', 'wb'))

def generate1000NodeSources():
    _, _, nodeToGender = pickle.load(open('100000-graph74.pickle', 'rb'))
    graphs = pickle.load(open('1000-node-graphs.pickle', 'rb'))

    graphsAndSources = {}

    for s in [3,6,9,12]:
        graphsAndSources[s] = []
        while len(graphsAndSources[s]) < 20:
            print('.')
            for graph in range(len(graphs)):
                sources = [random.sample(list(graphs[graph].nodes()), s), None]
                g1names, g2names = gen.getGroupsNames(nodeToGender, list(graphs[graph].nodes()))
                g1s, g2s = sd.FxForMPPFair(graphs[graph], sources[0], g1names, g2names, 0.5)
                disparity = g1s/g2s if g1s/g2s > 1 else g2s/g1s
                if (disparity > 1.2 and disparity < 1.25):
                    graphsAndSources[s].append((graphs[graph], sources, nodeToGender))
                    if len(graphsAndSources[s]) == 20:
                        break

    pickle.dump(graphsAndSources, open('1000-graphs-and-sources.pickle', 'wb'))


def main():
    threads = []

    pool = Pool(4)

    #generateMediumLargeGrahps()
    #for i in [11, 13, 14, 19, 3, 6, 20]:
    #   pool.apply_async(OptimalityExperiment, args=(i,))

    #_, _ , nodeToGender = pickle.load(open('100000-graph74.pickle', 'rb'))
    #for nodesPerIteration in [200,400,600,800]:
    #   for k in [2,3,4,5,]:p
    #       for trial in range(5):
    #           pool.apply_async(ForestFireOptions, args=(trial, k, nodesPerIteration, nodeToGender))

    # ForestFire100000()
    # ForestFireOptions(0, 4, 600, nodeToGender)
    #Experiment10()

    # VaryingKExperiment(3, 0.5, 'fof')
    # for k in [2,3,4,5]:
    #     pool.apply_async(VaryingKExperiment, args=(k,0.5,'community'))
    #     pool.apply_async(VaryingKExperiment, args=(k,0.5,'community', True))
    #     pool.apply_async(VaryingKExperiment, args=(k,0.5,'fof', True))
    # for sources in [3,6,9,12]:
        # pool.apply_async(VaryingSourcesExperient, args=(sources,0.5,'community'))
    #    pool.apply_async(VaryingSourcesExperient, args=(sources,0.5,'fof'))
        # pool.apply_async(VaryingSourcesExperient, args=(sources,0.5,'community', True))
    # for p in [0.3, 0.5, 0.7,0.9]:
    #     pool.apply_async(VaryingPExperient, args=(p, 'community'))
    #     pool.apply_async(VaryingPExperient, args=(p, 'fof', True))
    #     pool.apply_async(VaryingPExperient, args=(p, 'community', True))

    #    generate1000NodeGraphs()
    # generate1000NodeSources()

    # MediumGraphsExperiment('ForestFire', 0, 0, 'fof', .5)
    # MediumGraphsExperiment('`IterFCS`', 0, 0, 'fof', .5)
    # for algorithm in ['ForestFire']:
    for algorithm in ['ForestFire', 'CG', 'IterFCS', 'SPGREEDY', 'ACR', 'IRFA']:
        for size in range(4):
            for trial in range(5):
                MediumGraphsExperiment(algorithm, size, tiral, 'fof', p=0.5)
    #MediumGraphsExperiment('SPGREEDY', 5, 0, 'fof', p=0.5)
    #MediumGraphsExperiment('ACR', 5, 0, 'fof', p=0.5)
    #MediumGraphsExperiment('CG', 5, 0, 'fof', p=0.5)
    #MediumGraphsExperiment('ForestFire', 5, 0, 'fof', p=0.5)
    #MediumGraphsExperiment('IRFA', 5, 0, 'fof', p=0.5)

    #for trial in range(5):
    #     MediumGraphsExperiment('IterFCS', 0, trial, 'fof', p=0.5)

    #MediumGraphsExperiment('ACR', 0, 0, 'fof', p=0.5)

    #generateOptimalFailGraphs()
    #generateFullGraphSources()
    # ForestFireFull()
    # generateFoF()
    #sys.setrecursionlimit(550000000)


    #for i in [50000, 100000, 200000, 500000]:
    #   ForestFireScale(i)
    # ForestFireScale(50000)
    # generateScalingGraphsAndSources()
    pool.close()
    pool.join()

    print('all done!')

if __name__ == "__main__":
    main()
