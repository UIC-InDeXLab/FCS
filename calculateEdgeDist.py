import shortestDistances as sd
import buildNetworkX as bnx

def calculateEdgeDist(G, edges, nodes, sources, length):
    #return value
    distances = {}
    i = 1

    #convert candidate edges to list

    
    #find shortest distances adding each edge
    for edge in range(len(edges)):
        G.add_edge(edges[edge][0], edges[edge][1])
        result = sd.shortestDistances(G, sources)
        i += 1

        #update edge distances for each node
        for node in range(len(nodes)):
            if nodes[node] in result:


                distanceOnEdge = result[nodes[node]]
                
                #add edge to node not yet in distances
                if not node in distances:
                    distances[node] = {}
                    if distanceOnEdge <= length:
                        distances[node][distanceOnEdge] = [edge]

                #add edge to node already in distances
                else:
                    if distanceOnEdge <= length:
                        #new distance for node
                        if not distanceOnEdge in distances[node]:
                            distances[node][distanceOnEdge] = [edge]

                        #add aedge to distance for node
                        else:
                            distances[node][distanceOnEdge].append(edge)

        G.remove_edge(edges[edge][0], edges[edge][1])
    return distances

def getNodeNeighbors(nodes, edges):
    nodeNeighbors = {}
    for node in range(len(nodes)):
        nodeNeighbors[node] = []
        for edge in range(len(edges)):
            if edges[edge][0] == nodes[node]:
                nodeNeighbors[node].append(edge)
    return nodeNeighbors

def test():
    adjacencyLists = {}
    adjacencyLists['1'] = {'2', '3'}
    adjacencyLists['2'] = {'1', '5'}
    adjacencyLists['3'] = {'2', '4'}

    G = bnx.buildNetworkXFromAM(adjacencyLists)

    nodes = ['1','2','3','4','5']
    edges = [('1', '4'), ('1', '5'), ('3', '5')]
    
    sources = ['1']

    result = calculateEdgeDist(G, edges, nodes, sources, 3)

    print(result)

    result = getNodeNeighbors(nodes, edges)
    
    print (result)