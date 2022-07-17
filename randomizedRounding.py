import random

def round(edges, values):
    edgePerNode = {}
    for m in range(len(edges)):
        val = random.uniform(0,1)
        if values[m] >= val:
            if not edges[m][0] in edgePerNode:
                edgePerNode[edges[m][0]] = [edges[m][1]]
            else:
                edgePerNode[edges[m][0]].append(edges[m][1])
    return edgePerNode

def test():
    edges = [(1,2), (1,3), (2,3), (3,1)]
    values = [.65, .03, .2, .25]
    random.seed(42)
    edgePerNode = round(edges, values)
    if not 2 in edgePerNode[1]:
        print('test failed! should have edge from 1 to 2')
        return
    elif not 3 in edgePerNode[1]:
        print('test failed! should have edge from 1 to 3')
        return
    elif 2 in edgePerNode:
        print('test failed! should not havea any edges from 2')
        return
    elif not 1 in edgePerNode[3]:
        print('test failed! should have edge from 3 to 1')
        return

    edges = [(1,2), (1,3), (1,4), (1,5)]
    values = [.65, .03, .5, .25]
    random.seed(42)
    edgePerNode = round(edges, values)
    if len(edgePerNode[1]) != 2:
        print('test failed! should have no more than 2 edges per node')
    else:
        print('tests passed!')