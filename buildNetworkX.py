import networkx as nx

def buildNetworkXFromAM(adjacencyLists):
    G = nx.DiGraph()
    for source in adjacencyLists:
        if not G.has_node(source):
            G.add_node(source)
        for dest in adjacencyLists[source]:
            if not G.has_node(dest):
                G.add_node(dest)
            G.add_edge(source, dest)
    return G

def getGender(nodeToGender, node):
    if (nodeToGender[node] == "1"):
        return "male"
    else:
        return "female"

def buildGenderedNetworkXFromAM(adjacencyLists, nodeToGender):
    G = nx.DiGraph()
    for source in adjacencyLists:
        if not G.has_node(source):
            G.add_node(source, gender=getGender(nodeToGender, source))
        for dest in adjacencyLists[source]:
            if not G.has_node(dest):
                G.add_node(dest, gender=getGender(nodeToGender, dest))
            G.add_edge(source, dest)
    return G


def test():
    adjacencyLists = {}
    adjacencyLists['1'] = {'2', '3'}
    adjacencyLists['2'] = {'1', '5'}
    adjacencyLists['3'] = {'2', '4'}

    G = buildNetworkXFromAM(adjacencyLists)

    if not G.has_node('1'):
        print('tests failed! 1 missing from graph')
    elif not G.has_node('2'):
        print('tests failed! 2 missing from graph')
    elif not G.has_node('3'):
        print('tests failed! 3 missing from graph')
    elif not G.has_node('4'):
        print('tests failed! 4 missing from graph')
    elif not G.has_node('5'):
        print('tests failed! 5 missing from graph')
    elif len(G.nodes()) != 5:
        print('tests failed! does not have 5 nodes')
    elif not G.has_edge('1','2'):
        print('tests faild! no edge from 1 to 2')
    elif not G.has_edge('1','3'):
        print('tests faild! no edge from 1 to 3')
    elif not G.has_edge('2','1'):
        print('tests faild! no edge from 2 to 1')
    elif not G.has_edge('2','5'):
        print('tests faild! no edge from 2 to 5')
    elif not G.has_edge('3','2'):
        print('tests faild! no edge from 3 to 2')
    elif not G.has_edge('3','4'):
        print('tests faild! no edge from 3 to 4')
    elif len(G.edges()) != 6:
        print('tests failed! does not have 6 edges')

    else:
        print('tests passed!')
