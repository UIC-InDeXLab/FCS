import networkx
import numpy as np
import networkx as nx


def dfs_visit_recursively(g, node, nodes_color, edges_to_be_removed):
    nodes_color[node] = 1
    nodes_order = list(g.successors(node))
    nodes_order = np.random.permutation(nodes_order)
    for child in nodes_order:
        if nodes_color[child] == 0:
            dfs_visit_recursively(g, child, nodes_color, edges_to_be_removed)
        elif nodes_color[child] == 1:
            edges_to_be_removed.append((node, child))

    nodes_color[node] = 2


def dfs_remove_back_edges(g):
    '''
    0: white, not visited
    1: grey, being visited
    2: black, already visited
    '''

    nodes_color = {}
    edges_to_be_removed = []
    for node in list(g.nodes()):
        nodes_color[node] = 0

    nodes_order = list(g.nodes())
    nodes_order = np.random.permutation(nodes_order)
    num_dfs = 0
    for node in nodes_order:

        if nodes_color[node] == 0:
            num_dfs += 1
            dfs_visit_recursively(g, node, nodes_color, edges_to_be_removed)

    return edges_to_be_removed

def testRemoveEdges():
    Graph = networkx.DiGraph()
    Graph.add_edges_from([(3,1),(1,4),(3,2),(4,3),(4,5),(5,4),(6,5),(4,7),(6,1),(7,6),(7,4)])
    print(dfs_remove_back_edges(Graph))
    print(backEdgesNonRecursive(Graph))
    print(remove_cycle_edges_by_mfas(Graph))


def backEdgesNonRecursive(g):
    g = g.copy()
    backEdges = set()
    previous = []
    previousDict = {}
    seenNodes = {}
    for node in list(g.nodes()):
        if node in seenNodes:
            continue
        stack = [(node, 0)]
        while len(stack) > 0:
            # print(stack)
            g.remove_edges_from(list(backEdges))
            current, count = stack.pop()
            seenNodes[current] = True
            for removed in previous[count:]:
                del previousDict[removed]
            previous = previous[:count]
            neighborFound = False
            # print(current)
            # print(previousDict)
            # print(list(g.successors(current)))
            for neighbor in list(g.successors(current)):
                if neighbor in previousDict:
                    backEdges.add((current, neighbor))
                else:
                    neighborFound = True
                    stack.append((neighbor, count+1))
            if neighborFound:
                previous.append(current)
                previousDict[current] = True

    return list(backEdges)

def filter_big_scc(g,edges_to_be_removed):
    g.remove_edges_from(edges_to_be_removed)
    sub_graphs = filter(lambda scc: scc.number_of_nodes() >= 2, [g.subgraph(x).copy() for x in nx.strongly_connected_components(g)])
    return sub_graphs

def get_nodes_degree_dict(g,nodes):
    in_degrees = g.in_degree(nodes)
    out_degrees = g.out_degree(nodes)
    degree_dict = {}
    for node in nodes:
        in_d = in_degrees[node]
        out_d = out_degrees[node]
        if in_d >= out_d:
            try:
                value = in_d * 1.0 / out_d
            except:
                value = 0
            f = "in"
        else:
            try:
                value = out_d * 1.0 / in_d
            except:
                value = 0
            f = "out"
        degree_dict[node] = (value,f)
        #print("node: %d: %s" % (node,degree_dict[node]))
    return degree_dict

def pick_randomly(source):
    np.random.shuffle(source)
    np.random.shuffle(source)
    np.random.shuffle(source)
    return source[0]

def pick_from_dict(d, order="max"):
    min_k, min_v = 0, 10000

    min_items = []
    max_k, max_v = 0, -10000

    max_items = []
    for k, v in d.items():
        if v > max_v:
            max_v = v
            max_items = [(k, max_v)]
        elif v == max_v:
            max_items.append((k, v))

        if v < min_v:
            min_v = v
            min_items = [(k, min_v)]
        elif v == min_v:
            min_items.append((k, v))

    max_k, max_v = pick_randomly(max_items)
    min_k, min_v = pick_randomly(min_items)

    if order == "max":
        return max_k, max_v
    if order == "min":
        return min_k, min_v
    else:
        return max_k, max_v, min_k,

def greedy_local_heuristic(sccs,degree_dict,edges_to_be_removed):
    while True:
        graph = sccs.pop()
        temp_nodes_degree_dict = {}
        for node in graph.nodes():
            temp_nodes_degree_dict[node] = degree_dict[node][0]
        max_node,_ = pick_from_dict(temp_nodes_degree_dict)
        max_value = degree_dict[max_node]
        if max_value[1] == "in":
            edges = [(max_node,o) for o in graph.neighbors(max_node)]
        else:
            edges = [(i,max_node) for i in graph.predecessors(max_node)]
        edges_to_be_removed += edges
        sub_graphs = filter_big_scc(graph,edges_to_be_removed)
        if sub_graphs:
            for index,sub in enumerate(sub_graphs):
                sccs.append(sub)
        if not sccs:
            return

def scc_nodes_edges(g):
    scc_nodes = set()
    scc_edges = set()
    num_big_sccs = 0
    num_nodes_biggest_scc = 0
    biggest_scc = None
    for sub in nx.strongly_connected_components(g):
        sub = g.subgraph(sub).copy()
        number_nodes = sub.number_of_nodes()
        if number_nodes >= 2:
            scc_nodes.update(sub.nodes())
            scc_edges.update(sub.edges())
            num_big_sccs += 1
            if num_nodes_biggest_scc < number_nodes:
                num_nodes_biggest_scc = number_nodes
                biggest_scc = sub
    if biggest_scc == None:
        return scc_nodes,None, None, None
    return scc_nodes, None, None, None

def get_big_sccs(g):
    num_big_sccs = 0
    big_sccs = []
    for sub in nx.strongly_connected_components(g):
        sub = g.subgraph(sub).copy()
        number_of_nodes = sub.number_of_nodes()
        if number_of_nodes >= 2:
            num_big_sccs += 1
            big_sccs.append(sub)
    return big_sccs

def remove_cycle_edges_by_mfas(g):
    scc_nodes,_,_,_ = scc_nodes_edges(g)
    degree_dict = get_nodes_degree_dict(g,scc_nodes)
    sccs = get_big_sccs(g)
    edges_to_be_removed = []
    greedy_local_heuristic(sccs,degree_dict,edges_to_be_removed)
    edges_to_be_removed = list(set(edges_to_be_removed))
    g.remove_edges_from(edges_to_be_removed)
    return edges_to_be_removed

testRemoveEdges()