import math

import buildNetworkX as bnx
import networkx

def friendsOfFriendsNX(G):
    fof = {}
    for node in list(G.nodes()):
        fof[node] = set()
        for friend in list(G.neighbors(node)):
            for friendOfFriend in list(G.neighbors(friend)):
                if (not friendOfFriend in G.neighbors(node)) and (friendOfFriend != node):
                    fof[node].add(friendOfFriend)
    return fof

def CommunityMembership(G: networkx.DiGraph):
    communities = []
    for node in list(G.nodes()):
        community = set([node])
        community.update(G.neighbors(node))
        communities.append(community)
    scores = {}
    for node in list(G.nodes()):
        scores[node] = {}
        neighbors = set(G.neighbors(node))
        for community in communities:
            if len(neighbors.intersection(community)) > 0:
                for contact in community:
                    if contact not in neighbors and contact != node:
                        if contact not in scores[node]:
                            scores[node][contact] = 0
                        scores[node][contact] = scores[node][contact] + 1
    suggestions = {}
    for node in list(G.nodes()):
        sortedSuggestions = sorted(scores[node].items(), key=lambda e: e[1], reverse=True)
        suggestions[node] = [x[0] for x in sortedSuggestions[:math.floor(len(sortedSuggestions)/3)]]
    return suggestions


def test():
    adjacencyLists = {}
    adjacencyLists['1'] = {'2', '3'}
    adjacencyLists['2'] = {'1', '5'}
    adjacencyLists['3'] = {'2', '4'}

    g = bnx.buildNetworkXFromAM(adjacencyLists)

    result2 = friendsOfFriendsNX(g)

    if '2' in result2['1']:
        print("test failed! 2 in result!")
        return
    elif '3' in result2['1']:
        print('test failed! 3 in result!')
    elif '1' in result2['1']:
        print('test failed! 1 in result!')
    elif not '4' in result2['1']:
        print('test failed! 4 not in result!')
    elif not '5' in result2['1']:
        print('test failed! 5 not in result!')
    else:
        print('tests pased!')
