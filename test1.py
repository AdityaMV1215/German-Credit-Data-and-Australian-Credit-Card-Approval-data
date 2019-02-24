import sklearn
import bs4
import networkx as nx
import heapq

G = nx.Graph()
nodes = [1,2,3,4,5,6,7,8,9]
edges = [(1,2,{'weight':2}),(1,3,{'weight':3}),(2,3,{'weight':4}),(2,5,{'weight':4}),(3,5,{'weight':5}), (3,4,{'weight':3}), (4,5,{'weight':1})]

G.add_nodes_from(nodes)
G.add_edges_from(edges)

def ucs(g, src, tar):
    if src not in G.nodes():
        return "Invalid input"
    visited = []
    q = []
    heapq.heappush(q,(0,src,[src]))
    while len(q) != 0:
        cost, temp, path = heapq.heappop(q)
        if temp == tar:
            return cost, path
        else:
            visited.append(temp)
            for x in g.neighbors(temp):
                if x not in visited:
                    heapq.heappush(q,(G[temp][x]['weight']+cost, x, path+[x]))

    return -1, []

for i in G.nodes():
    for j in G.nodes():
        if i != j:
            cost, path = ucs(G,i,j)
            print("Shortest distance between {} and {} is {} with path {}".format(i, j, cost, path))





