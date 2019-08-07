#!/usr/bin/env python

import networkx as nx
import time
import re

def ligand_topology(filename):
    G = nx.Graph()
    f = open(filename, 'r')
    atm = {}
    for line in f.readlines():
        if line.startswith("ATOM"):
            entry = re.split('\s+', line)
            atm[entry[1]] = {'type': entry[2], 'charge': float(entry[3])}
        if line.startswith('BOND'):
            entry = re.split('\s+', line)
            G.add_node(entry[1], **atm[entry[1]])
            G.add_node(entry[2], **atm[entry[2]])
            G.add_edge(entry[1], entry[2])
        if line.startswith('LONEPAIR'):
            entry = re.split('\s+', line)
            G.add_node(entry[2], **atm[entry[2]])
            G.add_node(entry[3], **atm[entry[3]])
            G.add_edge(entry[2], entry[3])
    return G

def is_cycle(graph, source):
    def dfs(node):
        visited.append(node)
        for each in graph[node]:
            if (cycle):
                return 
            if graph.degree(each) < 2:
                continue
            if each not in visited:
                spanning_tree[node] = each
                dfs(each)
                nodes.append(each)
                if each != spanning_tree[source] and source in graph[each]:
                    cycle.append(each)
            if not cycle and node == source:
                while nodes: nodes.pop()
                pass
                #while cycle: cycle.pop()
                #while visited: visited.pop()
    
    visited = []
    cycle = []
    nodes = []
    spanning_tree = {}
    dfs(source)
    return nodes

def find_cycle_root(graph, subgraph):
    # this is not true, but i'll just skip when a ring has more than one entry point
    count = 0
    entry = list(subgraph.nodes())[0]
    for n in subgraph:
        if graph.degree(n) is not subgraph.degree(n):
            count +=1
            entry = n
    if count > 1: return None
    
    root = subgraph[entry].keys()[0]
    return root

def is_symmetric(graph, subgraph, nbunch):
    # build distance list from the root element
    # TODO: currently, it only compares the atom types to determine whether it is symmetric. 
    # this is potential pot hole because if some kind of isomers are there, it will not able
    # to detect it.

    root = find_cycle_root(graph, subgraph)
    if not root: return False

    dist = {}
    for n in nbunch:
        if n == root: continue
        distance = nx.shortest_path_length(subgraph,root, n)
        if dist.has_key(distance): dist[distance].append(n)
        else: dist[distance] = [n]
        # reject cyclohexane
        if graph.degree(n) > 3: return False
    
    for d,nodes in dist.items():
        sub = None
        if len(nodes) < 2: continue
        for n in nodes:
            if sub == None: sub = set([x[0] for x in subgraph.neighbors(n)])
            else: 
                if sub != set([x[0] for x in subgraph.neighbors(n)]): return False
    
    return True
  
#def find_all_cycles(graph, build=False):
#    # remove all nodes that have only one connection
#    degrees = {}
#    for node in graph.nodes():
#        degree = graph.degree(node)
#        if degrees.has_key(degree): degrees[degree].append(node)
#        else: degrees[degree] = [node]
#    
#    # find all cycles
#    max_degree = max(degrees.keys())
#    cycles = []
#    for i in range(max_degree, 2, -1):
#        if not degrees.has_key(i): continue
#        for node in degrees[i]:
#            cycle = set(is_cycle(graph, node))
#            
#            # uniqify cycles
#            if cycle:
#                uc = [node]
#                for c in cycle:
#                    u = set(is_cycle(graph, c))
#                    if node not in u: continue
#                    else: uc.append(c)
#                h = G.subgraph(uc)
#                s = set(nx.node_connected_component(h, node))
#                if s not in cycles:
#                    cycles.append(s)
#    
#    if build:
#        tmp_cycles = []
#        for cycle in cycles:
#            h = nx.Graph()
#            h.add_edges_from(graph.edges(cycle))
#            tmp_cycles.append(h)
#        cycles = tmp_cycles
#                    
#    return cycles

def find_all_sym_cycles(graph):
    #cycles = find_all_cycles(graph)
    cycles = nx.cycle_basis(graph)
    sym_cycles = []
    
    for cycle in cycles:
        h = nx.Graph()
        h.add_edges_from(graph.edges(cycle))
        is_sym = is_symmetric(graph, h, cycle)
        if is_sym:
            sym_cycles.append(h)
    
    return sym_cycles

def group_nodes(graph, size=3):
    root = nx.center(graph)[0]
    groups = []
    group = []
    
    for n in nx.dfs_preorder_nodes(graph, root):
        if graph.degree(n) < 2: continue
        # hit the terminal
        if len(group) >= 1 and not set(graph[n]).intersection(set(group)):
            groups.append(group)
            group = []
        # size cut
        if len(group) >= size:
            groups.append(group)
            group = []
        group.append(n)
    if group not in groups: groups.append(group)
    
    for i in range(len(groups)):
        # build sub-graph that includes degree = 1 elements
        group = groups[i]
        for n in group:
            for nn in graph[n]:
                if graph.degree(nn) == 1: group.append(nn)
        groups[i] = group
        
    return groups

_tert = nx.Graph([(1, 2), (1, 3), (1, 4)])
def find_all_tert_sym_groups(graph):
    # remove hydrogens from the graph
    edges = [e for e in graph.edges() if e[0][0] != 'H' and e[1][0] != 'H']
    _graph = nx.Graph(edges)
    gm = nx.algorithms.isomorphism.GraphMatcher(_graph, _tert)
    terts = {}
    for match in gm.subgraph_isomorphisms_iter():
        # tert groups only composed of same atom type
        if len(set([x[0] for x in match.keys()])) > 1: continue
        
        # atoms has to be terminal
        flip = dict([(v, k) for (k, v) in match.iteritems()])
        if terts.has_key(flip[1]): continue
        if len(_graph[flip[2]]) > 1: continue
        if len(_graph[flip[3]]) > 1: continue
        if len(_graph[flip[4]]) > 1: continue
        terts[flip[1]] = (flip[2], flip[3], flip[4])
        
    return terts

if __name__ == '__main__':
    # ubigraph server should already be running
    #G = nx.UbiGraph()
    import sys, os

    filename = sys.argv[1]
    basename = os.path.dirname(filename)
    rtfname = os.path.basename(filename)
    ligandname = rtfname.split('.')[0]
    G = ligand_topology(filename)
    rtflines = open(filename, 'r').readlines()

    # dihedral setup
    fp = open('ndihe.str', 'w')
    cycles = find_all_sym_cycles(G)
    i = 1
    dihe = []
    cons = []
    for cycle in cycles:
        dihe1 = dihe2 = dihe3 = dihe4 = None
        root = find_cycle_root(G, cycle)
        dihe2 = root
        for n in cycle[root]:
            if cycle.degree(n) != G.degree(n): dihe3 = n
        for n in cycle[root]:
            if cycle.degree(n) == 1: continue
            if n not in [dihe2, dihe3]: dihe1 = n
        if dihe3 != None:
            for n in G[dihe3]:
                if G.degree(n) == 1: continue
                if n not in [dihe2, dihe3]: dihe4 = n
        else: continue
        dihe_str = "set diheatom%d = %s %s %s %s" % (i, dihe1, dihe2, dihe3, dihe4)
        i += 1
        dihe.append(dihe_str)
        [cons.append(x) for x in cycle.nodes() if not x.startswith('H')]
    
    # find tert-butyl
    terts = find_all_tert_sym_groups(G)
    for root,tert in terts.items():
        dihe2 = root
        dihe1 = tert[0]
        dihe3 = [x for x in G[root].keys() if x not in tert].pop()
        try:    dihe4 = [x for x in G[dihe3].keys() if x != dihe2 and x[0] != 'H'].pop()
        except: continue
        dihe_str = "set diheatom%d = %s %s %s %s" % (i, dihe1, dihe2, dihe3, dihe4)
        dihe.append(dihe_str)

    fp.write("""* dihedral setup for symmetric unit
*

set ndihe = %d
%s
""" % (len(dihe), "\n".join(dihe)))
    if cons: fp.write("""set consatom = %s
""" % (" ".join(cons)))
    fp.close()
    
    # rtf group
    groups = group_nodes(G, 2)
    nodes = G.nodes(data=True)
    str = ""
    for group in groups:
        str += "GROUP\n"
        for n in group:
            str += "ATOM %6s %6s %8.4f\n" % (n,G.node[n]['type'],G.node[n]['charge'])

    flag = False
    fp = open("%s_g.rtf" % ligandname, 'w')
    for line in rtflines:
        if line.startswith('ATOM') or line.startswith('GROUP'):
            flag = True
            continue
        if flag and not (line.startswith('ATOM') or line.startswith('GROUP')):
            flag = False
            fp.write(str)
        fp.write(line)
    fp.close()
