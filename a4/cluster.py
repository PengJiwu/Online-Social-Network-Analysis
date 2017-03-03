"""
cluster.py

1. draw a graph indicate the relationship between the target and the users we collected. 
   If the user following target, there is an edge from user point to target. 
   If the user followed by target, there is an edge from target point to user.
   Store this graph as relation1.png

2. draw a graph indicates the relationship of the users we collected. 
   However, we collect the description of each user, and calculate there jaccard_similarity. 
   If the similarity is bigger than 0.1, there is an edge between these two users.
   We store this graph as relation2.png
3. Also, we user girvan newman algorithm to partition the graph in step 2 to several commiunity.  
   Print the items in each cluster.
"""
"""
In this file, I borrow some functions from assignment 1
the graph is: if the user is following taylor of followed by taylor, draw edges(directed)
cluster: collect information of that user, identify and cluster.
"""
from collections import Counter, deque, defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import pandas as pd
import numpy as np
import re

target = 'taylorswift13'
tweets = pd.read_csv('datasets.csv')



#arrow means the user follow the other user, point to the followed user.
def draw_graph(names, relationship):
	# this function we draw a graph to show the relationship between the users we gathered and the target.
	graph = nx.DiGraph()
	graph.add_node(target)
	for i in range(0, len(names)):
		graph.add_node(names[i])
		if relationship[i] == 1:
			graph.add_edge(names[i], target)
		if relationship[i] == 2:
			graph.add_edge(target,names[i])
		if relationship[i] == 3:
			graph.add_edge(target,names[i])
			graph.add_edge(names[i], target)
	#nx.draw(graph)
	#plt.show()
	labeldic = {}
	labeldic[target] = target
	for i in range(0, len(names)):
		if relationship[i] !=0:
			labeldic[names[i]] = i
		else:
			labeldic[names[i]] = ''
	pos = nx.spring_layout(graph)
	nx.draw_networkx(graph,pos,labels = labeldic, font_size= 8, alpha=0.5,node_size=5,width=0.5,edge_color ='r')
	plt.axis('off')
	plt.savefig('relation1.png')
	pass


"""
Flowing we draw a graph, which show the connection about the users we collect. 
However, we use the similarity of their description to draw the graph.
Then we user garvan newman algorithm to partition the graph into several components.
"""

def tokenize(doc):
    """
    Tokenize a string.
    """
    doc = doc.lower()
    results = doc.split()
    for it in range(0, len(results)):
    	results[it] = re.sub('\A\W+', '', results[it])
    	results[it] = re.sub('\W+\Z', '', results[it])
     
    return np.array(results)
    pass


def jaccard_sim(A,B):
	"""
	calculate the jaccard_sim between A and B.
	if the score is biger than 0.5, there are edges between two A,B
	"""
	#A,B is list
	sa = set(A)
	sb = set(B)
	if len(sa)==0 or len(sb)==0:
		return 0
	result = 1.0 * len(sa & sb) / len(sa | sb)
	return result



def get_description_jaccard(names, description):
	tokens = []
	for items in description:
		if type(items) != type("string"):
			tokens.append([])
		else:
			tokens.append(tokenize(items))
	tweets['d_tokens'] = tokens

	j_array = np.zeros((len(names),len(names)))
	for i in range(0,len(names)):
		for j in range(0,len(names)):
			if j < i:
				j_array[i][j] = j_array[j][i]
			elif j==i:
				j_array[i][j] = 0
			else:
				j_array[i][j] = jaccard_sim(tokens[i], tokens[j])
	jaccard = []
	for row in range(0,len(names)):
		jaccard.append(j_array[row])
	tweets['d_jaccard'] = jaccard
	return j_array
	pass


def draw_connection_graph(names, j_array):
	# if jaccard similarity bigger than 0, has a edge.
	graph2 = nx.Graph()
	for i in range(0, len(names)):
		graph2.add_node(names[i])
		for j in range(i, len(names)):
			if j_array[i][j] > 0.1: 
				graph2.add_edge(names[i],names[j])
	
	labeldic = {}
	for m in range(0, len(names)):
		labeldic[names[m]] = m

	pos = nx.spring_layout(graph2)
	nx.draw_networkx(graph2,pos,arrows = False, labels = labeldic, font_size= 8, alpha=1.0,node_size=20,width=0.5,edge_color ='r')
	plt.axis('off')
	plt.savefig('relation2.png')
	#print(len(graph2.edges()))
	return graph2


#now we use girvan newman algorigthm to partition the graph. some code borrowed from a1 assignment.
def bfs(graph, root, max_depth):
    """ 
    Perform breadth-first search to compute the shortest paths from a root node to all
    other nodes in the graph. To reduce running time, the max_depth parameter ends
    the search after the specified depth.
    """
  
    #modify letter use graph.neighbors(x)   https://github.com/iit-cs579/main/issues/190
    #inital 
    
    visited = deque()
    visiting = deque()
    node2num_paths = {}
    node2parents = {}
    node2distances = {}
    visiting.append(root)
    node2distances[root] = 0
    node2num_paths[root] = 1
    while len(visiting) >0:
    	node = visiting.popleft()
    	visited.append(node)
    	if node2distances[node] >= max_depth:
    		break
    	else:
    		for n in graph.neighbors(node):
    			if not(n in visited):
    				if n in visiting and node2distances[n] != node2distances[node]:
    					node2parents[n].append(node)
    					node2num_paths[n] = len(node2parents[n])
    				elif not(n in visiting):
    					visiting.append(n)
    					node2distances[n] = node2distances[node] + 1
    					node2num_paths[n] = 1
    					node2parents[n] = [node]
    return node2distances, node2num_paths, node2parents
    pass


def bottom_up(root, node2distances, node2num_paths, node2parents):
    """
    Compute the final step of the Girvan-Newman algorithm.
    """
    #inital each nodes' credit to be 1.
    node_credit = {}
    result = {}
    for items in node2distances:
      node_credit[items] = 1.0
    #sort the node2distances by the descending order of their level. thus root will be the last element.
    list2distances = sorted(node2distances.items(), key = lambda d:d[1], reverse = True)
    #scan each elements in the list2distances. for each node if its not root, scan its parent_list
    #####for each parents, update the credit of the parent node, add a new edge to the result.
    ########attention: if this node have more than one parent, we should split the credit of this node.
    for items in list2distances:
      
      credit = node_credit[items[0]]
      #there are no 'E' in node2parents
      if items[0] != root:
        plist = node2parents[items[0]]
        l = len(plist)
        if l == 0:
          break
        if l == 1:
          node_credit[plist[0]] = node_credit[plist[0]] + credit
          templist = [items[0], plist[0]]
          templist = sorted(templist)
          e = tuple(templist)
          result[e] = credit
        else:
          c = credit / l
          for parent in plist:
            node_credit[parent] += c
            templist = [items[0], parent]
            templist = sorted(templist)
            e = tuple(templist)
            result[e] = c
      else:
        break
    return result
    pass


def approximate_betweenness(graph, max_depth):
    """
    Compute the approximate betweenness of each edge, using max_depth to reduce
    computation time in breadth-first search.
    """
  
    # each node as a root, calculate each edges credit. 
    between = {}
    for node in graph.nodes():
      node2distances, node2num_paths, node2parents = bfs(graph, node, max_depth)
      result = bottom_up(node, node2distances, node2num_paths, node2parents)
      for items in result:
        if len(between) == 0:
          between[items] = result[items]
        elif items in between:
          between[items] += result[items]
        else:
          between[items] = result[items]
    for ele in between:
      between[ele] = between[ele] / 2
    return between
    pass



def clustering(graph2,names,description,count,max_depth):
	"""
	graph2: the graph we want to partition
	names: the names of the users we collect.
	description: the description of the users we collect
	count: the number of clusters
	max_depth: max depth to search
	"""

	# cluster the users, at first one cluster is the users have no connection to others.
	graph3 = graph2.copy()
	#remove nodes that have no neighbors, these is one cluster.
	c0 = []
	for node in graph3.nodes():
		#print(node)
		if len(graph3.neighbors(node)) == 0:
			#print('stop')
			graph3.remove_node(node)
			c0.append(node)


	cluster_num = 1 + nx.number_connected_components(graph3)
	if cluster_num < count:
		#get the approximate betweenness of the graph
		betweenness = approximate_betweenness(graph3, max_depth)
		#store the betweenness as a list, and sort the list. cause we want to remove the edges from the biggest betweenness.
		betlist = betweenness.items()
		betlist = sorted(betlist, key = lambda d: d[0][0])
		betlist = sorted(betlist, key = lambda d:d[1], reverse = True)
		
		#scan the elements in the list, remove. check the components.
		for ele in betlist:
			edge = ele[0]
			graph3.remove_edge(*edge)
			number = 1 + nx.number_connected_components(graph3)
			if number >= count:
				break
	components = [c.nodes() for c in nx.connected_component_subgraphs(graph3)]
	components.append(c0) 
	#print(components)
	file = {'cluster': components}
	dataf = pd.DataFrame(file)
	dataf.to_csv('cluster_result.csv')
	return components



def main():
	names = tweets['name'].tolist()
	description = tweets['description'].tolist()
	text = tweets['text'].tolist()
	relationship = tweets['relationship'].tolist()
	j_array = get_description_jaccard(names,description)
	graph = draw_connection_graph(names, j_array)
	clustering(graph,names, description,4,5)
	draw_graph(names,relationship)
	print('draw graph finished, relation1.png is the relation between target and users')
	print('relation2.png is the relation between users')
	print('cluster finished, the result of cluster saved in cluster_result.csv')



if __name__ == '__main__':
    main()



