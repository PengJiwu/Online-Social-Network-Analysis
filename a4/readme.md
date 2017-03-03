1.collect.py
Collecting data from twitter.
We have a target twitter screen_name, and we want to find 100 twitter than mention that target.
Using twitter api to collect the content of the tweets, the relationship of the users etc.
All the information we collect in this part will be stored in datasets.csv
2.cluster.py
  a. Draw a graph indicate the relationship between the target and the users we collected. 
     Store this graph as relation1.png, that is: Each node represent one user or target. However, only target node show the name of target, other node only show the index . If the user following target, there will be an edge from the user point to target, if target following the user, there will be an edge from target point to the user. Otherwise, there will be no edges between two node.
  b. Draw a graph indicates the relationship of the users we collected. 
      Store this graph as relation2.png. That is: Each node represent one user, no node for target in this graph. If two user have a very high similarity, there will be an edge between these two node.
  c. Using girvan newman algorithm to partition the graph in step 2 to several community.  
      Save the result of clustering to cluster_result.csv.
3. classify.py
Classify the sentiment of each tweet. Pick the top 3 positive and top 1 negative, print these tweets.
Save the top 10 positive result and top 10 negative result to classify_result.csv.

Conclusions:
In the classify result, not all the result is right. The sentiment of one sentence is hard to classify just use lexicon method.
However, the highest positive scored tweets and the highest negative scored tweets seems reasonable.
