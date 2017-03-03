"""
collect.py

collecting data from twitter.
We have a target twitter screen_name, and we want to find 100 twitter than mention that target.
The information we collected is the user's screen_name, the user's description, and the content of the tweets.
Also, we collect the friendship relations between target user and other 100 users we collected. 
All the information we collect in this part will be stored in dataset.csv
"""
"""
get 100 twitter mention taylor. and pick several of it that may interest taylor swift most.
First, collect data. then filter the twitter that tweet by herself.
"""
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import pandas as pd
import os
import time

consumer_key = 'qXiSj2jzG23YXCQQsHBIJzfAu'
consumer_secret = 'jtcZKyciqGm8DlOQduKQNWPwKvmXvclzTBgUY88Y9VqW0yfS3b'
access_token = '3666905002-fBNKWXAYcI3FHABK01Onzaaa4A1bVI98rZw8KGE'
access_token_secret = 'JrB4H8I4cMOpgsxEihVZHNJuqDG0ChSX0DbM5dPK4ziMS'
target = 'taylorswift13'

twitter = TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

#may be this tweets should be a little popular, and the data should be recently.
# in cluster, cluster the user. and the content of the tweets.
# in classify, use lexicon anaylsis the neg/pos of the tweets. pick top neg and top pos
# also, match the tweets taylor has tweets, if the content is similar....
request = twitter.request('search/tweets',{'q': 'taylor swift', 'count' :100})
data = [items for items in request]
#items = [r for r in request if r['screen_name'] != 'taylorswift13']
#we only need the screen name to search the relationships and the text to analysis the content.
#store it as a dataframe.
new_file = open('datasets.csv', 'w+')
users = []
text = []
description = []
relationship = []
ids = []
for tweet in data:
	if tweet['user']['screen_name'] != 'taylorswift13':
		users.append(tweet['user']['screen_name']) 
		#ids.append(tweet['user']['id'])
		text.append(tweet['text'])
		name = tweet['user']['screen_name']
		request2 = twitter.request('users/show',{'screen_name': name})
		temp = [r for r in request2]
		description.append(temp[0]['description'])

#find relationship between the user and taylor. 
# 0 represent no relationship.
# 1 represent following  taylor
# 2 represent followed by taylor
# 3 represent 1 & 2
for u in users:
	request5 = twitter.request('friendships/show', {'source_screen_name': u, 'target_screen_name': target})
	temp = [r for r in request5]
	relation = 0
	if temp[0]['relationship']['target']['following'] == True:
		relation += 2
	if temp[0]['relationship']['target']['followed_by'] == True:
		relation += 1
	print('name is %s, ralation is %d' %(u, relation))
	relationship.append(relation)
dic = {'name': users, 'text': text, 'description': description, 'relationship': relationship} #, 'friends': friends}
df = pd.DataFrame(dic)
df.to_csv('datasets.csv')
new_file.close()

#done.