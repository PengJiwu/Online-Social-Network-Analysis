"""
sumarize.py
"""
import pandas as pd
import numpy as np
import os

info1 = pd.read_csv('datasets.csv')
users = info1['name'].tolist()
description = info1['description'].tolist()
num_user = len(users)
num_messages = 2 * len(description)

info2 = pd.read_csv('cluster_result.csv')
num_com = len(info2.index)
community = info2['cluster']
avg_user = num_user/num_com

info3 = pd.read_csv('classify_result.csv')
num_instances = 10
pos = info3['neg_content'].tolist()
neg = info3['pos_content'].tolist()

new_file = open('summary.txt', 'w+')
new_file.write('number of users collected: %d' % num_user)
new_file.write('\nnumber of messages collected: %d' % num_messages)
new_file.write('\nnumber of communities discovered: %d' % num_com)
new_file.write('\naverage number of users per community: %d' % avg_user)
new_file.write('\nnumber of instances per class found: %d' % num_instances)
new_file.write('\nexamples:')
new_file.write('\n  from positive class: %s' % pos[0])
new_file.write('\n  from negative class: %s' % neg[0])

new_file.close()
