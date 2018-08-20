
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import scipy
from sklearn.cluster import AffinityPropagation
import numpy as np
import pandas as pd
import networkx as nx
# import newspaper
# from newspaper import Article 

from os import listdir
from os.path import isfile, join
import itertools
import gensim
from gensim.models import Doc2Vec
# import textacy
# import textacy.keyterms
from gensim.summarization import keywords
import spacy
#spacy.load('en')


def nearestExempler(exempler, centers):
    dist = []
    for i in range(len(centers)):
        other_exempler = int(centers[i])
        dist.append(scipy.spatial.distance.euclidean(g.node[dictionary[exempler]]['vector'], g.node[dictionary[other_exempler]]['vector']))
    return dist

def nearbyCluster(dis):
    for i in range(1,len(dis)):
        if(len(doc_clusters[dis.index(sorted(dis)[i])]) < 5):
            continue
        else:
            return dis.index(sorted(dis)[i])
            break


docLabels_1 = [f for f in listdir("/Users/AR/Desktop/naturesCall/TOI_Data/actor1/") if f.endswith('.txt')]
docLabels_2 = [f for f in listdir("/Users/AR/Desktop/naturesCall/TOI_Data/actor2/") if f.endswith('.txt')]

docLabels = list(itertools.chain(docLabels_1,docLabels_2))

doc_tag = gensim.models.doc2vec.TaggedDocument
sentence_tag = gensim.models.doc2vec.TaggedLineDocument

#string = 'Shah Jahan build tajmahal. He was a prominent king of his time. His legacy traced back to founder babar of mughal dynasty in India. His father was also a very good ruler.'  

data = []
gensim_keyterms = []
gensim_keyterms_prob = []
 
i = 0
for doc in docLabels_1:
    try:   
        file = open('/Users/AR/Desktop/naturesCall/TOI_Data/actor1/' + doc, 'r').read()
        data.append(doc_tag(file.lower().split(), [docLabels_1[i]]))
        gensim_keyterms_prob.append([keywords(file.lower(),words =10, lemmatize = True,scores = True),docLabels_1[i]])
        gensim_keyterms.append(keywords(file.lower(),words =10, lemmatize = True))

    except:
        file = open('/Users/AR/Desktop/naturesCall/TOI_Data/actor1/' + doc, 'r').read()
        data.append(doc_tag(file.lower().split(), [docLabels_1[i]]))
        gensim_keyterms_prob.append([[('nan',0),('nan',0),('nan',0),('nan',0),('nan',0),('nan',0),('nan',0),('nan',0),('nan',0),('nan',0)],docLabels_1[i]])
        gensim_keyterms.append(['nan','nan','nan','nan','nan','nan','nan','nan','nan','nan'])

    i = i+1
    
j = 0
for doc in docLabels_2:
    try:
        file = open('/Users/AR/Desktop/naturesCall/TOI_Data/actor2/' + doc, 'r').read()
        data.append(doc_tag(file.lower().split(), [docLabels_2[j]]))
        gensim_keyterms_prob.append([keywords(file.lower(),words =10, lemmatize = True,scores = True), docLabels_2[j]])
        gensim_keyterms.append(keywords(file.lower(),words =10, lemmatize = True))
    except:
        file = open('/Users/AR/Desktop/naturesCall/TOI_Data/actor2/' + doc, 'r').read()
        data.append(doc_tag(file.lower().split(), [docLabels_2[j]]))
        gensim_keyterms_prob.append([[('nan',0),('nan',0),('nan',0),('nan',0),('nan',0),('nan',0),('nan',0),('nan',0),('nan',0),('nan',0)],docLabels_2[j]])
        gensim_keyterms.append(['nan','nan','nan','nan','nan','nan','nan','nan','nan','nan'])

    j = j+1


# In[ ]:


keyterms_prob_df = pd.DataFrame()
count = 0

for i in range(len(gensim_keyterms_prob)):
    word_column = gensim_keyterms_prob[i][0]
    for j in range(len(word_column)):
        keyterms_prob_df.loc[count,'keywords'] = word_column[j][0]
        keyterms_prob_df.loc[count,'prob'] = word_column[j][1]
        keyterms_prob_df.loc[count,'file_name'] = gensim_keyterms_prob[i][1]
        count = count+1

#keyterms_prob_df.to_csv('/Users/AR/Desktop/naturesCall/TOI_Data/keywords_prob.csv')

model= Doc2Vec(data, size=25, window=5, alpha=0.025, min_alpha=0.025)

for epoch in range(10):
    model.alpha -= 0.002 
    model.min_alpha = model.alpha 
    model.train(data,total_examples=model.corpus_count,epochs=model.iter)

## model.docvecs.offset2doctag ## to view documents tags


# In[ ]:


g = nx.DiGraph()

h = nx.Graph()

for i in range(len(docLabels)):
    g.add_node(i, actor=docLabels[i][11:17], vector=model.docvecs.doctag_syn0[i], date=docLabels[i][:10], keyword = gensim_keyterms[i].split('\n'), file_name = docLabels[i])
    h.add_node(i, actor=docLabels[i][11:17], vector=model.docvecs.doctag_syn0[i], date=docLabels[i][:10], keyword = gensim_keyterms[i].split('\n'), file_name = docLabels[i])
    if(i>=1):
        for j in list(g.node):
            if((j!=i)&(g.node[i]['actor'] == g.node[j]['actor'])):  
                dist = scipy.spatial.distance.euclidean(g.node[i]['vector'], g.node[j]['vector'])
                if(dist<8):
                    g.remove_node(i)
                    break

dictionary = {}
for i in range(len(g.node)):
    dictionary[i] = list(g.node)[i]
inv_dict = {v: k for k, v in dictionary.items()}

X = [g.node[i]['vector'] for i in (list(g.node))] ## to convert into numpy array


# In[ ]:


# Compute Affinity Propagation
af = AffinityPropagation().fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)
    
doc_clusters = {i: np.where(labels == i)[0] for i in range(n_clusters_)}

new_labels = labels
min_cluster_size = 5
new_cluster_indices = cluster_centers_indices

nearby_exempler = []
for i in range(len(cluster_centers_indices)):
    if(len(doc_clusters[i]) < min_cluster_size):
        exempler = int(cluster_centers_indices[i])
        distance = nearestExempler(exempler, cluster_centers_indices)
        nearby_exempler.append([i, nearbyCluster(distance)])      

for i in range(len(nearby_exempler)):
    documentsforRename = doc_clusters[nearby_exempler[i][0]]
    new_cluster_indices[nearby_exempler[i][0]] = cluster_centers_indices[nearby_exempler[i][1]]
    for j in range(len(documentsforRename)):
        new_labels[documentsforRename[j]] = labels[cluster_centers_indices[nearby_exempler[i][1]]]

new_cluster_indices_list = list(set(new_cluster_indices))

new_labels_list = list(set(new_labels))

doc_newclusters = {i: np.where(new_labels == i)[0] for i in new_labels_list}

for i in g.node:
    g.node[i]['label'] = new_labels[inv_dict[i]]

# tareek = []
# for i in doc_newclusters:
#     cluster_date = []
#     for j in doc_newclusters[i]:
#         cluster_date.append(g.node[dictionary[j]]['date'])
#     tareek.append(cluster_date)
# #sorted_tareek = sorted(list(set(tareek)))

# count = 0
# dict_newcluster_list = {}
# for i in doc_newclusters:
#     dict_newcluster_list[i] = count
#     count += 1
#dict_newcluster_list = {i:j for j,i in dict_newcluster_list.items()}

#cluster_sortdate_mapping = []

# for i in doc_newclusters:
#     cluster_tareek = tareek[dict_newcluster_list[i]]
#     sort_cluster_tareek = sorted(set(tareek[dict_newcluster_list[i]]))
#     #datewise_dict = {}
#     for j in doc_newclusters[i]:
#         for k in range(len(sort_cluster_tareek)):
#             if(g.node[dictionary[j]]['date']==sort_cluster_tareek[k]):
#                 g.node[dictionary[j]]['withincluster_sortdate_label'] = k

# for i in doc_newclusters:
#     j = 0
#     while(j+1 < len(doc_newclusters[i])):
#         for k in range(j+1,len(doc_newclusters[i])):
#             if(g.node[dictionary[doc_newclusters[i][j]]]['withincluster_sortdate_label']==g.node[dictionary[doc_newclusters[i][k]]]['withincluster_sortdate_label']):
#                 g.add_edge(dictionary[doc_newclusters[i][j]],dictionary[doc_newclusters[i][k]])
#                 g.add_edge(dictionary[doc_newclusters[i][k]],dictionary[doc_newclusters[i][j]])
#             else:
#                 if(g.node[dictionary[doc_newclusters[i][j]]]['withincluster_sortdate_label']<g.node[dictionary[doc_newclusters[i][k]]]['withincluster_sortdate_label']):
#                     g.add_edge(dictionary[doc_newclusters[i][j]],dictionary[doc_newclusters[i][k]])
#                 else:
#                     g.add_edge(dictionary[doc_newclusters[i][k]],dictionary[doc_newclusters[i][j]])
#         j = j+1

#print(doc_newclusters)


# In[ ]:


# for imp in doc_newclusters:
#     cluster_tareek = tareek[dict_newcluster_list[imp]]
#     sort_cluster_tareek = sorted(set(tareek[dict_newcluster_list[imp]]))
#     start_node = []
#     end_node = []
#     for j in doc_newclusters[imp]:
#         if(g.node[dictionary[j]]['withincluster_sortdate_label'] == 0):
#             start_node.append(j)
#         if(g.node[dictionary[j]]['withincluster_sortdate_label'] == len(sort_cluster_tareek)-1):
#             end_node.append(j)


#     for k in range(len(start_node)):
#         #pattern_count = 0
#         for m in range(len(end_node)):
#             #print(k,m)
#             paths = list(nx.all_simple_paths(g, source = dictionary[start_node[k]], target = dictionary[end_node[m]]))
#             for n in range(len(paths)):
#                 keywords_list = []
#                 for q in range(len(paths[n])):
#                     if(len(g.node[paths[n][q]]['keyword']) == 10):
#                         keywords_list.append(list(itertools.chain(g.node[paths[n][q]]['keyword'], [g.node[paths[n][q]]['file_name']])))
#                     else:
#                         diff = 10 - len(g.node[paths[n][q]]['keyword'])
#                         for p in range(diff):
#                             g.node[paths[n][q]]['keyword'].append('nan')
#                         keywords_list.append(list(itertools.chain(g.node[paths[n][q]]['keyword'], [g.node[paths[n][q]]['file_name']])))

#             keyword_pattern_df = pd.DataFrame()

#             for s in range(len(keywords_list)):
#                 keyword_pattern_df = keyword_pattern_df.append(pd.Series(keywords_list[s], index=['k1','k2','k3','k4','k5','k6','k7','k8','k9','k10','file_name']), ignore_index=True)                      

#             keyword_pattern_df.to_csv('/Users/AR/Desktop/naturescall/TOI_Data/pattern_csv/'+str(sort_cluster_tareek[0])+'_path_'+str(sort_cluster_tareek[-1])+'_'+str(pattern_count)+'.csv', sep = ',')                 
#             #pattern_count += 1


# In[ ]:


sab_tareek = []

for node in g.node:
    sab_tareek.append(g.node[node]['date'])
    
sort_tareek = sorted(set(sab_tareek))

for i in g.node:   
    for k in range(len(sort_tareek)):
        if(g.node[i]['date']==sort_tareek[k]):
            g.node[i]['sortdate_label'] = k
        else:
            continue


# In[ ]:


sorted_nodes = sorted(list(g.node))

for i in range(len(g.node)):
    while(i+1 < len(g.node)):
        if(g.node[sorted_nodes[i]]['sortdate_label']==g.node[sorted_nodes[i+1]]['sortdate_label']):
            g.add_edge(i,j)
            g.add_edge(j,i)
        else:
            if(g.node[sorted_nodes[i]]['sortdate_label']<g.node[sorted_nodes[i+1]]['sortdate_label']):
                g.add_edge(i,j)
            else:
                g.add_edge(j,i)


# In[ ]:


for imp in len(g.node):
#     cluster_tareek = tareek[dict_newcluster_list[imp]]
#     sort_cluster_tareek = sorted(set(tareek[dict_newcluster_list[imp]]))

    start_node = []
    end_node = []

    if(g.node[imp]['sortdate_label'] == 0):
        start_node.append(imp)
    if(g.node[imp]['sortdate_label'] == sort_tareek[-1]):
        end_node.append(imp)
                
for k in range(len(start_node)):
    pattern = 0
    for m in range(len(end_node)):
        paths = list(nx.all_simple_paths(g, source = start_node[k], target = end_node[m]))

        for n in range(len(paths)):
            keywords_list = []
            for q in range(len(paths[n])):
                if(len(g.node[paths[n][q]]['keyword']) == 10):
                    keywords_list.append(list(itertools.chain(g.node[paths[n][q]]['keyword'], [g.node[paths[n][q]]['file_name']])))
                else:
                    diff = 10 - len(g.node[paths[n][q]]['keyword'])
                    for p in range(diff):
                        g.node[paths[n][q]]['keyword'].append('nan')
                    keywords_list.append(list(itertools.chain(g.node[paths[n][q]]['keyword'], [g.node[paths[n][q]]['file_name']])))

        keyword_pattern_df = pd.DataFrame()

        for s in range(len(keywords_list)):
            keyword_pattern_df = keyword_pattern_df.append(pd.Series(keywords_list[s], index=['k1','k2','k3','k4','k5','k6','k7','k8','k9','k10','file_name']), ignore_index=True)                      

        keyword_pattern_df.to_csv('/Users/AR/Desktop/naturescall/TOI_Data/pattern_csv/'+str(g.node[start_node[k]]['date'])+'_path_'+str(g.node[end_node[m]]['date'])+'_'+str(pattern_count)+'.csv', sep = ',')                 
        pattern_count += 1

