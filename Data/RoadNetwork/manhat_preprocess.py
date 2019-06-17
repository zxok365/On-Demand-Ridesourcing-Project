# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 10:59:47 2018

@author: NLESM
"""
import pandas as pd
import numpy as np
import networkx as nx
import io

from math import sin, cos, sqrt, atan2, radians
from geopandas import read_file
from shapely.geometry import Point

import requests
#------------------------------------------------------------------------------
'''Geographical distance'''

def distance(loc1,loc2):
    R = 6373.0
    
    lat1 = radians(list(loc1)[0])
    lon1 = radians(list(loc1)[1])
    lat2 = radians(list(loc2)[0])
    lon2 = radians(list(loc2)[1])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c

#------------------------------------------------------------------------------
'''Region indexing based on NYC Boroughs'''
manhat_map = read_file('D:\\RIDESHARING\\NIPS19\\DataPreprocess\\Manhattan_shapely\\Manhattan_clean.shp')
manhat_map.index=range(1,18)

def Region(loc, n = 17):
    
    global manhat_map
    
    pnt = Point(loc)
    j = 1
    
    while manhat_map['geometry'].loc[j].contains(pnt) == 0:
        j += 1
        if j > n:
            break
            
    if j <= n:
        return int(j)
    else:
        return float('NaN')

#------------------------------------------------------------------------------
'''Create manhat_point.pkl'''

url = "https://raw.githubusercontent.com/zxok365/RidsharingProject/master/Data/RoadPoint.csv?token=Agkfs9kobeFLiB4-oxe1X3Wux76MD0m_ks5axaXuwA%3D%3D"
s = requests.get(url).content
manhat_point = pd.read_csv(io.StringIO(s.decode('utf-8')), header=0, dtype='object')

manhat_point.loc[:,"Latitude"] = manhat_point.loc[:,"Latitude"].astype(float)
manhat_point.loc[:,"Longitude"] = manhat_point.loc[:,"Longitude"].astype(float)
manhat_point.loc[:,"Index"] = manhat_point.loc[:,"Index"].astype(int)

manhat_point.index = np.array(manhat_point.loc[:,"Index"])

#Adding in coordinates
Coor1 = [tuple([x,y]) for x,y in zip(manhat_point["Longitude"], manhat_point["Latitude"])]
manhat_point["Coordinate"] = Coor1

#Removal of coordinates outside Manhattan Boroughs (NaN)
region = [Region(loc) for loc in manhat_point.loc[:,"Coordinate"]]
manhat_point["Region"] = region
manhat_point = manhat_point.dropna(subset = ["Region"])
manhat_point.loc[:,"Region"] = manhat_point.loc[:,"Region"].astype(int)

manhat_point = manhat_point.drop(["Index","Latitude","Longitude"], axis=1)
#---------------------------------------------------------------------------------
'''Create manhat_edge.pkl'''

url="https://raw.githubusercontent.com/zxok365/RidsharingProject/master/Data/RoadEdge.csv?token=Agkfs85LBJZs9VjsgHianFTe5MBE2k3uks5avbSZwA%3D%3D"
s=requests.get(url).content
manhat_edge = pd.read_csv(io.StringIO(s.decode('utf-8')), header=0, dtype='object')

manhat_edge.loc[:,"PointLongitude1"] = manhat_edge.loc[:,"PointLongitude1"].astype(float)
manhat_edge.loc[:,"PointLongitude2"] = manhat_edge.loc[:,"PointLongitude2"].astype(float)
manhat_edge.loc[:,"PointLatitude1"] = manhat_edge.loc[:,"PointLatitude1"].astype(float)
manhat_edge.loc[:,"PointLatitude2"] = manhat_edge.loc[:,"PointLatitude2"].astype(float)
manhat_edge.loc[:,"PointIndex1"] = manhat_edge.loc[:,"PointIndex1"].astype(int)
manhat_edge.loc[:,"PointIndex2"] = manhat_edge.loc[:,"PointIndex2"].astype(int)
manhat_edge.loc[:,"EdgeIndex"] = manhat_edge.loc[:,"EdgeIndex"].astype(int)
manhat_edge.index = np.array(manhat_edge.loc[:,'EdgeIndex'])

#Adding in Coordinates of endpoints
Coor1 = [tuple([x,y]) for x,y in zip(manhat_edge["PointLongitude1"],manhat_edge["PointLatitude1"])]
manhat_edge["Coordinate1"] = Coor1
Coor2 = [tuple([x,y]) for x,y in zip(manhat_edge["PointLongitude2"],manhat_edge["PointLatitude2"])]
manhat_edge["Coordinate2"] = Coor2

#Checking if PointIndexes are in manhat_point
check = np.array([(x in manhat_point.index) and (y in manhat_point.index) for (x,y) in zip(manhat_edge["PointIndex1"],manhat_edge["PointIndex2"])])
manhat_edge = manhat_edge[check == 1]

#Checking if there are paths between the endpoints
manhat_graph = nx.DiGraph()

for v in manhat_point.index:
    manhat_graph.add_node(v)

for e in manhat_edge.index:
    from_node = manhat_edge.loc[e,'PointIndex1']
    to_node = manhat_edge.loc[e,'PointIndex2']
    
    manhat_graph.add_edge(from_node, to_node)

check = np.array([nx.has_path(manhat_graph, o, d) for o,d in zip(manhat_edge['PointIndex1'],manhat_edge['PointIndex2'])])
manhat_edge = manhat_edge[check == 1]

#Adding in geographical distance
manhat_edge['Length'] = [distance(loc1,loc2) for loc1,loc2 in zip(manhat_edge["Coordinate1"],manhat_edge["Coordinate2"])]

manhat_edge = manhat_edge.drop(["EdgeIndex","PointLatitude1","PointLatitude2","PointLongitude1","PointLongitude2"], axis=1)

manhat_edge.to_pickle('D:\\RIDESHARING\\NIPS19\\DataPreprocess\\RoadNetwork\\manhat_edge.pkl')

#------------------------------------------------------------------------------
from Waypoint import WayPoint, MapSystem

'''Construct network'''
TMP_msg = manhat_edge.values
NodePos = {}
NearbyNode = {} #Neighbors

for rd_msg in TMP_msg:
    p = int(rd_msg[0])
    q = int(rd_msg[1])
    NodePos[p] = rd_msg[2]
    NodePos[q] = rd_msg[3]

    if NearbyNode.get(p) == None:
        NearbyNode[p] = [q]
    else:
        NearbyNode[p] += [q]

mapsystem = MapSystem()
mapsystem._node_dict = NodePos
mapsystem._nearby_node = NearbyNode

'''Update manhat_point with Neighbors column (-used in singlebatch experiment)'''

node_in_areas = {}

for n1 in set(manhat_edge['PointIndex1']):
    if n1 not in node_in_areas:
        node_in_areas[n1] = []
    
    for n2 in manhat_point.index:
        if mapsystem.distance(WayPoint(n2, n2, 0, 0), WayPoint(n1, n1, 0, 0)) <= 210:
            node_in_areas[n1] += [n2]

length = []
for n1 in node_in_areas.keys():
    length += [len(node_in_areas[n1])]
    
manhat_point['ExtendedNeighbors_210s_1'] = [node_in_areas[i] for i in manhat_point.index]

manhat_point.to_pickle('D:\\RIDESHARING\\NIPS19\\DataPreprocess\\RoadNetwork\\manhat_point.pkl')