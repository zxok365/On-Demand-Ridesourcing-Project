import pandas as pd
import numpy as np
import datetime
from math import sin, cos, sqrt, atan2, radians

from Waypoint import WayPoint, MapSystem

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

'''Network distance'''

manhat_point = pd.read_pickle('D:\\RIDESHARING\\NIPS19\\DataPreprocess\\RoadNetwork\\manhat_point.pkl')
manhat_edge = pd.read_pickle('D:\\RIDESHARING\\NIPS19\\DataPreprocess\\RoadNetwork\\manhat_edge.pkl')

def closest_node(loc):

    global manhat_point
    
    loc = np.array([loc])
    nodes = np.array(list(manhat_point['Coordinate']))
    dist_2 = np.sum((nodes - loc)**2, axis=1)
    pos = np.argmin(dist_2)
    
    return manhat_point.index[pos]

TMP_msg = manhat_edge.values
NodePos = {}
NearbyNode = {}

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

#------------------------------------------------------------------------------
'''Filter only Sunday, 5 May 2013, 1700 - 1900 & Data Cleaning'''

taxi_csv1 = pd.read_pickle('D:\\RIDESHARING\\NIPS19\\DataPreprocess\\RequestGenerator\\taxi_csv5.pkl') #search counter 5 May

'''Filtering dates & hours'''
for day in range(1, 31):
    taxi_csv = taxi_csv1[taxi_csv1['bydates'] == datetime.date(2013,5,day)]

taxi_csv_17 = taxi_csv1[taxi_csv1['byhours'] == 17]
taxi_csv_18 = taxi_csv1[taxi_csv1['byhours'] == 18]
taxi_csv = pd.concat([taxi_csv_17, taxi_csv_18])

'''Adding in arrival time in seconds'''
seconds = [pd.Timestamp(x,tz=None).second for x in taxi_csv[' pickup_datetime']]
taxi_csv['bysecs'] = seconds
taxi_csv = taxi_csv.drop([' pickup_datetime',' dropoff_datetime'], axis = 1)

arr_time = [h*3600+m*60+s for h,m,s in zip(taxi_csv['byhours'],taxi_csv['bymins'],
                                           taxi_csv['bysecs'])]
taxi_csv['arrival_time'] = arr_time

taxi_csv = taxi_csv.drop(['byhours','bymins','bysecs'], axis = 1)

'''Checking trip validity'''
#1. Remove identical pickup & dropoff locations
taxi_csv = taxi_csv[taxi_csv['pickup_coordinates'] != taxi_csv['dropoff_coordinates']]

#2. Remove of missing and incorrect (negative) coordinates
pu_coor = list(taxi_csv['pickup_coordinates'])
x_list, y_list = zip(*pu_coor)
x_list = np.array(x_list)
taxi_csv = taxi_csv[np.isnan(x_list) == False]
print(taxi_csv.shape)

pu_coor = list(taxi_csv['pickup_coordinates'])
x_list, y_list = zip(*pu_coor)
x_list = np.array(x_list)
taxi_csv = taxi_csv[x_list < 0]
print(taxi_csv.shape)

pu_coor = list(taxi_csv['pickup_coordinates'])
x_list, y_list = zip(*pu_coor)
y_list = np.array(y_list)
taxi_csv = taxi_csv[np.isnan(y_list) == False]
print(taxi_csv.shape)

pu_coor = list(taxi_csv['pickup_coordinates'])
x_list, y_list = zip(*pu_coor)
y_list = np.array(y_list)
taxi_csv= taxi_csv[y_list > 0]
print(taxi_csv.shape)

do_coor = list(taxi_csv['dropoff_coordinates'])
x_list, y_list = zip(*do_coor)
x_list = np.array(x_list)
taxi_csv = taxi_csv[np.isnan(x_list) == False]
print(taxi_csv.shape)

do_coor = list(taxi_csv['dropoff_coordinates'])
x_list, y_list = zip(*do_coor)
x_list = np.array(x_list)
taxi_csv = taxi_csv[x_list < 0]
print(taxi_csv.shape)

do_coor = list(taxi_csv['dropoff_coordinates'])
x_list, y_list = zip(*do_coor)
y_list = np.array(y_list)
taxi_csv = taxi_csv[np.isnan(y_list) == False]
print(taxi_csv.shape)

do_coor = list(taxi_csv['dropoff_coordinates'])
x_list, y_list = zip(*do_coor)
y_list = np.array(y_list)
taxi_csv= taxi_csv[y_list > 0]
print(taxi_csv.shape)

#3. Remove trip outliers (with approx. travel time < 1 mins or > 1 hour)
velo = 45/3600 #km/sec

tt = [distance(loc1, loc2)/velo for loc1, loc2 in zip(taxi_csv['pickup_coordinates'],
      taxi_csv['dropoff_coordinates'])]
taxi_csv = taxi_csv[np.array(tt) >= 80]
print(taxi_csv.shape)

tt = [distance(loc1, loc2)/velo for loc1, loc2 in zip(taxi_csv['pickup_coordinates'],
      taxi_csv['dropoff_coordinates'])]
taxi_csv = taxi_csv[np.array(tt) <= 3600]

#------------------------------------------------------------------------------
'''Adding in pickup & dropoff nodes, trip_length'''
pu_nodes = []
for i in range(len(taxi_csv.index)):
    pu_nodes += [closest_node(taxi_csv['pickup_coordinates'].iloc[i])]

taxi_csv['pickup_nodes'] = pu_nodes

do_nodes = []
for i in range(len(taxi_csv.index)):
    do_nodes += [closest_node(taxi_csv['dropoff_coordinates'].iloc[i])]

taxi_csv['dropoff_nodes'] = do_nodes

'''Adding in trip_length'''
trip_length = []
    
for i in range(len(taxi_csv.index)):
    o_waypoint = WayPoint(pu_nodes[i], pu_nodes[i], 0, 0)
    d_waypoint = WayPoint(do_nodes[i], do_nodes[i], 0, 0)
    trip_length += [mapsystem.distance(o_waypoint, d_waypoint)]

'''Categorize the trip length into bins of 60s'''
bin_60 = [int((x - 30)//60) for x in trip_length]
for i in range(len(bin_60)):
    if bin_60[i] < 2:
        bin_60[i] = -1
    else:
        bin_60[i] -= 2      

taxi_csv.to_pickle('D:\\RIDESHARING\\NIPS19\\DataPreprocess\\RequestGenerator\\May_2013_HOUR17.pkl')

'''Filter the request data (first 30s, within capacity of 4, trip length >= 400 secs)'''
valid_data = taxi_csv
valid_data = valid_data[valid_data['arrival_time'] >= 61200]
valid_data = valid_data[valid_data['arrival_time'] < 61230]
valid_data = valid_data[valid_data[' passenger_count'] <= 4]
valid_data = valid_data[valid_data['trip_length'] >= 400]

valid_data.to_pickle('D:\\RIDESHARING\\NIPS19\\DataPreprocess\\RequestGenerator\\May_2013_HOUR17_singlebatch.pkl')