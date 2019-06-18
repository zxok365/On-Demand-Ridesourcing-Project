import csv
import numpy as np
import pandas as pd
import random
import time
import datetime

import bipartite3_algo
from request import *
from vehicle import *
from Waypoint import WayPoint, MapSystem



'''Parameter Setting'''
eps = 1e-6
cardinal_const = 1000000

'''Simulation Setup'''
period_length = 30 #i.e. assignment is done each 30-secs
H = 1 #i.e. number of periods
CAPACITY = 4 #i.e. maximum capacity of vehicles
MAX_WAITING_TIME = 240 #i.e. maximum waiting time for request before picked up
c = 1 #i.e. weightage for trip benefits (active time) compared to the loss (cruising time)

REMOVAL = 500 #i.e. set the allowance for trip removal to manage the network sparsity problems
ratio = 1.2
divisor = 1 #i.e. ratio, divisor to set the balance between HIGH and LOW activity vehicles
seed = 0
day = 6
HOUR = 17
filter_treshold = 10 #to gauge the assignment potential based on reachability of vehicle initial locations (nodes)
w = 0 #{0 - efficient allocation, 1 - fair allocation}

DATA_TYPE = 'WHOLE' #i.e. {'WHOLE': whole Manhattan area, 'REGION': parts of the map}
if DATA_TYPE == 'WHOLE':
    MIN_LIST = [-float("inf")]
    MAX_LIST = [float("inf")]
else:
    MIN_LIST = [-float("inf")] + list(np.linspace(40.72, 40.74, 3)) + [40.75]
    MAX_LIST = [40.75] + list(np.linspace(40.72, 40.74, 3) + .035) + [float("inf")]


#------------------------------------------------------------------------------
'''Generate the road network'''

manhat_point = pd.read_pickle('..\\..\\Data\\RoadNetwork\\manhat_point.pkl')
manhat_edge = pd.read_pickle('..\\..\\Data\\RoadNetwork\\manhat_edge.pkl')

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

#------------------------------------------------------------------------------
'''Generate the inter-node travel time'''

tt_dict = {}
node_dict = {}
tt_list = []

traveltime = pd.read_csv('..\\..\\Data\\TravelTimePrediction\\TravelTime\\time_cost' + str(HOUR) + '.csv', index_col=[0, 1], header=None)
traveltime.columns = [HOUR]
tmp_list = list(traveltime[HOUR])
tot_num = 0

for n1, n2 in zip(traveltime.index.get_level_values(0), traveltime.index.get_level_values(1)):
    
    tt_dict[(n1, n2)] = tmp_list[tot_num]
    node_dict[(n1,n2)] = tot_num
    tot_num += 1        

mapsystem.update_distance(cur_time = 0, distance = tt_dict) #i.e. distance = travel time (in secs)

tt_list += [list(tt_dict.values())]

#------------------------------------------------------------------------------
'''Reading Request Data'''

REQ_DATA = pd.read_pickle('..\..\\Data\\RequestGenerator\\May_2013_HOUR17_singlebatch.pkl')
#------------------------------------------------------------------------------
'''Function lists:
    1. closest_node - convert geocoordinates to nearest point index in the road network
    2. status_update - update vehicles and requests' attributes & some arrays 
    for route recording (see l.116-124, 133-140)
    3. compute_routing - take in available taxis and coming requests & output 
    optimal (dependent on specified efficiency-fairness conditions) routes'''

def closest_node(loc):

    global manhat_point
    
    loc = np.array([loc])
    nodes = np.array(list(manhat_point['Coordinate']))
    dist_2 = np.sum((nodes - loc)**2, axis=1)
    pos = np.argmin(dist_2)
    
    return manhat_point.index[pos]

def status_update(t, V_STORAGE, FIN_ALLOC, STACKED_ROUTE, STACKED_TIME):
    
    for vid in FIN_ALLOC:
        
        cur_v = V_STORAGE[vid]
        destination_list = FIN_ALLOC[vid] #i.e. the assigned routing for vehicle with ID vid
        moving_time = t - cur_v._overall_time #i.e. variable for 'remaining time'; starting with 30 secs (period length)
        prev_moving_time = moving_time
                    
        for (rid, p_or_d) in destination_list: 
            #i.e. rid: request ID, p_or_d: 0-pickup/1-dropoff
            
            cur_r = REQ_STORAGE[rid]
                
            if moving_time < 0:
                break
        
            if p_or_d == 0:
                
                reached, moving_time = mapsystem.move_to_destination(cur_v._waypoint, cur_r._origin_waypoint, moving_time)
                #i.e. reached: binary variable; indication if vehicle cur_v has reached the specified node in the destination_list,
                #moving_time: reduced by the traveltime from cur_v's 'current' position (cur_v._waypoint)
                #to the specified PICKUP node (cur_r._origin_waypoint) in destination_list
                
                if len(cur_v._boarded_requests) != 0:
                    cur_v._active_timecost += prev_moving_time - moving_time
                    cur_r.update_trip_value(0, moving_time - prev_moving_time)
                    
                if reached:
                    cur_v.picking_up(cur_r)                    
                    cur_r.picking_up(t - moving_time)
                    STACKED_ROUTE[vid] += [(rid, p_or_d)]
                    STACKED_TIME[vid] += [t - moving_time]
                
                else:
                    break
    
            if p_or_d == 1:
                reached, moving_time = mapsystem.move_to_destination(cur_v._waypoint,cur_r._destination_waypoint, moving_time)
                #moving_time: see l.113-114, with destination the specified DROPOFF node (cur_r._destination_waypoint)

                cur_v._active_timecost += prev_moving_time - moving_time
                cur_r.update_trip_value(0, moving_time - prev_moving_time)
                    
                if reached:
                    cur_v.finishing(cur_r)
                    cur_r.finishing(t - moving_time)
                    STACKED_ROUTE[vid] += [(rid, p_or_d)]
                    STACKED_TIME[vid] += [t - moving_time]
                
                else:
                    break
            
            prev_moving_time = moving_time
            
def compute_routing(h, t, act_corr, V_STORAGE, REQ_STORAGE, DEMAND_LIST, FIN_ALLOC, switch = 0):

    global req_pos
    
    print('req_pos bf:', req_pos)
    
    for i in range(req_pos, len(REQ_STORAGE)):
        r = REQ_STORAGE[i]
        
        if r._request_time >= h*period_length:
            break
        
        DEMAND_LIST += [r]
        req_pos += 1
    
    print('req_pos aft:', req_pos)    

    TAXI_LIST = [v for v in V_STORAGE if v._arr_time < t]
    
    '''Create bipartite graph and other information to be passed into the main algorithm'''
    RTV_graph = {} #i.e. built based on constraints specified below (see l.211-212, 218-219)
    All_edges = []
    Sorting_weight = [] #i.e. for fairness; needs to account for historical values accumulated by each vehicle
    Value_weight = [] #i.e. for efficiency; needs to capture the maximization of service rate (see l.198 adding in large constant), 
    #wlog it disregards history
    
    NEW_TAXI_LIST = []
    NEW_DEMAND_LIST = []
    index = -1
    
    for v in TAXI_LIST:
        index += 1
        
        '''There are boarded requests, direct addition to FIN_ALLOC'''
        if len(v._boarded_requests) > 0:
            r = v._boarded_requests[0]
            RTV_graph[v._vid] = {r._rid: [r._trip_value, r._trip_active, 0]}            
            FIN_ALLOC[v._vid] = ((r._rid, 1),)
            continue
        
        '''Once assigned, it commits to the assignment, direct addition to FIN_ALLOC'''
        if len(v._assigned_requests) > 0:            
            r = v._assigned_requests[0]
            RTV_graph[v._vid] = {r._rid: [r._trip_value, r._trip_active, 0]}
            FIN_ALLOC[v._vid] = ((r._rid, 0), (r._rid, 1))
            continue
        
        NEW_TAXI_LIST += [v]
        
        '''Adding in empty trip'''
        RTV_graph[v._vid] = {-v._vid - 1: [0, 0, 0]}
        All_edges += [[v._vid, -v._vid - 1]]
        
        Sorting_weight += [v._allocated_value1 + 0]
        Value_weight += [0  + 100000000]
        
        for r in DEMAND_LIST:
            
            if r._assigned == True or r._size > CAPACITY:
                continue
            
            NEW_DEMAND_LIST += [r._rid]
            
            pickup_duration = mapsystem.distance(v._waypoint, r._origin_waypoint)
            waiting_time = (t - r._request_time) + pickup_duration
            
            if r.latest_acceptable_pickup_time <= t + pickup_duration:
                continue
            
            else:
                trip_active = r._trip_length
                trip_value = trip_active - waiting_time
                
                if trip_value < 0 :
                    continue
                    
                RTV_graph[v._vid][r._rid] = [trip_value, trip_active, pickup_duration]
                All_edges += [[v._vid, r._rid]]
                
                Sorting_weight += [v._allocated_value1 + c*trip_active - pickup_duration]
                Value_weight += [c*trip_active + 100000000 - pickup_duration]
    
    print('NEW_DEMAND_LIST, len:', len(set(NEW_DEMAND_LIST)))
    
    '''Recording purposes'''
    hist_act = [v._active_timecost for v in TAXI_LIST]
    hist_val = [v._allocated_value1 for v in TAXI_LIST]
    min_vid = hist_val.index(min(hist_val))
    
    print('Active time minmax:', [min(hist_act), max(hist_act)])
    print('activity value minmax:', [min(hist_val), max(hist_val)])
    print('vid min?', min_vid)
            
    print('NUM OF AVAILABLE TAXIS:', len(NEW_TAXI_LIST))
    print('NUM OF REQUESTS:', len(DEMAND_LIST))
    
    '''Solving for optimal route'''
    if len(NEW_TAXI_LIST) > 0:
        route_dict = bipartite3_algo.solve_rtv_graph(h, H, RTV_graph, NEW_TAXI_LIST, DEMAND_LIST, All_edges, Sorting_weight, Value_weight, act_corr, c)
    else:
        route_dict = {}
    
    act_list = [v._allocated_value1 for v in V_STORAGE]
    min_driver = min(act_list)
    ilp_discr = np.std(act_list)
    value_sum = sum(act_list)
    
    unassigned_supply = len([vid for vid in route_dict if list(route_dict[vid]) == []])
                    
    idle_taxis = []
    unassigned_supply = []
    
    '''Storing the optimal routes (i.e. route_dict) into FIN_ALLOC & updating of vehicle/request attributes'''
    for vid in route_dict:
        if list(route_dict[vid]) == []:
            unassigned_supply += [vid]
            
        FIN_ALLOC[vid] = route_dict[vid]
        cur_v = V_STORAGE[vid]
        cur_v.new_assigning()
        
        for (rid, p_or_d) in FIN_ALLOC[vid]:
            cur_r = REQ_STORAGE[rid]
            if p_or_d == 0:
                cur_r.assigning(vid,t)
                cur_v.assigning(cur_r)
                
        if cur_v.isidle:
            idle_taxis += [cur_v]
        
    for vid in FIN_ALLOC:
        V_STORAGE[vid]._overall_time = t
        
    print('------------------------------------------------------------------')
    print('UNASSIGNED TAXIS',len(unassigned_supply))
    print('IDLE TAXIS', len(idle_taxis))
            
    return value_sum, ilp_discr, min_driver


#------------------------------------------------------------------------------
'''STARTING SIMULATION'''

print('DAY:', day)
print('TRESH:', filter_treshold)
lat_min, lat_max = list(zip(MIN_LIST, MAX_LIST))[0]
    
'''Filter the request data (first 30s, within capacity, trip length >= 400 secs)'''
valid_data = REQ_DATA[REQ_DATA['bydates'] == datetime.date(2013,5,day)]

ORI_REQ_NUM = len(valid_data)
print('NUMREQ from original data:', ORI_REQ_NUM)

chosen_tripid = [valid_data.index[i] for i in range(len(valid_data.index))]

'''Generate artificial vehicle locations'''
chosen_nodes = []
for tid in chosen_tripid:
    chosen_nodes += [valid_data.loc[tid, 'pickup_nodes']]

print('ORIGINAL TRIP NODES:', chosen_nodes)
print('how large the duplicates? (len vs lenset):', (len(chosen_nodes), len(set(chosen_nodes))))

rand_num = np.random.uniform(0,1,len(chosen_nodes))
v_location_ACT = [] #i.e. to initialize locations of HIGH activity vehicles - extremize the environment: exactly on the node of chosen requests
for i in range(len(rand_num)):
    if rand_num[i] <= divisor:
        v_location_ACT += [manhat_point.loc[chosen_nodes[i], 'Coordinate']]

TAXI_TOTAL_NUM = int(ratio/divisor*len(v_location_ACT)) #i.e. set the ratio of high vs low activity vehicles

node_in_areas = {} #i.e. nodes that are within 210s travelling time
for n1 in manhat_point.index:
    node_in_areas[n1] = manhat_point.loc[n1, 'ExtendedNeighbors_210s_1']

v_init_nodes = []
for node in chosen_nodes:
    v_init_nodes += node_in_areas[node]
    
filter_init = [] #i.e. store only the nodes with 'sufficient' reachability potential
for vloc in set(v_init_nodes):
    potential = sum([(vloc in node_in_areas[n1]) for n1 in chosen_nodes])
    
    if potential <= filter_treshold:
        continue
    
    filter_init += [vloc]

print('len v_init:', len(set(v_init_nodes)))
print('HOW LARGE?', len(filter_init))
print('REQUIRED?', TAXI_TOTAL_NUM - len(v_location_ACT))
    
random.seed(seed)
loc_id = np.random.choice(filter_init, TAXI_TOTAL_NUM - len(v_location_ACT), replace = False)
v_location = [manhat_point.loc[filter_init[i], 'Coordinate'] for i in range(len(loc_id))] #i.e. to initialize locations for LOW activity vehicles

'''Reset V_STORAGE'''
V_STORAGE = np.array([])

for loc in v_location_ACT:
    vid = len(V_STORAGE)
    newv = Vehicle(vid, loc, CAPACITY)
    
    newv._point_id = closest_node(newv._cur_position)
    newv._waypoint = WayPoint(newv._point_id, newv._point_id, 0, 0)
    newv._arr_time = 0
    
    random.seed(vid + 1 + seed)
    newv._allocated_value1 = random.uniform(200, 400) #i.e. set the HIGH historical active time
    
    V_STORAGE = np.append(V_STORAGE, newv)
            
for loc in v_location:
    vid = len(V_STORAGE)
    newv = Vehicle(vid, loc, CAPACITY)
    newv._point_id = closest_node(newv._cur_position)
    newv._waypoint = WayPoint(newv._point_id, newv._point_id, 0, 0)
    
    newv._arr_time = 0
    
    random.seed(vid + 1 + seed)
    newv._allocated_value1 = random.uniform(50, 100) #i.e. set the LOW historical active time
    
    V_STORAGE = np.append(V_STORAGE, newv)             

print('NUM OF TAXIS:', len(V_STORAGE))

'''Reset REQ_STORAGE'''
REQ_STORAGE=np.array([])

for i in chosen_tripid:
    region = 'None'
    origin = valid_data.loc[i, 'pickup_coordinates']
    destination = valid_data.loc[i, 'dropoff_coordinates']
    trip_length = valid_data.loc[i, 'trip_length']
    size = valid_data.loc[i, ' passenger_count']
    arr_time = 0
    req_ID = len(REQ_STORAGE)
    
    newreq = Request(req_ID, region, origin, destination, arr_time, size, trip_length)
    
    p = valid_data.loc[i, 'pickup_nodes']
    q = valid_data.loc[i, 'dropoff_nodes']
    newreq._origin_closest_node = p
    newreq._destination_closest_node = q
    newreq._origin_waypoint = WayPoint(p, p, 0., 0.)
    newreq._destination_waypoint = WayPoint(q, q, 0., 0.)
    newreq._trip_length = mapsystem.distance(newreq._origin_waypoint, newreq._destination_waypoint)
    newreq._max_waiting_time = MAX_WAITING_TIME
        
    REQ_STORAGE = np.append(REQ_STORAGE, newreq)

REQ_NUM = len(REQ_STORAGE)
print('NUM OF REQS:', REQ_NUM)
removal_rate = REQ_NUM/ORI_REQ_NUM

'''Starting Simulation'''
DEMAND_LIST = []
STACKED_ROUTE = [[] for i in range(TAXI_TOTAL_NUM)]
STACKED_TIME = [[] for i in range(TAXI_TOTAL_NUM)]

req_pos = 0          
weight_trial = [w for i in range(H)]

print('WEIGHT:', w)
weight_type = w

switch = 0
for h in range(1, H+1):
   
    print('PERIOD', h)
    print('===========================================')

    act_corr = weight_trial[h-1]
    t = h*period_length
    FIN_ALLOC = {}
    
    DEMAND_LIST = [r for r in DEMAND_LIST if r._assigned == True]
    
    ILP_VALUE, ILP_DISCR, MIN_DRIVER = compute_routing(h, t, act_corr, V_STORAGE, REQ_STORAGE, DEMAND_LIST, FIN_ALLOC, switch)
    
    list_normal = [v._allocated_value1 for v in V_STORAGE]
    
    if np.min(list_normal) > 0:
        switch = 1
        
    print('normal')
    print('std:', np.std(list_normal))
    print('minmax:', (np.min(list_normal), np.max(list_normal)))
    print('----')
    
    if h < H:
        t = (h+1)*period_length
        status_update(t, V_STORAGE, FIN_ALLOC, STACKED_ROUTE, STACKED_TIME)
    
TOT_SERVED, SERVICE_RATE, MEAN_WAIT, MEAN_DELAY, MEAN_INTRIP = [0 for i in range(5)]

f = open('Tradeoff_w_Horizons.csv', 'a', newline = '')
to_append = [[DATA_TYPE, REMOVAL, removal_rate, MAX_WAITING_TIME, lat_min, HOUR, H, TAXI_TOTAL_NUM, REQ_NUM, weight_trial, 
              TOT_SERVED, SERVICE_RATE, MEAN_WAIT, MEAN_DELAY, MEAN_INTRIP,
              np.sum(list_normal), np.std(list_normal), np.min(list_normal), np.max(list_normal)]]
writer = csv.writer(f)
writer.writerows(to_append)
f.close()

'''Record list_normal as column'''
f = open('ActiveTime.csv', 'a', newline = '')
to_append = [[HOUR, weight_type] + list_normal]
writer = csv.writer(f)
writer.writerows(to_append)
f.close()
