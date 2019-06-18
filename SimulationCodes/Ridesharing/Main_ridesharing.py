import csv
import numpy as np
import pandas as pd
import random
import time
import datetime

import rideshare_algo
from vehicle import *
from request import *
import searching
import calculate_min_routing
from Waypoint import WayPoint, MapSystem

'''Parameter Setting'''
eps = 1e-6
RANDOM_SEED1 = 1997
RANDOM_SEED2 = 1993
cardinal_const = 1000000

'''Simulation Setting'''

period_length = 30 #i.e. assignment is done each 30-secs
H = 2 * 3600 // period_length #i.e. number of periods in 2-hour horizon
TAXI_TOTAL_NUM = 2000
CAPACITY = 4 #i.e. maximum capacity of vehicles
ACTIVE_CONSTANT_WEIGHT = 1 #i.e. weightage for trip benefits (active time) compared to the loss (cruising time)
MAX_WAITING_TIME = 150 #i.e. maximum waiting time for request before picked up
MAX_SERVE_NUM = 2 #i.e. maximum number of requests to be served together by 1 vehicle
REMOVAL = 500 #i.e. set the allowance for trip removal to manage the network sparsity problems

DATA_TYPE = 'WHOLE' #i.e. {'WHOLE': whole Manhattan area, 'REGION': parts of the map}
if DATA_TYPE == 'WHOLE':
    MIN_LIST = [-float("inf")]
    MAX_LIST = [float("inf")]
else:
    MIN_LIST = [40.71, 40.72, 40.73, 40.74, 40.75, 40.76, 40.77]
    MAX_LIST = [40.74, 40.75, 40.76, 40.77, 40.78, 40.79, float("inf")]

SIMULATION_DAY = [3, 29, 14, 11, 27, 25, 6, 17, 7, 8] #i.e. the date to simulate
HOUR = 17

'''Vehicle Behaviour Setting'''
RANDOM_VEHICLE_POS_START = False

'''Generate the road network'''

manhat_point = pd.read_pickle("..\\..\\Data\\RoadNetwork\\manhat_point.pkl")
manhat_edge = pd.read_pickle("..\\..\\Data\\RoadNetwork\\manhat_edge.pkl")

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

REQ_DATA = pd.read_pickle('..\\..\\Data\\RequestGenerator\\May_2013_HOUR17.pkl')

#------------------------------------------------------------------------------
'''Function lists:
    1. closest_node - convert geocoordinates to nearest point index in the road network
    2. status_update - update vehicles and requests' attributes, arrays 
    for route recording (see l.116-124, 133-140), and accumulated historical active time (see l.127)
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
    
        UpdateLabel = True
        plan_dis = 0.
        plan_pos = None
        #i.e. plan_dis & plan_pos are used to update the maximum waiting time of previously assigned requests;
        #this is based on constraint that reallocated requests should wait less than the currently assigned waiting time (see l.141)
        
        for (rid, p_or_d) in destination_list: 
            #i.e. rid: request ID, p_or_d: 0-pickup/1-dropoff
            
            cur_r = REQ_STORAGE[rid]
                
            if moving_time < 0:
                break

            if p_or_d == 0:
                
                if UpdateLabel:
                    reached, moving_time = mapsystem.move_to_destination(cur_v._waypoint, cur_r._origin_waypoint, moving_time)
                    #i.e. reached: binary variable; indication if vehicle cur_v has reached the specified node in the destination_list,
                    #moving_time: reduced by the traveltime from cur_v's 'current' position (cur_v._waypoint)
                    #to the specified PICKUP node (cur_r._origin_waypoint) in destination_list
                    
                    if len(cur_v._boarded_requests) != 0:
                        cur_v._active_timecost += prev_moving_time - moving_time
                        cur_r.update_trip_value(0, moving_time - prev_moving_time)
                        
                    else:
                        cur_v.update_allocated(moving_time - prev_moving_time) #i.e. to update 'realized' trip active time
                        
                    if reached:
                        cur_v.picking_up(cur_r)                    
                        cur_r.picking_up(t - moving_time)
                        STACKED_ROUTE[vid] += [(rid, p_or_d)]
                        STACKED_TIME[vid] += [t - moving_time]
                        
                    else:
                        plan_pos = cur_v._waypoint
                        UpdateLabel = False
                        
                if not UpdateLabel:
                    plan_dis += mapsystem.distance(plan_pos,cur_r._origin_waypoint)
                    cur_r.set_max_waiting_time(min(cur_r._max_waiting_time, plan_dis + t - cur_r._request_time))
                    plan_pos = cur_r._origin_waypoint
    
            if p_or_d == 1:
                if UpdateLabel:
                    reached, moving_time = mapsystem.move_to_destination(cur_v._waypoint,cur_r._destination_waypoint, moving_time)
                    
                    cur_v._active_timecost += prev_moving_time - moving_time
                    cur_r.update_trip_value(0, moving_time - prev_moving_time)
                    
                    if reached:
                        cur_v.finishing(cur_r)
                        cur_r.finishing(t - moving_time)
                        STACKED_ROUTE[vid] += [(rid, p_or_d)]
                        STACKED_TIME[vid] += [t - moving_time]
                        
                    else:
                        plan_pos = cur_v._waypoint
                        UpdateLabel = False
                        
                if not UpdateLabel:
                    plan_dis += mapsystem.distance(plan_pos, cur_r._destination_waypoint)
                    plan_pos = cur_r._destination_waypoint
                    
            prev_moving_time = moving_time



def compute_routing(h, t, act_corr, V_STORAGE, REQ_STORAGE, DEMAND_LIST, FIN_ALLOC):
    
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
    
    '''Building 'current' road network; recording information on used nodes for speed-up'''
    LocToOrder = {}
    wp_list = []
    NODE_DICT = {}
        
    tot_num = 0
    all_num = 0
    
    for v in TAXI_LIST:
        for r in v.boarded_requests:
            all_num += 1
    
            cur_pos = tuple(r.get_destination)
            cur_node = r._destination_closest_node
            if NODE_DICT.get(cur_node):
                xx = NODE_DICT.get(cur_node)
                LocToOrder[cur_pos] = xx
                r.set_destination_id(xx)
            else:
                NODE_DICT[cur_node] = tot_num
                LocToOrder[cur_pos] = tot_num
                r.set_destination_id(tot_num)
                wp_list += [r._destination_waypoint]
                tot_num += 1

    for r in DEMAND_LIST:
        all_num += 2
    
        cur_pos = tuple(r.get_origin)
        cur_node = r._origin_closest_node
        if NODE_DICT.get(cur_node):
            xx = NODE_DICT.get(cur_node)
            LocToOrder[cur_pos] = xx
            r.set_origin_id(xx)
        else:
            NODE_DICT[cur_node] = tot_num
            LocToOrder[cur_pos] = tot_num
            r.set_origin_id(tot_num)
            wp_list += [r._origin_waypoint]
            tot_num += 1
    
        cur_pos = tuple(r.get_destination)
        cur_node = r._destination_closest_node
        if NODE_DICT.get(cur_node):
            xx = NODE_DICT.get(cur_node)
            LocToOrder[cur_pos] = xx
            r.set_destination_id(xx)
        else:
            NODE_DICT[cur_node] = tot_num
            LocToOrder[cur_pos] = tot_num
            r.set_destination_id(tot_num)
            wp_list += [r._destination_waypoint]
            tot_num += 1

    cur_network = np.zeros(shape=[tot_num + len(TAXI_LIST), tot_num + len(TAXI_LIST)])
    for pp in range(0, tot_num):
        for qq in range(0, tot_num):
            cur_network[pp][qq] = mapsystem.distance(wp_list[pp], wp_list[qq])

    tmp_num = tot_num
    for v in TAXI_LIST:
        all_num += 1
        wp_list += [v._waypoint]
        v.set_point_id(tot_num)
        tot_num += 1

    for pp in range(tmp_num, tot_num):
        for qq in range(0, tmp_num):
            cur_network[pp][qq] = mapsystem.distance(wp_list[pp], wp_list[qq])


    '''Create bipartite graph and other information to be passed into the main algorithm'''
    RTV_graph = {} #i.e. built based on constraints specified below (see l.211-212, 218-219)
    All_edges = []
    Sorting_weight = [] #i.e. for fairness; needs to account for historical values accumulated by each vehicle
    Value_weight = [] #i.e. for efficiency; needs to capture the maximization of service rate (see l.198 adding in large constant), 
    #wlog it disregards history
    
    NEW_TAXI_LIST = []

    for v in TAXI_LIST:
        
        '''Not supporting reassignment,the following case leads to direct addition to FIN_ALLOC'''
        if len(v._boarded_requests) + len(v._assigned_requests) == MAX_SERVE_NUM:            
            r_list = v._assigned_requests
            feasible, route, min_cost, total_length, idle = calculate_min_routing.calc_min_routing(v, r_list, t, mapsystem, cur_network)
            active = ACTIVE_CONSTANT_WEIGHT * total_length - idle
            
            RTV_graph[v._vid] = {- v._vid - 1: [route, active, idle]}

            a, b, c, d = zip(*route)
            route = tuple(list(zip(b, c)))
            FIN_ALLOC[v._vid] = route
            
        else:
            NEW_TAXI_LIST += [v]
        
            '''Adding in empty trip'''
            r_list = v._assigned_requests
            feasible, route, min_cost, total_length, idle = calculate_min_routing.calc_min_routing(v, r_list, t, mapsystem, cur_network)
            active = ACTIVE_CONSTANT_WEIGHT * total_length - idle
            
            RTV_graph[v._vid] = {- v._vid - 1: [route, active, idle]}
            
            All_edges += [[v._vid, - v._vid - 1]]
            
            r_list = v._assigned_requests + []
            assigned_sum = ACTIVE_CONSTANT_WEIGHT * sum([r0._trip_length for r0 in r_list])
            
            Sorting_weight += [v._allocated_value1 + assigned_sum - idle]
            Value_weight += [cardinal_const + assigned_sum - idle]
            remaining_space = CAPACITY
            
            if len(v.boarded_requests) == 1:
                for r in v.boarded_requests:
                    remaining_space -= r._size

            '''Non-empty trips - support appending of 1 request per vehicle per round due to the nature of bipartite graph'''
            for r in DEMAND_LIST:

                if r._assigned == True:
                    continue

                if r._size > remaining_space:
                    continue

                r_list = v._assigned_requests + [r]
                feasible, route, min_cost, total_length, idle = calculate_min_routing.calc_min_routing(v, r_list, t, mapsystem, cur_network)
                active = ACTIVE_CONSTANT_WEIGHT * total_length - idle
                
                if feasible == 0 or r._trip_length - idle < 0:
                    continue
                else:
                    
                    RTV_graph[v._vid][r._rid] = [route, active, idle]
                    
                    All_edges += [[v._vid, r._rid]]
                    
                    r_list = v._assigned_requests + [r]
                    assigned_sum = ACTIVE_CONSTANT_WEIGHT * sum([r0._trip_length for r0 in r_list])
            
                    Sorting_weight += [v._allocated_value1 + assigned_sum - idle]
                    Value_weight += [cardinal_const + assigned_sum - idle]

    '''Recording purposes'''
    hist_act = [v._active_timecost for v in TAXI_LIST]
    hist_val = [v._allocated_value1 for v in TAXI_LIST]
    min_vid = hist_val.index(min(hist_val))
    
    print('Active time minmax:', [min(hist_act), max(hist_act)])
    print('activity value minmax:', [min(hist_val), max(hist_val)])
    print('vid min?', min_vid)
        
    print('NUM OF NEW TAXI LIST:', len(NEW_TAXI_LIST))
    #--------------------------------------------------------------------------
    '''Solving for optimal routes'''
    if len(NEW_TAXI_LIST) > 0:
        route_dict, value_sum, ilp_discr, min_driver = rideshare_algo.solve_rtv_graph(h, H, CAPACITY, RTV_graph, NEW_TAXI_LIST, DEMAND_LIST,
                                                                                      All_edges, Sorting_weight, Value_weight, act_corr, min_vid, t)
    else:
        route_dict, value_sum, ilp_discr, min_driver = [{}, 0, 0, 0]
    
    act_list = [v._allocated_value1 for v in V_STORAGE]
    min_driver = min(act_list)
    ilp_discr = np.std(act_list)
    value_sum = sum(act_list)
    
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
                
                if float(MAX_WAITING_TIME - cur_r._max_waiting_time) < eps:
                    cur_v.update_allocated(cur_r._trip_length)
                
                cur_v._last_finished_time = cur_r.latest_dropoff_time(mapsystem)
                cur_v._last_finished_req = rid
                    
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

for day in SIMULATION_DAY:
    print('DAY:', day)
    REQ_DATA1 = REQ_DATA[REQ_DATA['bydates'] == datetime.date(2013, 5, day)]

    for lat_min, lat_max in zip(MIN_LIST, MAX_LIST):
        print('(LAT_MIN, LAT_MAX):', (lat_min, lat_max))
        print('-------')

        '''Restricting Region of Real Data'''
        if DATA_TYPE == 'WHOLE':
            REQ_DATA0 = REQ_DATA1
        else:
            restrict_region = [list(REQ_DATA1['pickup_coordinates'].iloc[i])[1] < lat_max + eps and
                               list(REQ_DATA0['dropoff_coordinates'].iloc[i])[1] <= lat_max and
                               list(REQ_DATA1['pickup_coordinates'].iloc[i])[1] > lat_min - eps and
                               list(REQ_DATA0['dropoff_coordinates'].iloc[i])[1] >= lat_min
                               for i in range(len(REQ_DATA1))]

            REQ_DATA0 = REQ_DATA1[restrict_region]

        if len(REQ_DATA0) <= 1000:
            continue
        
        valid_data = REQ_DATA0[REQ_DATA0['trip_length'] >= MAX_WAITING_TIME]
        ORI_REQ_NUM = len(valid_data)
        print('NUMREQ from valid_data:', ORI_REQ_NUM)
        
        node_to_num = {}
        for node in set(valid_data['pickup_nodes']):
            num = len(valid_data[valid_data['pickup_nodes'] == node])
            node_to_num[node] = num

        '''Removal of trips that possibly cause vehicles to be stuck for lengthened period'''
        if REMOVAL != 'None':
            pu_nodes = np.array(list(set(valid_data['pickup_nodes'])))
            valid_dest = {}
            for d_pid in set(valid_data['dropoff_nodes']):
                d_waypoint = WayPoint(d_pid, d_pid, 0, 0)
                dist_list = [mapsystem.distance(d_waypoint, WayPoint(o_pid, o_pid, 0, 0)) for o_pid in pu_nodes]

                num = sum([node_to_num[node] for node in pu_nodes[np.array(dist_list) <= MAX_WAITING_TIME - period_length]])

                if num not in valid_dest.keys():
                    valid_dest[num] = []

                valid_dest[num] += [d_pid]

            invalid_id = []
            for num in sorted(list(valid_dest.keys())):
                if num >= len(valid_data) / REMOVAL:
                    break

                for node in valid_dest[num]:
                    invalid_id += list(valid_data[valid_data['dropoff_nodes'] == node].index)

            valid_data = valid_data.drop(invalid_id, axis=0)

        '''Sample reasonable positions (e.g. non-deserted nodes) to generate vehicles'''
        num_to_node = {}
        for node in set(valid_data['pickup_nodes']):
            num = len(valid_data[valid_data['pickup_nodes'] == node])
            if num not in num_to_node.keys():
                num_to_node[num] = []

            num_to_node[num] += [node]

        total_keynum = sum(num_to_node.keys())
        
        '''Choose 2 'popular'nodes based on trip (request) data'''
        node_to_num = []
        for node in set(valid_data['pickup_nodes']):
            num = len(valid_data[valid_data['pickup_nodes'] == node])
            node_to_num += [(num,node)]
        
        POPULAR_NODE1 = sorted(node_to_num)[0]
        POPULAR_NODE2 = sorted(node_to_num)[1]
        #----------------------------------------------------------------------
        '''Weight types (to balance the efficiency-fairness effect)
        1. Constant: w = [0] + list(np.linspace(.7,1,4)) + [.92, .95, .97]; weight_trial = [w for i in range(H)]
        2. Increasing: w = np.linspace(0, .9, 10); weight_trial = np.linspace(w, 1, H)
        3. Binary: w = [int(H - math.sqrt(H)), int(3/4*H), int(H/2), int(H/4)]; weight_trial = [0 for period in range(w)] + [1 for period in range(H-w)]'''
        
        for w in [0] + list(np.linspace(.7,1,4)) + [.92, .95, .97]:

            req_pos = 0
            weight_trial = [w for i in range(H)]
            
            print('WEIGHT:', w)

            '''Reset V_STORAGE - array of vehicle objects'''
            v_location = []
            for num in num_to_node.keys():
                freq = round(num/total_keynum*TAXI_TOTAL_NUM)
                how_many_nodes = len(num_to_node[num])
                
                for node_id in num_to_node[num]:
                    v_location += [manhat_point.loc[node_id, 'Coordinate'] for i in range(round(freq/how_many_nodes))]
            
            random.seed(0)
            loc_id = [random.randint(0, len(v_location) - 1) for i in range(TAXI_TOTAL_NUM)]
            V_STORAGE = np.array([Vehicle(i,v_location[loc_id[i]],CAPACITY) for i in range(TAXI_TOTAL_NUM)])    
            
            for vid in range(len(V_STORAGE)):
                v = V_STORAGE[vid]
                v._point_id = closest_node(v._cur_position)
                
                if RANDOM_VEHICLE_POS_START == False:
                    v._waypoint = WayPoint(v._point_id, v._point_id, 0., 0.)
                else:
                    v._waypoint = mapsystem.GEN_START_POINT(v._point_id, 60)
                
                v._arr_time = 0
               
            print('NUM OF TAXIS:', len(V_STORAGE))

            '''Reset REQ_STORAGE - array of request objects'''
            REQ_STORAGE = np.array([]) #store generated new requests
            TEMP_STORAGE = []
                
            for i in valid_data.index:
                region = 'None'
                origin = valid_data.loc[i, 'pickup_coordinates']
                destination = valid_data.loc[i, 'dropoff_coordinates']
                trip_length = valid_data.loc[i, 'trip_length']
                arr_time = (valid_data.loc[i, 'arrival_time'] - HOUR*3600)//period_length*period_length
                pass_count = valid_data.loc[i, ' passenger_count']

                TEMP_STORAGE += [(arr_time, -trip_length, region, origin, destination, pass_count, i)]

            TEMP_STORAGE.sort()

            for (arr_time, trip_length, region, origin, destination, pass_count, i) in TEMP_STORAGE:
                req_ID = len(REQ_STORAGE)

                newreq = Request(req_ID, region, origin, destination, arr_time, pass_count)

                p = valid_data.loc[i, 'pickup_nodes']
                q = valid_data.loc[i, 'dropoff_nodes']
                newreq._size = pass_count
                newreq._origin_closest_node = p
                newreq._destination_closest_node = q
                newreq._origin_waypoint = WayPoint(p, p, 0., 0.)
                newreq._destination_waypoint = WayPoint(q, q, 0., 0.)
                newreq._trip_length = mapsystem.distance(newreq._origin_waypoint, newreq._destination_waypoint)

                newreq._max_waiting_time = MAX_WAITING_TIME
                newreq._max_delay = newreq._max_waiting_time * 2

                REQ_STORAGE = np.append(REQ_STORAGE, newreq)

            REQ_NUM = len(REQ_STORAGE)
            print('NUM OF REQS:', REQ_NUM)
            removal_rate = REQ_NUM/ORI_REQ_NUM
            
            '''STARTING SIMULATION'''
            DEMAND_LIST = []
            STACKED_ROUTE = [[] for i in range(TAXI_TOTAL_NUM)]
            STACKED_TIME = [[] for i in range(TAXI_TOTAL_NUM)]
            weight_list = weight_trial

            for h in range(1, H+1):

                print('PERIOD', h)
                print('===========================================')
                
                act_corr = weight_list[h-1]
                t = h*period_length
                FIN_ALLOC = {}
                
                DEMAND_LIST = [r for r in DEMAND_LIST if r._assigned == True]
                
                ILP_VALUE, ILP_DISCR, MIN_DRIVER = compute_routing(h, t, act_corr, V_STORAGE, REQ_STORAGE, DEMAND_LIST, FIN_ALLOC)
                
                list_normal = [v._allocated_value1 for v in V_STORAGE] #i.e. store accumulated value (active time) for whole horizon
                
                print('normal')
                print('std:', np.std(list_normal))
                print('minmax:', (np.min(list_normal), np.max(list_normal)))
                print('----')
                
                if h < H:
                    t = (h+1)*period_length
                    status_update(t, V_STORAGE, FIN_ALLOC, STACKED_ROUTE, STACKED_TIME)
                
            served_reqs = [r._rid for r in REQ_STORAGE if (r._assigned == True or r._picked == True or r._finished == True)]
            finished_reqs = [r for r in REQ_STORAGE if r._finished == True]
            picked_reqs = [r for r in REQ_STORAGE if r._pickup_time != None]

            '''Recording Mean Waiting Time (request's perspective)'''
            TOT_WAIT = 0
            for r in picked_reqs:
                TOT_WAIT += r._pickup_time - r._request_time

            MEAN_WAIT = TOT_WAIT/len(picked_reqs)

            '''Recording Mean Delay (request's perspective)'''
            TOT_DELAY = 0
            for r in finished_reqs:
                TOT_DELAY += r._dropoff_time - (r._request_time + r._trip_length)

            if len(finished_reqs) == 0:
                MEAN_DELAY = 0
            else:
                MEAN_DELAY = TOT_DELAY/len(finished_reqs)

            INTRIP_DELAY = 0
            for r in finished_reqs:
                INTRIP_DELAY += r._dropoff_time - r._pickup_time - r._trip_length
            if len(finished_reqs) == 0:
                MEAN_INTRIP = 0
            else:
                MEAN_INTRIP = INTRIP_DELAY/len(finished_reqs)

            '''Recording Service Rate'''
            TOT_SERVED = len(served_reqs)
            SERVICE_RATE = TOT_SERVED/len(REQ_STORAGE)

            f = open('Tradeoff_w_Horizons.csv', 'a', newline = '')
            to_append = [[DATA_TYPE, REMOVAL, removal_rate, MAX_WAITING_TIME, lat_min, HOUR, H, TAXI_TOTAL_NUM, REQ_NUM, weight_trial,
                          TOT_SERVED, SERVICE_RATE, MEAN_WAIT, MEAN_DELAY, MEAN_INTRIP,
                          np.sum(list_normal), np.std(list_normal), np.min(list_normal), np.max(list_normal), day]]
            writer = csv.writer(f)
            writer.writerows(to_append)
            f.close()

            '''Record list_normal as column'''
            f = open('ActiveTime.csv', 'a', newline = '')
            to_append = [list_normal]
            writer = csv.writer(f)
            writer.writerows(to_append)
            f.close()
