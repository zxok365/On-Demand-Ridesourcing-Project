import numpy as np
from Waypoint import MapSystem

eps = 1e-6

'''return the distance between p and q by the time cost network'''
def distance(p : int, q : int, time_cost_network : np.ndarray) -> float:
    t = time_cost_network[p][q]
    if  t < 0:
        return 9999999
    else:
        return t

'''Given vehicle and a number of request, using DFS to check whether vehicle can pick up these requests, if yes then find the optimal routes'''
def check_feasible(cur_time : float, cur_pos : int, dst_list : list, time_cost_network : np.ndarray,
                   now_cost : float = 0, Routing = None, cur_num = 1, min_cost = -9999999, idle_time = 0.) -> (bool, float, list):

    if min_cost != -9999999 and now_cost > min_cost + eps:
         return (False, -1, [], -1, 0)

    if Routing == None:
        Routing = []
        for i in dst_list:
            Routing += [0]
    
    feasible = True
    
    for t in Routing:
        if t == 0:
            feasible = False
            break
    
    if feasible == True:
        return (True, now_cost, Routing, cur_time, idle_time)
    
    min_Routing = None
    min_end_time = -1
    min_idle_time = 0
    r_num = 0
    tmp = 0
    r_tot = 0
    
    for t in dst_list:
        tmp = tmp + 1
        if Routing[tmp - 1] == 0:
            r_tot = r_tot + t[4]
            r_num = r_num + t[3] * t[4]

    tmp = 0

    for t in dst_list:
        tmp = tmp + 1
        if Routing[tmp - 1] == 0:
            nxt_pos, lmt_time, r_id, up_down, r_size = t
            tmp_cost = r_size * distance(cur_pos, nxt_pos, time_cost_network)
            if cur_time + tmp_cost > lmt_time + eps:
                return (False, -1, [], -1, 0)

    tmp = 0
    for t in dst_list:
        tmp = tmp + 1
        if Routing[tmp - 1] == 0:
            nxt_pos, lmt_time, r_id, up_down, r_size= t

            '''to make sure that dropoff happens after pickup'''
            if up_down == 1:
                BO = True
                ttmp = 0
                for x in dst_list:
                    a, b, c, d, e = x
                    if c == r_id and d == 0:
                        if Routing[ttmp] == 0:
                            BO = False
                            break
                        else:
                            BO = True
                            break
                    ttmp += 1
                if not BO:
                    continue

            tmp_cost = distance(cur_pos,nxt_pos, time_cost_network)

            k = now_cost + r_num * tmp_cost
            idle_tmp = 0

            if up_down == 0 and r_num * 2 == r_tot:
                idle_tmp = tmp_cost

            Routing[tmp - 1] = cur_num

            if min_cost == -9999999 or float(min_cost) > k:
                new_feasible, new_cost, new_Routing, end_time, tmp_idle_time = check_feasible(cur_time + tmp_cost, nxt_pos, dst_list,time_cost_network, k, Routing, cur_num + 1, min_cost, idle_time + idle_tmp)

                if new_feasible:
                    feasible = True
                    if new_cost <  min_cost or min_cost == -9999999:
                        min_cost = new_cost
                        min_Routing = list(new_Routing)
                        min_end_time = end_time
                        min_idle_time = tmp_idle_time
                        
            Routing[tmp - 1] = 0

    if cur_num == 1:
        ans_Routing = []
        
        if feasible:
            order = []
            
            for i in range(0, len(dst_list)):
                order += [0]
            
            for i in range(0, len(dst_list)):
                order[min_Routing[i] - 1] = i
            
            pos = cur_pos
            
            for i in range(0, len(dst_list)):
                j = order[i]
                
                if min_Routing[j] == i + 1:
                    nxt_pos, lmt_time, r_id, up_down, r_size = dst_list[j]
                    tmp_cost = distance(pos, nxt_pos, time_cost_network)
                    ans_Routing += [(cur_time + tmp_cost, r_id, up_down, tmp_cost * r_num)]
                    r_num = r_num - up_down * r_size
                    pos = nxt_pos
                    cur_time = cur_time + tmp_cost
                else:
                    print("ERROR!!")
            
            min_end_time = cur_time
        
        return (feasible, min_cost, ans_Routing, min_end_time, min_idle_time)
    
    else:
        return (feasible, min_cost, min_Routing, min_end_time, min_idle_time)

'''The following returns the optimal route (regardless of the feasible constraints)'''
def calculate_minimum_cost(cur_time : float, cur_pos: int, dst_list: list,time_cost_network, now_cost : float = 0, Routing = None, cur_num = 1,idle_time = 0.,min_cost = -9999999) -> (float, list):

    if min_cost != -9999999 and now_cost > min_cost + eps:
         return (-1, [], -1, 0)

    if Routing == None:
        Routing = []
        for i in dst_list:
            Routing += [0]

    feasible = True
    for t in Routing:
        if t == 0:
            feasible = False
            break
    if feasible == True:
        return (now_cost, Routing, cur_time, idle_time)

    min_Routing = None
    min_end_time = -1
    min_idle_time = 0.
    r_num = 0
    tmp = 0

    for t in dst_list:
        tmp = tmp + 1
        a, b, c, d, e= t
        if Routing[tmp - 1] == 0:
            r_num = r_num + d * e

    tmp = 0
    for t in dst_list:
        tmp = tmp + 1
        if Routing[tmp - 1] == 0:
            nxt_pos, lmt_time, r_id, up_down, r_size = t

            if up_down == 1: #i.e. to make sure that dropoff happens after pickup
                BO = True
                ttmp = 0
                for x in dst_list:
                    a, b, c, d, e = x
                    if c == r_id and d == 0:
                        if Routing[ttmp] == 0:
                            BO = False
                            break
                        else:
                            BO = True
                            break
                    ttmp += 1
                if not BO:
                    continue

            tmp_cost = distance(cur_pos, nxt_pos, time_cost_network)
            k = now_cost + tmp_cost * r_num
            idle_tmp = 0

            if up_down == 0 and r_num == 1:
                idle_tmp = tmp_cost

            Routing[tmp - 1] = cur_num

            if min_cost == -9999999 or float(min_cost) > k:
                new_cost, new_Routing, end_time, tmp_idle_time= calculate_minimum_cost(cur_time + tmp_cost, nxt_pos, dst_list,
                                                                     time_cost_network, k, Routing, cur_num + 1, idle_time + idle_tmp)
                if new_cost < min_cost or min_cost == -9999999:
                    min_cost = new_cost
                    min_Routing = list(new_Routing)
                    min_end_time = end_time
                    min_idle_time = tmp_idle_time

            Routing[tmp - 1] = 0

    if cur_num == 1:
        ans_Routing = []
        order = []

        for i in range(0, len(dst_list)):
            order += [0]

        for i in range(0, len(dst_list)):
            order[min_Routing[i] - 1] = i

        pos = cur_pos
        for i in range(0, len(dst_list)):
            j = order[i]
            
            if min_Routing[j] == i + 1:
                nxt_pos, lmt_time, r_id, up_down, r_size = dst_list[j]
                tmp_cost = distance(pos, nxt_pos, time_cost_network) #i.e. tmp_cost per passenger
                ans_Routing += [(cur_time + tmp_cost, r_id, up_down, tmp_cost * r_num)]
                r_num = r_num - up_down * r_size
                pos = nxt_pos
                cur_time = cur_time + tmp_cost
            else:
                print("ERROR")
            
            min_end_time = cur_time
        
        return (min_cost, ans_Routing, min_end_time, min_idle_time)
    
    else:
        return (min_cost, min_Routing, min_end_time, min_idle_time)

'''Given vehicle and a request list, the following checks its trip feasibility and finds optimal route'''
def calc_min_routing(v : Vehicle, r : list, cur_time, map : MapSystem, cur_network : np.ndarray):
    Todo = []
    min_time_cost = 0
    
    for x in v.boarded_requests:
        Todo += [(x.get_destination_id, x.latest_dropoff_time(map), x.get_id, 1, 1)]
        min_time_cost += distance(v.get_point_id, x.get_destination_id, cur_network)

    if r == -1:
        feasible, min_cost, routing, end_time, idle_time = check_feasible(cur_time=cur_time, cur_pos=v.get_point_id, dst_list=Todo, now_cost=0.- min_time_cost, time_cost_network=cur_network)

        if not feasible:
            min_cost, routing, end_time, idle_time = calculate_minimum_cost(cur_time=cur_time, cur_pos=v.get_point_id, dst_list=Todo, now_cost=0.- min_time_cost, time_cost_network=cur_network)

        totel_length = end_time - cur_time

        return feasible, routing, min_cost, totel_length, idle_time
    else:
        for req in r:
            lat_pu_time = req.latest_acceptable_pickup_time

            if lat_pu_time + eps < cur_time + distance(v.get_point_id,req.get_origin_id,cur_network):
                return False, -1, -1, -1, -1

            Todo += [(req.get_origin_id, lat_pu_time, req.get_id, 0, 1)]
            Todo += [(req.get_destination_id, req.latest_dropoff_time(map), req.get_id, 1, 1)]
            min_time_cost += distance(req.get_origin_id, req.get_destination_id, cur_network)
    
        feasible, min_cost, routing, end_time, idle_time = check_feasible(cur_time=cur_time, cur_pos=v.get_point_id, dst_list=Todo, now_cost=0.- min_time_cost, time_cost_network=cur_network)
        totel_length = end_time - cur_time

        return feasible, routing, min_cost, totel_length, idle_time