import numpy as np
import csv
import HungarianAlgorithm as HA

ROUND_NUM = 100
eps = 1e-6

'''Functions list:
    1. solve_rtv_graph - main function; taking in the bipartite graph essentials and other infos constructed in the main script 
    to output final routing and internally update attributes
    2. ReallocAlgo - Algorithm 1 implementation
    3. match_func - computing maximum weight matching (using Hungarian Algorithm)'''

def ReallocAlgo(lmbd, M_eff, M_fair, act_eff, act_fair, min_eff, min_fair, taxi_list, sorting_weight, all_edges):

    M_eff = sorted(M_eff)
    M_fair = sorted(M_fair)
    V_eff, R_eff = zip(*M_eff)
    V_fair, R_fair = zip(*M_fair)

    V_eff = list(V_eff)
    R_eff = list(R_eff)
    V_fair = list(V_fair)
    R_fair = list(R_fair)

    V_fin = list(V_eff)
    R_fin = list(R_eff)

    act_fin = act_eff[:]
    act_check = [x < lmbd*min_fair for x in act_fin] #i.e. binaries indicating the vehicle-request pair that needs reallocation
    
    while sum(act_check)>0:
        print('LOOP 1')
        V_sort = sorted([v._vid for v in taxi_list])

        ind = act_check.index(1) #i.e. initialize the index of reallocation 
        v0 = V_sort[ind]
        r0 = R_fair[ind] 
        print(v0)

        while True:
            print('LOOP 2')
            ind0 = V_fin.index(v0)
            
            R_fin[ind0] = R_fair[ind0]
            r0 = R_fin[ind0]
            
            act_check[ind0] = 0
            act_fin[ind0] = act_fair[ind0]
            
            if r0 in R_eff and r0 >= 0:
                print(r0)
                v0 = V_eff[R_eff.index(r0)]
                print(v0)
                print('---')
                print('old edge:', (v0, r0))
                print('old value:', act_eff[R_eff.index(r0)])
                
                
                print('new edge:', (v0, R_fair[V_eff.index(v0)]))
                print('new value:', act_fair[V_eff.index(v0)])
                
                if v0 == V_sort[ind]:
                    print('chain -> cycle v')    
                    break
            else:
                if r0 == R_eff[ind]:
                    print('chain -> cycle')
                else:
                    print('chain breaks')    
                break
    
    M_fin = list(zip(V_fin, R_fin))
    
    return M_fin
        
def match_func(w, all_left, all_right):
    match_tmp = list(HA.maxWeightMatching(w))

    match_sol = []
    unassigned_v = []

    for i in match_tmp[0].keys():
        j = match_tmp[0][i]

        if i > len(all_left) - 1:
            continue
        else:
            vid = all_left[i]

        if j > len(all_right) - 1:
            print('j:', j)
            print('vid:', vid)
            unassigned_v += [vid]
            continue
        else:
            rid = all_right[j]

        if w[i][j] == 0:
            match_sol += [[vid, -vid - 1]]
            continue

        match_sol += [[vid, rid]]

    if len(unassigned_v) > 0:
        print('v_list:', unassigned_v)

    return match_sol

def solve_rtv_graph(h, H, capacity, rtv, taxi_list, demand_list, all_edges, sorting_weight,
                    value_weight, lmbd, min_vid, cur_time):

    '''To store all possible solutions and their corresponding measures'''
    END_SOLUTIONS = []
    TOT_VALUE = []
    TOT_VARIANCE = []
    TOT_MIN = []
    TOT_MAX = []
    
    taxi_id = [v._vid for v in taxi_list]
    demand_id = [r._rid for r in demand_list]

    all_left, all_right = zip(*all_edges)
    all_right = list(set(all_right))
    all_left = list(set(all_left))
    
    '''Constructing initial bipartite graph'''
    w = [[0 for v in range(len(all_right))] for u in range(len(all_left))]

    for eid in range(len(all_edges)):
        u, v = all_edges[eid]
        
        i = all_left.index(u)
        j = all_right.index(v)
        
        w[i][j] = int(ROUND_NUM*value_weight[eid])
    
    '''Start iterations for all possible allocations'''
    iter_num = 0
    cur_min = -999999999
    w0 = w[:]
    
    while True: 
        iter_num += 1
        print('===========================')
        print('iter', iter_num)
        
        match_sol = match_func(w0, all_left, all_right)
        act_list = [sorting_weight[all_edges.index(list(e))] for e in match_sol]                
            
        '''Recording solutions'''
        if min(act_list) > cur_min:
            END_SOLUTIONS += [match_sol]
            TOT_MIN += [min(act_list)]
            TOT_MAX += [max(act_list)]
            TOT_VARIANCE += [np.std(act_list)]
            TOT_VALUE += [sum(act_list)]
        else:
            break #i.e. when the minimum fairness shifts down compared to previous iteration
            
        '''Edge deletion'''
        to_remove = [i for i,x in enumerate(sorting_weight) if x <= min(act_list)]
        
        for idx in to_remove:
            v, r = all_edges[idx]
            
            i = all_left.index(v)
            j = all_right.index(r)
            
            if r < 0: #i.e. to speed up computation; r = -1 indicates empty request
                w0[i][j] = 100
                continue
            
            w0[i][j] = 0
        
        cur_min = min(act_list)
    
    print('TOT_VALUE:', TOT_VALUE)
    print('TOT_MIN:', TOT_MIN)
    print('TOT_MAX:', TOT_MAX)
    print('TOT_VARIANCE:', TOT_VARIANCE)

    '''Recording single-batch trade-off'''
    if len(set(TOT_VALUE)) > 1:
        f = open('F:\\ride sharing\\pnasresults\\BipartiteTradeoff_rideshare.csv', 'a', newline = '')
        to_append = [['round', ':', h]] + [[a, b, c] for a,b,c in zip(TOT_VALUE, TOT_VARIANCE, TOT_MIN)] + [['-', '-', '-']]
        writer = csv.writer(f)
        writer.writerows(to_append)
        f.close()

    '''Compute desired lambda-matching'''
    F_opt = TOT_MIN[-1]
    F_sol = lmbd * F_opt #i.e. threshold for minimum accumulated active time

    M_eff = sorted(END_SOLUTIONS[0])
    act_eff = [sorting_weight[all_edges.index(list(e))] for e in M_eff]
    min_eff = min(act_eff)
    
    if F_sol <= min_eff:
        M_fin = M_eff #i.e. minimum threshold can be smaller than the minimum in the efficient solution
    else:
        print('lambda*F_opt:', F_sol)
 
        M_fair = END_SOLUTIONS[len(END_SOLUTIONS) - 1]
        act_fair = [sorting_weight[all_edges.index(list(e))] for e in M_fair]

        M_fin = ReallocAlgo(lmbd, M_eff, M_fair, act_eff, act_fair, min_eff, F_opt, taxi_list, sorting_weight,
                            all_edges)
                            
    act_fin = [sorting_weight[all_edges.index(list(e))] for e in M_fin]
    tot_value = sum(act_fin)

    '''Assigning final optimal route to vehicles'''
    route_dict = {}
    act_list = []

    for m in M_fin:
        vid = list(m)[0]
        rid = list(m)[1]
        cur_vid = taxi_id.index(vid)
                    
        '''skip if fixed'''
        if len(taxi_list[cur_vid]._boarded_requests) == capacity:
            act_list += [taxi_list[cur_vid]._allocated_value1]
            continue
        
        act_list += [taxi_list[cur_vid]._allocated_value1 + rtv[vid][rid][1] - rtv[vid][rid][2]]
        
        '''Update vehicle accumulated values'''
        if h == H: #i.e. only last period, otherwise it is updated in the main script 
            #due to possible change of routing from appending new requests
            if rid < 0:
                r_list = taxi_list[cur_vid]._assigned_requests + []
                assigned_sum = sum([r0._trip_length for r0 in r_list])
                incr_value = assigned_sum - rtv[vid][rid][2]
            else:
                cur_rid = demand_id.index(rid)
                r_list = taxi_list[cur_vid]._assigned_requests + [demand_list[cur_rid]]
                assigned_sum = sum([r0._trip_length for r0 in r_list])
                incr_value = assigned_sum - rtv[vid][rid][2]
            
            taxi_list[cur_vid].update_allocated(incr_value)
                    
        '''Assign routing to route_dict'''
        tmp_route = rtv[vid][rid][0]
        
        if len(tmp_route) == 0:
            route_dict[vid] = ()
        else:
            reaching_time, rid, p_or_d, cost = zip(*rtv[vid][rid][0])
            route_dict[vid] = tuple(list(zip(rid, p_or_d)))
        
    return [route_dict, tot_value, np.std(act_list), np.min(act_list)]
