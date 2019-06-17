import pandas as pd
import numpy as np
import time
from multiprocessing import Pool
from itertools import product
import networkx as nx

year = 2013
m = 3
pu_hour = 23
v_init = 45/3600 #km/s

#------------------------------------------------------------------------------
'''Building directed graph for path computation'''
manhat_point = pd.read_pickle('D:\\RIDESHARING\\NIPS19\\DataPreprocess\\RoadNetwork\\manhat_point.pkl')
manhat_edge = pd.read_pickle('D:\\RIDESHARING\\NIPS19\\DataPreprocess\\RoadNetwork\\manhat_edge.pkl')

manhat_graph = nx.DiGraph()

for v in manhat_point.index:
    manhat_graph.add_node(v)

for e in manhat_edge.index:
    from_node = manhat_edge.loc[e,'PointIndex1']
    to_node = manhat_edge.loc[e,'PointIndex2']
    time_cost = manhat_edge.loc[e,'Length']/v_init
    dist = manhat_edge.loc[e,'Length']
    
    manhat_graph.add_edge(from_node, to_node, weight = time_cost, length = dist, eid = e)

#------------------------------------------------------------------------------
'''Reading hourly training data'''
trip1 = pd.read_pickle('D:\\RIDESHARING\\NIPS19\\DataPreprocess\\TravelTimePrediction\\TrainingData\\trip'+str(pu_hour)+'.pkl')

if len(trip1.index)>150000:
    frac = 150000/len(trip1.index)
    trip1 = trip1.sample(frac = frac) #random sampling for too large data

print(trip1.shape)

#------------------------------------------------------------------------------
def trip_mat(i):
    
    global manhat_graph, manhat_edge, trip1
    
    pu = list(trip1.index.get_level_values(0))[i]
    do = list(trip1.index.get_level_values(1))[i]
    path = nx.dijkstra_path(manhat_graph, pu, do)
    
    edge_id = []
    
    for j in range(len(path)-1):
        from_node = path[j]
        to_node = path[j+1]
        edge_id += [manhat_graph.edges[from_node, to_node]['eid']]
    
    trip_vec = [(e in edge_id) for e in manhat_edge.index]
    
    return trip_vec

def Compute_Offset(pos, mat, et, tt):
    et_ = et[mat[:,pos].flatten() == 1]
    tt_ = tt[mat[:,pos].flatten() == 1]
    
    return np.sum(et_ - tt_)

def Modify_Ts(i, k, Offset_val, t_S, S_trip):
    if i not in S_trip:
        return t_S[i][0]
    else:
        x = S_trip.index(i)
        if Offset_val[x]<0:
            return t_S[i][0]*k
        else:
            return t_S[i][0]/k

def Neighbor_Size(pos, S_trip_edges):
    
    global manhat_graph, manhat_edge
    
    e = manhat_edge.index[pos]
    n1 = manhat_edge.loc[e,'PointIndex1']
    n2 = manhat_edge.loc[e,'PointIndex2']
    
    #search for those edges in S_trip that contains n1 and n2
    n1_nbr = list(nx.all_neighbors(manhat_graph, n1)) 
    n2_nbr = list(nx.all_neighbors(manhat_graph, n2))
    e_nbr = [[x,n1] for x in n1_nbr if [x,n1] in S_trip_edges] + [[n2,y] for y in n2_nbr if [n2,y] in S_trip_edges]
    
    return len(e_nbr)

if __name__ == '__main__':
    with Pool(16) as p:
        mat = p.map(trip_mat, [i for i in range(len(trip1))])        
        mat = np.array(mat)
        
        tt = np.array(trip1) #i.e. array of node-based trips real travel time extracted from data
        tt = np.array([tt]).T
        L = np.array(manhat_edge.loc[:,'Length']) #i.e. geographical distance between the endpoints
        L = np.array([L]).T        
        avg_speed = np.matmul(mat,L)/tt
        avg_speed = avg_speed.flatten()
        
        '''Removal of outliers in avg_speed'''
        v_check1 = 1000*avg_speed>=0.5
        v_check2 = 1000*avg_speed<=30
        v_check = np.array([(x and y) for x,y in zip(v_check1, v_check2)])
        trip1 = trip1[v_check==1]
        tt = np.array(trip1)
        
        '''Initialization'''
        t_S = L/v_init #travel time on all edges in manhat_edge
        
        if sum(np.isnan(t_S).flatten()) > 0:
            print('PROBLEM IN FIRST INIT t_S')
            print(t_S)
            print('length of t_S', len(t_S))
            
            ind = [i for i,x in enumerate(np.isnan(t_S).flatten()) if x == True]
            print('problem indexes', ind)
            print('check the length', L[ind])
        
        counter = 0
        again = True
        
        print('Start iteration..')
        start = time.perf_counter()
        
        while again:
            counter +=1
            print(counter)
            print('duration', time.perf_counter()-start)
            again = False
            
            mat = p.map(trip_mat, [i for i in range(len(trip1))])
            mat = np.array(mat)
            #i.e. compute truth values of each edge being passed by at least one of the trip records in trip1

            et = np.matmul(mat,t_S) #i.e. expected travel time - to be updated each time t_S updated by Modify_Ts (see l. 152)
            et = et.flatten()
            
            RelErr = np.sum(abs(et - tt)/tt) #i.e. relative error
            
            S_trip = [i for i in range(len(manhat_edge.index)) if (mat[:,i].flatten()==1).any()==1]
            #i.e. set of streets (edge) being passed at least once in trip1
            
            Offset_val = p.starmap(Compute_Offset, product([pos for pos in S_trip],[mat],[et],[tt]))            
            #i.e. as a decision variable to update t_S; negative means the current travel time (et) is too small
            
            k = 1.2 #i.e. to determine the extent of modification of t_S (see l. 72-75)
            while True:                
                newt_S = p.starmap(Modify_Ts, product([i for i in range(len(t_S))], [k], [Offset_val], [t_S], [S_trip]))
                newt_S = np.array([newt_S]).T
                
                newet = np.matmul(mat,newt_S)
                newet = newet.flatten()
            
                NewRelErr = np.sum(abs(newet - tt)/tt)
                
                if NewRelErr<RelErr:
                    t_S = np.array(newt_S)
                    
                    if sum(np.isnan(t_S).flatten()) > 0:
                        print('PROBLEM IN ITERATION t_S') 
                        
                    RelErr = NewRelErr
                    
                    for i in S_trip:
                        eid = manhat_edge.index[i]
                        x = manhat_edge.loc[eid, 'PointIndex1']
                        y = manhat_edge.loc[eid, 'PointIndex2']
                        
                        if np.isnan(t_S[i][0]):
                            print('NAN ASSESSMENT in S_trip, problem edge points', (x,y))
                        
                        manhat_graph.edges[x,y]['weight'] = t_S[i][0]
                        '''if np.isnan(t_S[i][0]):
                            print('in S_trip', i)
                            print('eid', eid)'''
                            
                    again = True
                    break
                else: #i.e. since the error increases, lessen the degree of modification
                    k = 1 + .75*(k-1)
                    if k < 1.0001: 
                        break #i.e. here, k is deemed to be too small and the final t_S is concluded to be optimal
                    else:
                        continue
        print('finish iteration step',len(S_trip)) 
         
        '''Travel time on edges that are not passed by any trips in the data trip1'''
        N_S_trip = [i for i in range(len(manhat_edge.index)) if i not in S_trip] #i.e. set of these edges with 'unknown' travel time
        S_trip_edges = [[manhat_edge['PointIndex1'].iloc[i],manhat_edge['PointIndex2'].iloc[i]] for i in S_trip]
        
        while len(N_S_trip)!=0:    
            print('Length of N_S_trip', len(N_S_trip))
            
            '''Estimated by the average of travel time on trip1-recorded neighboring edges'''
            nbr_size = p.starmap(Neighbor_Size, product([pos for pos in N_S_trip], [S_trip_edges])) #i.e. number of neighboring edges in S_trip         
            N_S_trip = [x for _,x in sorted(zip(nbr_size,N_S_trip), reverse = True)] #i.e. sorting by the number of neighbors (more reliable estimate)
            
            e = manhat_edge.index[N_S_trip[0]]
            
            n1 = manhat_edge.loc[e,'PointIndex1']
            n2 = manhat_edge.loc[e,'PointIndex2']
            
            n1_nbr = list(nx.all_neighbors(manhat_graph, n1)) 
            n2_nbr = list(nx.all_neighbors(manhat_graph, n2))
            e_nbr = [[x,n1] for x in n1_nbr if [x,n1] in S_trip_edges] + [[n2,y] for y in n2_nbr if [n2,y] in S_trip_edges]
            #i.e. set of these trip1-recorded edges
            
            if len(e_nbr) == 0:
                print('While loop not in S_trip stop iterating')
                break
            
            sum_speed = 0
            for i in range(len(e_nbr)):
                x,y = e_nbr[i]
                time_cost = manhat_graph.edges[x,y]['weight']
                
                if np.isnan(time_cost):
                    print('NAN ASSESSMENT not in S_trip: edge points',e_nbr[i])
                    
                dist = manhat_graph.edges[x,y]['length']
                
                if time_cost !=0 and ~np.isnan(time_cost):
                    sum_speed += dist/time_cost
                else:
                    continue
            
            v_e = sum_speed/len(e_nbr) #i.e. average (by neighbor edges) speed
            t_S[N_S_trip[0]][0] = manhat_graph.edges[n1,n2]['length']/v_e #i.e.. update t_S
            manhat_graph.edges[n1,n2]['weight'] = t_S[N_S_trip[0]][0]
            
            if np.isnan(t_S[N_S_trip[0]][0]):
                print('NAN ASSESSMENT Final iteration not in S_trip: edge points', (n1, n2))
                #print('N_S_trip', N_S_trip[0])
                print('the e_nbr', e_nbr)
                print('v_e?', v_e)
            
            '''This 'unknown' edge is recorded into the set of edges passed by some trips in data trip1'''
            S_trip += [N_S_trip[0]]
            S_trip_edges += [[manhat_edge['PointIndex1'].iloc[N_S_trip[0]],manhat_edge['PointIndex2'].iloc[N_S_trip[0]]]]
            del N_S_trip[0]
            
        '''Recording into file'''
        traveltime = {}
        traveltime[pu_hour] = t_S.flatten()
        tt_df = pd.DataFrame(data = traveltime, columns = [pu_hour])
        tt_df.index = manhat_edge.index
        tt_df.to_pickle('D:\\RIDESHARING\\NIPS19\\DataPreprocess\\TravelTimePrediction\\TrainedTravelTime\\TravelTime'+str(pu_hour)+'.pkl')            
