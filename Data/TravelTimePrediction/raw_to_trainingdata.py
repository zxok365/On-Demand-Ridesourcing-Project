import pandas as pd
import numpy as np

y = 2013
m = 3

def closest_node(loc):

    global manhat_point
    
    loc = np.array([loc])
    nodes = np.array(list(manhat_point['Coordinate']))
    dist_2 = np.sum((nodes - loc)**2, axis=1)
    pos = np.argmin(dist_2)
    
    return manhat_point.index[pos]

'''Raw to taxi_csv'''
taxicsv = pd.read_csv('D:\\RIDESHARING\\NIPS19\\DataPreprocess\\RequestGenerator\\'+str(y)+'\\trip_data_'+str(m)+'.csv', header=0, dtype='object')
taxi_csv = taxicsv.drop([" vendor_id"," trip_distance"," rate_code"," store_and_fwd_flag",
                         "medallion"," trip_time_in_secs"], axis=1)
print('End reading')

'''Separate original timestamp to date, hour, minute columns'''
date = [pd.Timestamp(x,tz = None).date() for x in taxi_csv[' pickup_datetime']]
pickup_date = pd.DataFrame(data = date, index = list(taxi_csv.index), columns = ["bydates"])
taxi_csv = pd.concat([taxi_csv,pickup_date], axis = 1)

hour = [pd.Timestamp(x,tz=None).hour for x in taxi_csv[' pickup_datetime']]
pickup_hour = pd.DataFrame(data = hour, index = list(taxi_csv.index), columns = ["byhours"])
taxi_csv = pd.concat([taxi_csv,pickup_hour], axis=1)

minute = [pd.Timestamp(x,tz=None).minute for x in taxi_csv[' pickup_datetime']]
pickup_min = pd.DataFrame(data = minute, index=list(taxi_csv.index), columns=["bymins"])
taxi_csv = pd.concat([taxi_csv,pickup_min], axis=1)

print('convert lat,long to integers')
taxi_csv[" passenger_count"] = taxi_csv[" passenger_count"].astype(int)
taxi_csv[" pickup_longitude"] = taxi_csv[" pickup_longitude"].astype(float)
taxi_csv[" pickup_latitude"] = taxi_csv[" pickup_latitude"].astype(float)
taxi_csv[" dropoff_longitude"] = taxi_csv[" dropoff_longitude"].astype(float)
taxi_csv[" dropoff_latitude"] = taxi_csv[" dropoff_latitude"].astype(float)

'''Adding in pickup/dropoff coordinates'''
print('long,lat to coor')
pu_coor = [tuple([x,y]) for x,y in zip(taxi_csv[" pickup_longitude"], taxi_csv[" pickup_latitude"])]
taxi_csv["pickup_coordinates"] = pu_coor

do_coor = [tuple([x,y]) for x,y in zip(taxi_csv[" dropoff_longitude"], taxi_csv[" dropoff_latitude"])]
taxi_csv["dropoff_coordinates"] = do_coor

print('Dropping indv long, lat..')
taxi_csv = taxi_csv.drop([" pickup_longitude"," pickup_latitude"," dropoff_longitude",
                          " dropoff_latitude"], axis=1)

'''taxi_csv to taxi_tt'''
'''Adding in closest nodes to pickup/dropoff coordinates'''
pu_node = [closest_node(loc) for loc in taxi_csv['pickup_coordinates']]
pickup_nodes = pd.DataFrame(data = pu_node, index = list(taxi_csv.index), columns = ["pickup_node"])
taxi_csv = pd.concat([taxi_csv, pickup_nodes], axis = 1)

do_node = [closest_node(loc) for loc in taxi_csv['dropoff_coordinates']]
dropoff_nodes = pd.DataFrame(data = do_node, index = list(taxi_csv.index), columns = ["dropoff_node"])
taxi_csv = pd.concat([taxi_csv, dropoff_nodes], axis = 1)

'''taxi_tt to trip (training data) by extracting all node-based trips happening per hour'''
for h in range(24):
    print(h)
    trip1 = taxi_csv.loc[taxi_csv['pickup_hour'] == h]
    trip1.loc[:,' trip_time_in_secs'] = trip1.loc[:,' trip_time_in_secs'].astype(int)
    trip1.loc[:,' trip_distance'] = trip1.loc[:,' trip_distance'].astype(float)

    '''Removing invalid data'''
    trip1 = trip1[np.array(trip1['pickup_node']) != np.array(trip1['dropoff_node'])]
    trip1 = trip1[np.array(trip1[' trip_distance'])!=0]
    trip1 = trip1[np.array(trip1[' trip_time_in_secs'])!=0]

    #distance-related
    trip1 = trip1[np.array(trip1[' trip_distance'])<=32]
    trip1 = trip1[np.array(trip1[' trip_distance'])>=.4]

    #time-related
    trip1 = trip1[np.array(trip1[' trip_time_in_secs'])<=2000]
    trip1 = trip1[np.array(trip1[' trip_time_in_secs'])>=300]

    #velocity-related
    velo = [1609.34*trip1[' trip_distance'].iloc[i]/trip1[' trip_time_in_secs'].iloc[i] for 
            i in range(len(trip1.index))]
    f = np.array(velo)<=30
    s = np.array(velo)>=.5
    t_arr = np.array([(x and y) for x,y in zip(f,s)])
    trip1 = trip1[t_arr == 1]

    trip1 = trip1.groupby(['pickup_node','dropoff_node'])[' trip_time_in_secs'].mean() 
    #i.e. same type of node-based trip belonging to different trip data is merged by averaging
    trip1.to_pickle('D:\\RIDESHARING\\NIPS19\\DataPreprocess\\TravelTimePrediction\\TrainingData\\trip'+str(h)+'.pkl')


