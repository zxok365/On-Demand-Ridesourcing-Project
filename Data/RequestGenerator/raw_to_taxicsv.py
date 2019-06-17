import pandas as pd

y = 2013
m = 5

taxicsv = pd.read_csv('D:\\RIDESHARING\\NIPS19\\DataPreprocess\\RequestGenerator\\'+str(y)+'\\trip_data_'+str(m)+'.csv', header=0, dtype='object')
taxi_csv = taxicsv.drop([" vendor_id"," trip_distance"," rate_code"," store_and_fwd_flag",
                         "medallion"," trip_time_in_secs"], axis=1)

'''Adding in date, hours, and minutes'''
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

'''Adding in Coordinates'''
pu_coor = [tuple([x,y]) for x,y in zip(taxi_csv[" pickup_longitude"], taxi_csv[" pickup_latitude"])]
taxi_csv["pickup_coordinates"] = pu_coor
do_coor = [tuple([x,y]) for x,y in zip(taxi_csv[" dropoff_longitude"], taxi_csv[" dropoff_latitude"])]
taxi_csv["dropoff_coordinates"] = do_coor

print('Dropping indv long, lat..')
taxi_csv = taxi_csv.drop([" pickup_longitude"," pickup_latitude"," dropoff_longitude",
                          " dropoff_latitude"], axis=1)
taxi_csv.to_pickle('D:\\RIDESHARING\\NIPS19\\DataPreprocess\\RequestGenerator\\taxi_csv'+str(m)+'.pkl')