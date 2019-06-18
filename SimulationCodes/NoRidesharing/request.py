from Waypoint import WayPoint
import searching

MAX_WAITING_TIME = 180
MAX_DELAY = 6*60

class Request:
    def __init__(self,request_id : int, region: int = 0, origin : tuple = (0,0), destination : tuple = (0,0), 
                 request_time : int = 0, size : int = 1, length : float = None, pickup_time = None,
                 max_wait_time = -1, max_delay = -1 , maximum_dropoff_time = -1, assigned = False,
                 picked = False, vehicle = -1, origin_id = -1,destination_id = -1, origin_waypoint : WayPoint = WayPoint() , destination_waypoint : WayPoint = WayPoint()):
        
        self._rid = request_id #i.e. request ID
        self._region = region #i.e. region ID
        
        self._origin = origin #i.e. coordinates of pickup location
        self._origin_id = origin_id
        self._origin_waypoint = origin_waypoint
        
        self._destination = destination #i.e. coordinates of dropoff location
        self._destination_id = destination_id
        self._destination_waypoint = destination_waypoint
        
        self._request_time = request_time #i.e. request arrival time, rounded down to its 30s
        self._size = size #i.e. number of passengers in the request
        self._trip_length = length #i.e. travelling time between pickup-dropoff location (node-based)
        
        if max_wait_time == -1:
            self._max_waiting_time = MAX_WAITING_TIME
        else:
            self._max_waiting_time = max_wait_time

        if max_delay == -1:
            self._max_delay = MAX_DELAY
        else:
            self._max_delay = max_delay
            
        self._max_dropoff_time = maximum_dropoff_time
        
        '''Allocation attributes'''        
        self._assigned = assigned
        self._assign_time1 = None
        
        self._picked = picked
        self._pickup_time = pickup_time
        self._vehicle =  vehicle
        
        self._finished = False
        self._finish_time = -1
        self._dropoff_time = None
        
        self._delay = None
        self._wait = None
        
        '''Accumulated trip attributes'''
        self._trip_active = 0
        self._trip_value = 0
        
    @property
    def get_id(self) -> int:
        return self._rid

    @property
    def get_origin_id(self):
        return self._origin_id

    @property
    def get_destination_id(self):
        return self._destination_id

    @property
    def get_origin(self) -> tuple:
        return self._origin

    @property
    def get_destination(self) -> tuple:
        return self._destination

    def set_origin_id(self, SET : int):
        self._origin_id = SET

    def set_destination_id(self, SET : int):
        self._destination_id = SET

    def set_expect_dropoff_time(self, time):
        self._expect_dropoff_time = time

    def set_max_delay(self,value):
        self._max_delay = value

    def set_max_waiting_time(self,value):
        self._max_waiting_time = value

    @property
    def max_delay(self):
        return self._max_delay

    @property
    def max_waiting_time(self):
        return self._max_waiting_time
    
    def min_time_cost(self, time_cost_network) -> float:
        dist = searching.distance(self._origin_id, self._destination_id, time_cost_network)
        return dist
    
    def latest_dropoff_time(self, map) -> float:
        if self._max_dropoff_time == -1:
            self._max_dropoff_time = map.distance(self._origin_waypoint,self._destination_waypoint) + self._max_delay + self._request_time
        return self._max_dropoff_time
    
    @property
    def latest_acceptable_pickup_time(self) -> float:
        return self._request_time + self._max_waiting_time

    @property
    def is_picked(self) -> bool:
        return self._picked

    @property
    def is_assigned(self) -> bool:
        return self._assigned

    @property
    def assigned_vehicle(self) -> int:
        if self._assigned or self._picked:
            return self._vehicle
        return -1
    
    '''Request status update'''
    def updating_waiting_time(self, time):
        self._max_waiting_time = time
        
    def update_trip_value(self, value, active):
        self._trip_active += active
        self._trip_value += value
        
    def assigning(self, vid, assign_time):
        if self._picked or self._finished:
            print("ERROR! Assigning picked/finished request!")
            print(self._rid)
        self._assigned = True
        self._vehicle = vid
        self._assign_time1 = assign_time

    def picking_up(self, pickup_time):
        if not self._assigned:
            print("ERROR! Picking unassigned request!")
        if self._picked:
            print("ERROR! Picking picked up request!")

        self._assigned = False
        self._picked = True
        self._pickup_time = pickup_time

    def finishing(self, finish_time):
        if not self._picked:
            print("ERROR! Finishing unpicked up request!")
        self._picked = False
        self._vehicle = -1
        self._finished = True
        self._finish_time = finish_time
        self._dropoff_time = finish_time

    def outp(self, Detailed = True):
        print("Request[%d],Req_time[%d],LatAccTime[%d],Asgn[%r],Pick[%r],Size[%d],origin%s, destination%s" \
              % (self._rid, self._request_time, self.latest_acceptable_pickup_time, 
                 self._assigned, self._picked, self._size, self._origin, self._destination))