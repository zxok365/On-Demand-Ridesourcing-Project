import numpy as np

import searching
from request import Request
from Waypoint import WayPoint

class Vehicle:
    def __init__(self,vehicle_id : int= -1, cur_position : tuple = (0,0), capacity : int = 1, boarded : list = list(), 
                 assigned : list = list(), dst_pos : tuple = None, position_id = -1, waypoint : WayPoint = WayPoint(), overall_time = 0):
        
        self._vid = vehicle_id
        self._arr_time = 0 #i.e. all vehicles are assumed to be initialized at time 0
        self._init_position = cur_position
        self._point_id = position_id
        self._waypoint = waypoint
        self._capacity = capacity
        
        '''Per-round Allocation related attributes'''
        self._cur_time = -1
        self._cur_position = cur_position #i.e. current location of vehicles, to be updated per round
        self._available = None
        
        self._boarded_requests = list(boarded)
        self._assigned_requests = list(assigned) #i.e. request list assigned before picked up, remove from list after pickup
        self._dst_position = dst_pos
        self._overall_time = overall_time
        
        '''Recording accumulated trips'''
        self._allocated_value1 = 0
        self._active_timecost = 0
        
    @property
    def isidle(self) -> bool: #i.e. checking if there are any assigned or boarded
        if self._boarded_requests or self._assigned_requests or self._dst_position != None:
            return False
        else:
            return True

    def busy(self, cur_time, time_cost_network : np.ndarray) -> bool:
        if self._cur_time == cur_time:
            return not self._available
        else:
            self.available(cur_time, time_cost_network)
            return not self._available

    def available(self, cur_time, time_cost_network : np.ndarray) -> bool:

        if self._cur_time == cur_time:
            return self._available

        if self.empty_space == 0:
            return False

        waiting_list = []
        min_time_cost = 0

        for x in self._boarded_requests:
            waiting_list += [(x.get_destination_id, x.latest_dropoff_time(time_cost_network), x.get_id, 1, x._size)]
            min_time_cost += x._size * searching.distance(self.get_point_id, x.get_destination_id,time_cost_network)

        for x in self._assigned_requests:
            waiting_list += [(x.get_origin_id, x.latest_acceptable_pickup_time, x.get_id, 0, x._size)]
            waiting_list += [(x.get_destination_id, x.latest_dropoff_time(time_cost_network), x.get_id, 1, x._size)]
            min_time_cost += x._size * x.min_time_cost(time_cost_network)

        feasible, time_cost, routing = searching.check_feasible(cur_time=cur_time, cur_pos=self.get_point_id,dst_list=waiting_list, time_cost_network=time_cost_network, now_cost=0-min_time_cost)

        self._cur_time = cur_time
        self._available = feasible

        return feasible

    @property
    def get_id(self) -> int:
        return self._vid

    @property
    def get_point_id(self):
        return self._point_id

    def set_point_id(self, SET : int):
        self._point_id = SET

    @property
    def cur_position(self):
        return self._cur_position

    @property
    def boarded_requests(self):
        return self._boarded_requests

    @property
    def empty_space(self) -> int: #i.e. remaining space in the vehicle
        
        num = 0
        for r in self._boarded_requests:
            num = num + r._size
        
        return self._capacity - num

    def add_request(self, request : Request) -> bool: #i.e. adding request into vehicle; if success return True
        pass

    def add_destination(self, destination : tuple) -> bool: #i.e. adding destination for vacant (posssibly vacant) vehicle
        if self._dst_position != None:
            return False
        else:
            self._dst_position = destination
            return True

    '''Vehicle status update'''
    def new_assigning(self):
        self._assigned_requests = []

    def assigning(self, req : Request):
        self._assigned_requests.append(req)

    def picking_up(self, req : Request):
        X = self._assigned_requests.count(req)
        if X <= 0:
            print("ERROR! Picking up a non-existent request!")
        else:
            self._assigned_requests.remove(req)
        self._boarded_requests.append(req)

    def finishing(self, req : Request):
        X = self._boarded_requests.count(req)
        if X <= 0:
            print("ERROR! Finishing a non-existent request!")
        else:
            self._boarded_requests.remove(req)

    def update_allocated(self, incr1):
        self._allocated_value1 += incr1
    
    '''Output'''
    def outp(self, Detailed=False):
        print("Vehicle[%d],CurPos%s,Cap[%d],Empty[%d]" \
              % (self.get_id, self._cur_position, self._capacity, self.empty_space))