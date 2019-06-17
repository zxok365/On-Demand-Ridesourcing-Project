'''Given vehicle and a request list, the following checks its trip feasibility and finds optimal route'''

import searching
from vehicle import *
from Waypoint import MapSystem
eps = 1e-7


'''Given vehicle and a request list, the following checks its trip feasibility and finds optimal route'''
def calc_min_routing(v: Vehicle, r: list, cur_time, map: MapSystem, cur_network: np.ndarray):
    Todo = []
    min_time_cost = 0

    for x in v.boarded_requests:
        Todo += [(x.get_destination_id, x.latest_dropoff_time(map), x.get_id, 1, 1)]
        min_time_cost += searching.distance(v.get_point_id, x.get_destination_id, cur_network)

    if r == -1:
        feasible, min_cost, routing, end_time, idle_time = searching.check_feasible(cur_time=cur_time, cur_pos=v.get_point_id,
                                                                          dst_list=Todo, now_cost=0. - min_time_cost,
                                                                          time_cost_network=cur_network)

        if not feasible:
            min_cost, routing, end_time, idle_time = searching.calculate_minimum_cost(cur_time=cur_time, cur_pos=v.get_point_id,
                                                                            dst_list=Todo, now_cost=0. - min_time_cost,
                                                                            time_cost_network=cur_network)

        totel_length = end_time - cur_time

        return feasible, routing, min_cost, totel_length, idle_time
    else:
        for req in r:
            lat_pu_time = req.latest_acceptable_pickup_time

            if lat_pu_time + eps < cur_time + searching.distance(v.get_point_id, req.get_origin_id, cur_network):
                return False, -1, -1, -1, -1

            Todo += [(req.get_origin_id, lat_pu_time, req.get_id, 0, 1)]
            Todo += [(req.get_destination_id, req.latest_dropoff_time(map), req.get_id, 1, 1)]
            min_time_cost += searching.distance(req.get_origin_id, req.get_destination_id, cur_network)

        feasible, min_cost, routing, end_time, idle_time = searching.check_feasible(cur_time=cur_time, cur_pos=v.get_point_id,
                                                                          dst_list=Todo, now_cost=0. - min_time_cost,
                                                                          time_cost_network=cur_network)
        totel_length = end_time - cur_time

        return feasible, routing, min_cost, totel_length, idle_time