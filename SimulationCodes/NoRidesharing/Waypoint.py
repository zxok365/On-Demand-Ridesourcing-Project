eps = 1e-4

WANDERING_DIST = 300
import random

'''The following records the position of the request/vehicle, as they leave node A and move towards B'''
class WayPoint:
    def __init__(self, A_id : int = -1, B_id : int = -1, dis_to_A : float = -1, dis_to_B : float = -1):
        self.A_id = A_id
        self.B_id = B_id
        self.A_pid = 0
        self.B_pid = 0
        self.dis_to_A = dis_to_A
        self.dis_to_B = dis_to_B

    def copy(self, copy_point : "WayPoint"):
        self.A_id = copy_point.A_id
        self.B_id = copy_point.B_id
        self.A_pid = copy_point.A_pid
        self.B_pid = copy_point.B_pid
        self.dis_to_A = copy_point.dis_to_A
        self.dis_to_B = copy_point.dis_to_B

'''To deal with the Map/Road Data'''
class MapSystem:
    # node_dict: node_id -> position(Long, lat)
    # distance( <node_id, node_id> -> dist
    # nearby_node: for each node i: list[i] = list (nodes that is near i)
    def __init__(self, node_dict : dict = {}, distance : dict = {}, nearby_node : dict = {}):
        self._node_dict = node_dict
        self._node_distance = distance
        self._nearby_node = nearby_node
        self._hour = -1
        self._cnt = 0
        self._node_label = {}
        self._point_label = []
        self._point_distance = []

    '''update the MapSystem to the newest version'''
    def update_distance(self, cur_time : float, distance : dict):
        if self._hour != int(cur_time // 3600) % 24:
            self._node_distance = distance
            self._hour = int(cur_time // 3600) % 24
            self._cnt = 0
            self._node_label = {}
            self._point_label = [0 for x in range(40000)]
            self._point_distance = []

    def random_waypoint(self, random_seed = 0):
        random.seed(random_seed)
        x = random.choice(list(self._node_dict))
        return WayPoint(x, x, 0, 0)

    def random_nearby_waypoint(self, waypoint,random_seed = 0):
        if waypoint.A_id != waypoint.B_id:
            return WayPoint(waypoint.B_id, waypoint.B_id, 0, 0)
        else:
            random.seed(random_seed)
            x = random.choice(list(self._nearby_node))
            return WayPoint(x, x, 0, 0)

    def update_waypoint(self, p : WayPoint):
        if not self._node_label.get(p.A_id):
            for i in range(0, self._cnt):
               x = self._point_label[i]
               self._point_distance[i] += [self._node_distance[(x, p.A_id)]]

            self._node_label[p.A_id] = self._cnt
            self._point_label[self._cnt] = p.A_id
            self._cnt += 1
            self._point_distance += [[ self._node_distance[p.A_id, self._point_label[x]] for x in range(0, self._cnt) ]]

        if not self._node_label.get(p.B_id):
            for i in range(0, self._cnt):
                x = self._point_label[i]
                self._point_distance[i] += [self._node_distance[(x, p.B_id)]]

            self._node_label[p.B_id] = self._cnt
            self._point_label[self._cnt] = p.B_id
            self._cnt += 1
            self._point_distance += [[self._node_distance[p.B_id, self._point_label[x]] for x in range(0, self._cnt)]]

        p.A_pid = self._node_label[p.A_id]
        p.B_pid = self._node_label[p.B_id]

    '''return the position of the "waypoint"'''
    def get_position(self, waypoint : WayPoint) -> (float, float):
        A_x, A_y = self._node_dict.get(waypoint.A_id)
        B_x, B_y = self._node_dict.get(waypoint.B_id)
        total_distance= waypoint.dis_to_A + waypoint.dis_to_B
        
        if total_distance != 0:
            return ((A_x * waypoint.dis_to_B + B_x * waypoint.dis_to_A) / total_distance, (A_y * waypoint.dis_to_B + B_y * waypoint.dis_to_A) / total_distance)
        else:
            return (A_x, A_y)

    '''return the distance from WayPoint 'p' to WayPoint 'q', note that it's different from distance(q,p)'''
    def distance(self, p : WayPoint, q: WayPoint) -> float:

            #Already on the same road AND still not reach
            if p.A_id == q.A_id and p.B_id == q.B_id and p.dis_to_A < q.dis_to_A + eps:
                return max(q.dis_to_A - p.dis_to_A, 0.)
            if self._point_label[p.B_pid] != p.B_id:
                self.update_waypoint(p)

            if self._point_label[q.A_pid] != q.A_id:
                self.update_waypoint(q)
            #not on the same road OR miss
            return p.dis_to_B + q.dis_to_A + self._point_distance[p.B_pid][q.A_pid]



    def move_to_destination_node(self, Origin : WayPoint, Destination : int, moving_time : float) -> (bool, float):
        return self.move_to_destination(Origin,WayPoint(Destination,Destination,0,0), moving_time)

    ''' Try to move Origin to Destinition by (moving_time) seconds
        return reach the remaining time if reached
        The Original Waypoint would also be updated'''
    def move_to_destination(self, Origin : WayPoint, Destination : WayPoint, moving_time : float) -> (bool, float):
        
        X = self.distance(Origin, Destination)
        
        if X - eps < moving_time : #i.e. if the destination node can be reached
            Origin.copy(Destination)
            return (True, max(0, moving_time - X)) #i.e. reaching after X secs
        
        #if destination node cannot be reached,
        if moving_time < Origin.dis_to_B: 
            Origin.dis_to_B -= moving_time
            Origin.dis_to_A += moving_time
            return (False, 0)
        else:
            tmp_t = moving_time - Origin.dis_to_B

            Origin.A_id = Origin.B_id
            Origin.dis_to_A = 0
            Origin.dis_to_B = 0

            nxt_node = -1
            min_dis = self._node_distance[(Origin.A_id, Destination.A_id)]
            
            #Already at the start_point of Destination:
            if (Origin.A_id == Destination.A_id):
                nxt_node = Destination.B_id
            else:
                for tmp_node in self._nearby_node[Origin.A_id]:
                    if (tmp_node != Origin.A_id and self._node_distance[(Origin.A_id, tmp_node)] + self._node_distance[(tmp_node, Destination.A_id)] <= min_dis + eps):
                        nxt_node = tmp_node
                        break
            Origin.B_id = nxt_node

            Origin.dis_to_B = self._node_distance[(Origin.A_id, nxt_node)]
            return self.move_to_destination(Origin, Destination, tmp_t)

    '''Generate vehicles' initial position randomly'''
    def GEN_START_POINT(self, id, dis, random_seed = 0) -> WayPoint:
        for x in self._node_dict:
            for y in self._nearby_node[x]:
                if y == id:
                    ran = random.random() * dis
                    return WayPoint(y, x, self._node_distance[x, y] - ran, ran)