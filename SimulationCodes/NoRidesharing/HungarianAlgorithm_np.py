import time

def improveLabels(val):
    """ change the labels, and maintain minSlack. 
    """
    for u in S:
        lu[u] -= val
    for v in V:
        if v in T:
            lv[v] += val
        else:
            minSlack[v][0] -= val

def improveMatching(v):
    """ apply the alternating path from v to the root in the tree. 
    """
    u = T[v]
    if u in Mu:
        improveMatching(Mu[u])
    Mu[u] = v
    Mv[v] = u

def slack(u,v): return lu[u]+lv[v]-w[u][v]

def augment():
    ''' augment the matching, possibly improving the lablels on the way.
    '''
    while True:
        # select edge (u,v) with u in S, v not in T and min slack
        ((val, u), v) = min([(minSlack[v], v) for v in V if v not in T])
        assert u in S
        if val>0:        
            improveLabels(val)
        # now we are sure that (u,v) is saturated
        assert slack(u,v)==0
        T[v] = u                            # add (u,v) to the tree
        if v in Mv:
            u1 = Mv[v]                      # matched edge, 
            assert not u1 in S
            S[u1] = True                    # ... add endpoint to tree 
            for v in V:                     # maintain minSlack
                p = slack(u1,v)
                if not v in T and minSlack[v][0] > p:
                    minSlack[v] = [p, u1]
        else:
            improveMatching(v)              # v is a free vertex
            return

def maxWeightMatching(weights):

    ''' given w, the weight matrix of a complete bipartite graph,
        returns the mappings Mu : U->V ,Mv : V->U encoding the matching
        as well as the value of it.
    '''
    
    global U,V,S,T,Mu,Mv,lu,lv, minSlack, w

    count_start = time.time()
    
    print("matching start:")
    w  = weights
    n  = len(w)
    m  = max(n, len(w[0]))
    U  = range(n)
    V = range(m)
    lu = [ max([w[u][v] for v in V]) for u in U]
    lv = [ 0                         for v in V]
    Mu = {}
    Mv = {}
    
    while len(Mu)<n:
        free = [u for u in V if u not in Mu]
        u0 = free[0]
        S = {u0: True}                      
        T = {}
        minSlack = [[slack(u0,v), u0] for v in V]
        augment()
    
    val = sum(lu)+sum(lv) # i.e. val. of matching is total edge zweight
    print("matching end:", time.time() - count_start)
    
    return (Mu, Mv, val)