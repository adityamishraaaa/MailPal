# Name: Aditya Mishra 
# Roll Number - 21013

import networkx as nx
import math

# Campus coordinates

coords = {
    'mailroom': (0, 0),
    'main':     (1000, 700),
    'sports':   (700, 200),
    'acad1':    (440, 500),
    'acad2':    (400, 520),
    'hostel1':  (500, 50),
    'hostel2':  (520, 100),
    'hostel3':  (550, 50),
}

# Build a complete graph with Euclidean weights
G = nx.Graph()
for u, (x1, y1) in coords.items():
    for v, (x2, y2) in coords.items():
        if u == v:
            continue
        d = math.hypot(x2 - x1, y2 - y1)
        G.add_edge(u, v, weight=d)

def dist(u, v):
    """Euclidean distance on our campus graph."""
    if u == v:
        return 0
    return G[u][v]['weight']

# TSP approximation via MST doubling (2-approx)
def approx_tsp(nodes):
    subG = G.subgraph(nodes).copy()
    for u in nodes:
        for v in nodes:
            if u != v and not subG.has_edge(u, v):
                subG.add_edge(u, v, weight=dist(u, v))
    T = nx.minimum_spanning_tree(subG, weight='weight')
    order = list(nx.dfs_preorder_nodes(T, source='mailroom'))
    return order + ['mailroom']

def tour_cost(tour):
    return sum(dist(tour[i], tour[i+1]) for i in range(len(tour)-1))

# Split a tour into k subtours of roughly equal length
def split_tour(tour, k):
    edges = [(tour[i], tour[i+1]) for i in range(len(tour)-1)]
    lengths = [dist(u, v) for u, v in edges]
    total = sum(lengths)
    target = total / k
    subtours, current, acc, next_cut = [], ['mailroom'], 0, target

    for i, (u, v) in enumerate(edges):
        acc += lengths[i]
        current.append(v)
        if acc >= next_cut and len(subtours) < k-1:
            current.append('mailroom')
            subtours.append(current)
            current, next_cut = ['mailroom'], next_cut + target

    current.append('mailroom')
    subtours.append(current)
    while len(subtours) < k:
        subtours.append(['mailroom','mailroom'])
    return subtours

# Heterogeneous Split (feasibility check for given λ)
def hetero_split(agents, Ti, T0, lam):
    alloc = {a['id']: [] for a in agents}
    cost_alloc = {a['id']: 0 for a in agents}

    # Phase 1: Type-specific allocation
    for ttype in Ti:
        agents_of_type = [a for a in agents if a['type'] == ttype]
        if not agents_of_type or not Ti[ttype]:
            continue
        nodes = ['mailroom'] + [t['loc'] for t in Ti[ttype]]
        tour = approx_tsp(nodes)
        subtours = split_tour(tour, len(agents_of_type))
        for agent, subt in zip(agents_of_type, subtours):
            locs = subt[1:-1]
            alloc[agent['id']] = locs
            cost_alloc[agent['id']] = tour_cost(subt)

    # Phase 2: Generic allocation
    free_agents = set(a['id'] for a in agents)
    nodes0 = ['mailroom'] + [t['loc'] for t in T0]
    tour0 = approx_tsp(nodes0)
    remaining = T0.copy()
    ordered_tasks = []
    for loc in tour0:
        for t in remaining:
            if t['loc'] == loc and t not in ordered_tasks:
                ordered_tasks.append(t)
    idx = 0
    while idx < len(ordered_tasks) and free_agents:
        task = ordered_tasks[idx]
        best_agent, best_cost = None, float('inf')
        for aid in free_agents:
            new_locs = alloc[aid] + [task['loc']]
            nodes_new = ['mailroom'] + list(dict.fromkeys(new_locs))
            c = tour_cost(approx_tsp(nodes_new))
            if c <= lam and c < best_cost:
                best_agent, best_cost = aid, c
        if best_agent is None:
            return None
        while idx < len(ordered_tasks):
            t2 = ordered_tasks[idx]
            new_locs = alloc[best_agent] + [t2['loc']]
            nodes_new = ['mailroom'] + list(dict.fromkeys(new_locs))
            c2 = tour_cost(approx_tsp(nodes_new))
            if c2 <= lam:
                alloc[best_agent].append(t2['loc'])
                cost_alloc[best_agent] = c2
                idx += 1
            else:
                break
        free_agents.remove(best_agent)

    # Phase 3: Rebalance within each type (with λ check)
    for ttype in Ti:
        agents_of_type = [a for a in agents if a['type'] == ttype]
        if not agents_of_type:
            continue
        pooled = []
        for a in agents_of_type:
            pooled += alloc[a['id']]
        nodes_pooled = ['mailroom'] + list(dict.fromkeys(pooled))
        tour_p = approx_tsp(nodes_pooled)
        subt = split_tour(tour_p, len(agents_of_type))

        # Verify no subtour exceeds λ
        for sub in subt:
            if tour_cost(sub) > lam:
                return None

        for agent, sub in zip(agents_of_type, subt):
            alloc[agent['id']] = sub[1:-1]
            cost_alloc[agent['id']] = tour_cost(sub)

    return alloc, cost_alloc

# HeteroMinMaxSplit: binary search on λ
def hetero_min_max_split(agents, Ti, T0):
    dmax = max(dist('mailroom', loc) for loc in coords if loc != 'mailroom')
    lo, hi = 2 * dmax, 0

    if T0:
        nodes0 = ['mailroom'] + [t['loc'] for t in T0]
        hi += tour_cost(approx_tsp(nodes0))
    for ttype in Ti:
        if Ti[ttype]:
            nodesi = ['mailroom'] + [t['loc'] for t in Ti[ttype]]
            hi = max(hi, tour_cost(approx_tsp(nodesi)))

    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        res = hetero_split(agents, Ti, T0, mid)
        if res:
            best = (mid, res)
            hi = mid - 1
        else:
            lo = mid + 1
    return best

# Example Case
if __name__ == "__main__":
    # Prompt for agent counts
    n_students = int(input("Enter number of student agents: "))
    n_vehicle  = int(input("Enter number of vehicle staff agents: "))
    n_drones   = int(input("Enter number of drone agents: "))

    # Building agents list
    agents = []
    for i in range(1, n_students + 1):
        agents.append({'id': f'student{i}', 'type': 0})
    for i in range(1, n_vehicle + 1):
        agents.append({'id': f'vehicle{i}', 'type': 1})
    for i in range(1, n_drones + 1):
        agents.append({'id': f'drone{i}', 'type': 2})

 
    tasks = [
        {'id': 'gen1', 'loc': 'hostel1', 'type': None},
        {'id': 'gen2', 'loc': 'hostel2', 'type': None},
        {'id': 'gen3', 'loc': 'hostel3', 'type': None},
        {'id': 'gen4', 'loc': 'main',    'type': None},
        {'id': 'hp1',  'loc': 'acad1',   'type': 1},
        {'id': 'hp2',  'loc': 'acad2',   'type': 1},
        {'id': 'ud1',  'loc': 'main',    'type': 2},
    ]

    # Partition tasks into Ti (type‐specific) and T0 (generic)
    Ti = {0: [], 1: [], 2: []}
    T0 = []
    for t in tasks:
        if t['type'] is None:
            T0.append(t)
        else:
            Ti[t['type']].append(t)

    # Running the algorithm
    result = hetero_min_max_split(agents, Ti, T0)
    if result:
        lam_opt, (alloc, cost_alloc) = result
        print(f"\nOptimal max-tour bound λ = {lam_opt:.2f} m\n")
        for aid in alloc:
            route = " -> ".join(alloc[aid]) if alloc[aid] else "(no stops)"
            print(f"Agent {aid}:")
            print(f"  Route: mailroom -> {route} -> mailroom")
            print(f"  Total distance: {cost_alloc[aid]:.2f} m\n")
    else:
        print("No feasible allocation found under given constraints.")
