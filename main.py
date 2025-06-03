import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
import matplotlib.pyplot as plt

# --- Data Reading & Preprocessing ---
import numpy as np

def generate_feasible_vrptw(n_customers=20, less_pref_pad=10, seed=42):
    rng = np.random.default_rng(seed)
    n = n_customers
    depot = np.array([[0.0, 0.0]])
    cust_coords = rng.uniform(-10, 10, size=(n,2))
    coords = np.vstack([depot, cust_coords])

    dmat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    p = np.full(n+1, 2.0)

    # Make a feasible "reference route": depot -> 1->2->...->n->depot
    ref_route = np.arange(n+1)  # 0,1,...,n
    arrival = [0]
    for i in range(1, n+1):
        t = arrival[-1] + dmat[ref_route[i-1], ref_route[i]] + p[ref_route[i-1]]
        arrival.append(t)

    # For customers 1..n, set wide time windows; for depot, use [0, big]
    open_early = np.maximum(0, np.array(arrival[1:]) - 5)      # len=n
    close_late = np.array(arrival[1:]) + 40                    # len=n
    w_pref = np.zeros((n+1, 2))
    w_pref[0] = [0, 1e9]                # depot window
    w_pref[1:,0] = open_early           # customers
    w_pref[1:,1] = close_late

    # Less preferred: extend both ends except depot
    w_less = w_pref.copy()
    w_less[1:,0] = np.maximum(0, w_pref[1:,0] - less_pref_pad)
    w_less[1:,1] = w_pref[1:,1] + less_pref_pad

    # Compose nodes: depot + customers
    nodes = []
    for i in range(n+1):
        # Demand is 0, not used here
        nodes.append((
            i, coords[i,0], coords[i,1], 0, w_pref[i,0], w_pref[i,1], p[i]
        ))
    nodes = np.array(nodes)

    return nodes, dmat, p, w_pref, w_less

# Example usage in your framework:
nodes, dmat, p, w_pref, w_less = generate_feasible_vrptw(n_customers=20, less_pref_pad=30, seed=123)
# Now plug these directly into yo

def parse_solomon(path, n_customers=3):
    """
    Parse Solomon data but only read the depot + first `n_customers` customers.
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    cust_start = next(i for i, l in enumerate(lines) if l.strip().startswith('CUST'))
    lines = lines[cust_start + 1:]
    nodes = []
    for line in lines:
        if line.strip() == '':
            continue
        parts = line.split()
        if len(parts) != 7: continue
        nodes.append(tuple(map(float, parts)))
        # First line is depot, then customers. Stop after n_customers+1 lines (including depot)
        if len(nodes) >= n_customers + 1:
            break
    return np.array(nodes)

def build_vrptw_data(nodes, less_pref_pad=90):
    # nodes: n+1 x 7 array (depot + n customers)
    n = len(nodes) - 1
    x = nodes[:,1]
    y = nodes[:,2]
    p = nodes[:,6]
    w_pref = np.stack([nodes[:,4], nodes[:,5]], axis=1)  # preferred time windows

    # less preferred: extend both ends, except depot
    w_less = w_pref.copy()
    w_less[1:,0] = np.maximum(0, w_pref[1:,0] - less_pref_pad)
    w_less[1:,1] = w_pref[1:,1] + less_pref_pad

    # Distances for travel time (euclidean)
    coords = np.stack([x,y], axis=1)
    dmat = np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=2)

    return n, dmat, p, w_pref, w_less

# --- Custom Problem Class ---

class VRPTWPermBinary(Problem):
    def __init__(self, n, dmat, p, w_pref, w_less):
        # Decision vars: [x permutation | b_1,...,b_n]  (0=preferred, 1=less pref)
        super().__init__(
            n_var=n+n,
            n_obj=2,
            n_constr=2*n,
            xl=np.hstack([np.zeros(n), np.zeros(n)]),
            xu=np.hstack([np.full(n, n-1), np.ones(n)]),
            type_var=np.int_
        )
        self.n = n
        self.dmat = dmat
        self.p = p
        self.w_pref = w_pref
        self.w_less = w_less

    def _evaluate(self, X, out, *args, **kwargs):
        n = self.n
        N = X.shape[0]
        objs = np.zeros((N, 2))
        g = np.zeros((N, 2*n))

        for k in range(N):
            perm = X[k,:n].astype(np.int32)
            b = X[k,n:].astype(np.int32)
            # Evaluate route: depot (0) -> perm[0] -> perm[1] ... -> perm[-1] -> depot (0)
            T = [0.0]  # T0 = 0
            for i in range(n):
                node = perm[i]+1  # node index in data (1-based for customers)
                prev = 0 if i==0 else perm[i-1]+1
                arr = T[-1] + self.dmat[prev, node] + (0 if i==0 else self.p[prev])
                window = self.w_pref if b[i]==0 else self.w_less
                l,u = window[node]
                Ti = max(l, arr)
                T.append(Ti)
                # Constraint: l <= T <= u and l <= T+service <= u
                g[k,i] = l - Ti    # T >= l  -> l - T <= 0
                g[k,i+n] = Ti + self.p[node] - u  # T+service <= u -> T+service - u <= 0

            # Final depot return
            last = perm[-1]+1
            total_time = T[-1] + self.p[last] + self.dmat[last,0]

            # Objectives:
            objs[k,0] = np.sum(b)           # Number of less preferred slots
            objs[k,1] = total_time          # Total route time

        out["F"] = objs
        out["G"] = g

    def _repair(self, problem, X, **kwargs):
        n = self.n

        # If X is a Population or Individual, extract the "X" attribute
        if hasattr(X, "__len__") and hasattr(X[0], "get"):
            # Population of Individual objects
            for ind in X:
                xvec = ind.get("X")
                perm = xvec[:n].astype(np.int32)
                missing = set(range(n)) - set(perm)
                seen = set()
                for i in range(n):
                    if perm[i] in seen or perm[i] not in range(n):
                        perm[i] = missing.pop()
                    else:
                        seen.add(perm[i])
                xvec[:n] = perm
                ind.set("X", xvec)
            return X
        else:
            # X is a numpy array, make sure it's at least 2D
            X = np.atleast_2d(X)
            for k in range(X.shape[0]):
                perm = X[k,:n].astype(np.int32)
                missing = set(range(n)) - set(perm)
                seen = set()
                for i in range(n):
                    if perm[i] in seen or perm[i] not in range(n):
                        perm[i] = missing.pop()
                    else:
                        seen.add(perm[i])
                X[k,:n] = perm
            return X if X.shape[0] > 1 else X[0]

# --- Sampling and Crossover for Perm+Binary ---

class PermBinarySampling(Sampling):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def _do(self, problem, n_samples, **kwargs):
        n = self.n
        X = np.zeros((n_samples, 2*n), dtype=int)
        for i in range(n_samples):
            X[i,:n] = np.random.permutation(n)
            X[i,n:] = np.random.randint(0,2,size=n)
        return X

class PermBinaryCrossover(Crossover):
    def __init__(self, n):
        # 2 parents, 2 offsprings
        super().__init__(2, 2)
        self.n = n

    def _do(self, problem, X, **kwargs):
        n = self.n
        n_matings = X.shape[1]
        Y = np.empty((self.n_offsprings, X.shape[1], X.shape[2]), dtype=int)
        for k in range(n_matings):
            # Permutation part: Order crossover
            p1, p2 = X[0,k,:n], X[1,k,:n]
            cut1, cut2 = sorted(np.random.choice(range(n), 2, replace=False))
            offspring1 = -np.ones(n, dtype=int)
            offspring2 = -np.ones(n, dtype=int)
            offspring1[cut1:cut2] = p1[cut1:cut2]
            offspring2[cut1:cut2] = p2[cut1:cut2]
            fill1 = [x for x in p2 if x not in offspring1]
            fill2 = [x for x in p1 if x not in offspring2]
            c=0
            for idx in list(range(0,cut1))+list(range(cut2,n)):
                offspring1[idx] = fill1[c]
                offspring2[idx] = fill2[c]
                c+=1
            # Binary part: uniform crossover
            b1, b2 = X[0,k,n:], X[1,k,n:]
            mask = np.random.rand(n) > 0.5
            b1_new = np.where(mask, b1, b2)
            b2_new = np.where(mask, b2, b1)
            Y[0,k,:n] = offspring1
            Y[1,k,:n] = offspring2
            Y[0,k,n:] = b1_new
            Y[1,k,n:] = b2_new
        return Y

class PermBinaryMutation(Mutation):
    def __init__(self, n, pm=0.2):
        super().__init__()
        self.n = n
        self.pm = pm

    def _do(self, problem, X, **kwargs):
        n = self.n
        for i in range(X.shape[0]):
            # Swap mutation on permutation
            if np.random.rand() < self.pm:
                idx1, idx2 = np.random.choice(n, 2, replace=False)
                X[i,idx1], X[i,idx2] = X[i,idx2], X[i,idx1]
            # Bitflip on binary
            mask = np.random.rand(n) < self.pm
            X[i,n:][mask] = 1-X[i,n:][mask]
        return X

# --- Run Optimization ---

if __name__ == '__main__':
    # --- 1. Data ---
    n = 30
    nodes, dmat, p, w_pref, w_less = generate_feasible_vrptw(n_customers=n, less_pref_pad=30, seed=123)

    # --- 2. Problem and Operators ---
    problem = VRPTWPermBinary(n, dmat, p, w_pref, w_less)
    sampling = PermBinarySampling(n)
    crossover = PermBinaryCrossover(n)
    mutation = PermBinaryMutation(n, pm=0.2)

    algorithm = NSGA2(
        pop_size=200,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
        repair=problem._repair
    )

    # --- 3. Optimize and track Hypervolume ---
    n_gen = 1500
    hv = HV(ref_point=np.array([n, 5000]))  # ref: all less pref, big time
    hv_history = []
pareto_history = []

# Set your reference point larger than any expected [objective1, objective2]
hv = HV(ref_point=np.array([n+1, 1e4]))

def callback(algorithm):
    F = algorithm.pop.get("F")
    if F is not None and len(F) > 0:
        feasible = (algorithm.pop.get("CV") <= 0).flatten()
        front = F[feasible]
        if front.shape[0] > 0:
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
            nd_fronts = NonDominatedSorting().do(front)
            pareto_front = front[nd_fronts[0]]
            pareto_history.append(pareto_front)
            hv_val = hv(pareto_front)
        else:
            pareto_front = np.empty((0, F.shape[1]))
            pareto_history.append(pareto_front)
            hv_val = 0.0
        hv_history.append(hv_val)
    else:
        pareto_history.append(np.empty((0, 2)))
        hv_history.append(0.0)

res = minimize(
    problem,
    algorithm,
    ("n_gen", n_gen),
    seed=42,
    verbose=True,
    callback=callback
)

# --- Plot hypervolume progress (final curve) ---
plt.figure()
plt.plot(hv_history)
plt.xlabel("Generation")
plt.ylabel("Hypervolume")
plt.title("NSGA-II Hypervolume Progress")
plt.savefig("hv_progress.png", dpi=200)

# --- Plot final Pareto front ---
if res.F is not None:
    plt.figure()
    plt.scatter(res.F[:,0], res.F[:,1])
    plt.xlabel("Number of less preferred slots")
    plt.ylabel("Total route time")
    plt.title("Final Pareto front")
    plt.savefig("final_pareto.png", dpi=200)


# --- Animated video: left=Pareto, right=HV progress ---
import matplotlib.animation as animation

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
max_x = max([front[:,0].max() if len(front)>0 else 0 for front in pareto_history]) + 1
max_y = max([front[:,1].max() if len(front)>0 else 0 for front in pareto_history]) + 10
max_hv = max(hv_history)*1.05

def update(frame):
    ax1.clear()
    ax2.clear()
    # Pareto front
    ax1.set_xlim(0, max_x)
    ax1.set_ylim(0, max_y)
    ax1.set_xlabel("Number of less preferred slots")
    ax1.set_ylabel("Total route time")
    ax1.set_title(f"Pareto Front - Generation {frame+1}")
    front = pareto_history[frame]
    if len(front) > 0:
        ax1.scatter(front[:,0], front[:,1], color='red')
    # HV progress
    ax2.set_xlim(0, len(hv_history))
    ax2.set_ylim(0, max_hv)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Hypervolume")
    ax2.set_title("HV Progress")
    ax2.plot(hv_history[:frame+1], color='blue')
    ax2.scatter([frame], [hv_history[frame]], color='red')
    return []

n_frames = len(pareto_history)
interval_ms = 10000 / n_frames  # 10 seconds total duration

ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=False)
ani.save("pareto_with_hv.mp4", writer='ffmpeg', dpi=200)
print("Saved animation as pareto_with_hv.mp4")
