import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Data Reading & Preprocessing ---
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
    n = 10 # Reduced for better visualization of individual routes
    nodes, dmat, p, w_pref, w_less = generate_feasible_vrptw(n_customers=n, less_pref_pad=30, seed=123)
    coords = nodes[:, 1:3] # Extract coordinates for plotting

    # --- 2. Problem and Operators ---
    problem = VRPTWPermBinary(n, dmat, p, w_pref, w_less)
    sampling = PermBinarySampling(n)
    crossover = PermBinaryCrossover(n)
    mutation = PermBinaryMutation(n, pm=0.2)

    algorithm = NSGA2(
        pop_size=100,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
        repair=problem._repair
    )

    # --- 3. Optimize and track Hypervolume ---
    n_gen = 500 # Reduced generations for faster execution
    hv = HV(ref_point=np.array([n+1, 1e4]))  # ref: all less pref, big time
    hv_history = []
    pareto_history = []

    def callback(algorithm):
        F = algorithm.pop.get("F")
        if F is not None and len(F) > 0:
            feasible = (algorithm.pop.get("CV") <= 0).flatten()
            front = F[feasible]
            if front.shape[0] > 0:
                from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
                nd_fronts = NonDominatedSorting().do(front)
                pareto_front = front[nd_fronts[0]]
                # Store the actual solutions (X) along with objectives (F)
                pareto_solutions_X = algorithm.pop[feasible][nd_fronts[0]].get("X")
                pareto_history.append((pareto_front, pareto_solutions_X))
                hv_val = hv(pareto_front)
            else:
                pareto_history.append((np.empty((0, F.shape[1])), np.empty((0, problem.n_var))))
                hv_val = 0.0
            hv_history.append(hv_val)
        else:
            pareto_history.append((np.empty((0, 2)), np.empty((0, problem.n_var))))
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


    if res.F is not None and len(res.F) > 0:
        final_front = res.F
        final_solutions = res.X

        # Sort the final front by the first objective (less preferred slots)
        sort_indices = np.argsort(final_front[:, 0])
        sorted_front = final_front[sort_indices]
        sorted_solutions = final_solutions[sort_indices]

        # --- Select Four Solutions ---
        selected_solutions_info = []

        # 1. Solution with minimum less preferred slots (edge 1)
        sol_min_less_pref_X = sorted_solutions[0]
        sol_min_less_pref_F = sorted_front[0]
        selected_solutions_info.append(
            ("Min Less Preferred Slots", sol_min_less_pref_X, sol_min_less_pref_F, 'red')
        )

        # 2. Solution with minimum total route time (edge 2)
        idx_min_time_among_sorted = np.argmin(sorted_front[:, 1])
        sol_min_time_X = sorted_solutions[idx_min_time_among_sorted]
        sol_min_time_F = sorted_front[idx_min_time_among_sorted]
        selected_solutions_info.append(
            ("Min Total Route Time", sol_min_time_X, sol_min_time_F, 'blue')
        )

        # 3. Two solutions from the middle
        num_solutions = len(sorted_solutions)
        all_selected_indices = {0, idx_min_time_among_sorted}
        candidates = [i for i in range(num_solutions) if i not in all_selected_indices]
        mid_idx_1 = None
        mid_idx_2 = None

        if len(candidates) >= 2:
            mid_idx_1 = candidates[len(candidates)//3]
            mid_idx_2 = candidates[(2*len(candidates))//3]
            if mid_idx_1 == mid_idx_2 and len(candidates) > 1: # Ensure distinct
                mid_idx_2 = candidates[min(len(candidates)-1, candidates.index(mid_idx_1) + 1)]
        elif len(candidates) == 1:
            mid_idx_1 = candidates[0]

        if mid_idx_1 is not None:
            sol_middle_1_X = sorted_solutions[mid_idx_1]
            sol_middle_1_F = sorted_front[mid_idx_1]
            selected_solutions_info.append(
                ("Middle Solution 1", sol_middle_1_X, sol_middle_1_F, 'green')
            )
        if mid_idx_2 is not None:
            sol_middle_2_X = sorted_solutions[mid_idx_2]
            sol_middle_2_F = sorted_front[mid_idx_2]
            selected_solutions_info.append(
                ("Middle Solution 2", sol_middle_2_X, sol_middle_2_F, 'purple')
            )

        selected_solutions_info = selected_solutions_info[:4]


        # Generate circular coordinates for visualization
        viz_coords = np.zeros((n + 1, 2))
        radius = 10 # Adjust as needed
        for i in range(n):
            angle = 2 * np.pi * i / n
            viz_coords[i+1, 0] = radius * np.cos(angle)
            viz_coords[i+1, 1] = radius * np.sin(angle)

        # --- Plotting in a Grid ---
        fig, axes = plt.subplots(2, 2, figsize=(15, 12)) # 2x2 grid, larger figure
        axes = axes.flatten() # Flatten the 2x2 array of axes for easy iteration

        for i, (label, sol_X, sol_F, color) in enumerate(selected_solutions_info):
            ax = axes[i] # Get the current subplot
            perm = sol_X[:n].astype(int)

            # Add 1 to permutation elements for display in title (to match 1-based customer labels)
            perm_for_display = perm + 1
            N_DISPLAY = 15 # Number of elements to display from X
            perm_str = str(perm_for_display[:N_DISPLAY])
            b_str = str(sol_X[n:][:N_DISPLAY])
            if n > N_DISPLAY:
                perm_str = perm_str[:-1] + ',...]'
                b_str = b_str[:-1] + ',...]'

            # Title includes solution type, objectives, and (truncated) X array
            ax.set_title(f"{label}\nF=[{sol_F[0]:.0f}, {sol_F[1]:.1f}] Binary={b_str}", fontsize=10)

            # Plot nodes on current subplot (Depot is at 0,0, no offset applied here)
            ax.scatter(viz_coords[0, 0], viz_coords[0, 1],
                    marker='s', color='black', s=100, label='Depot (0)', zorder=4)

            for j in range(1, n + 1):
                ax.scatter(viz_coords[j, 0], viz_coords[j, 1], marker='o', color='grey', s=60, zorder=3)
                ax.text(viz_coords[j, 0] + 0.5, viz_coords[j, 1] + 0.5, str(j), fontsize=8, ha='center', va='center')


            # Plot route segments for this solution (lines will go to/from exact center of depot)
            current_route_nodes_indices = [0] + list(perm + 1) + [0] # List of viz_coords indices for the route

            for k in range(len(current_route_nodes_indices) - 1):
                start_node_idx = current_route_nodes_indices[k]
                end_node_idx = current_route_nodes_indices[k+1]

                start_coords = viz_coords[start_node_idx]
                end_coords = viz_coords[end_node_idx]

                # No adjustment for start/end points when connecting to depot, they go directly to its center
                ax.annotate("",
                            xy=end_coords, xycoords='data',
                            xytext=start_coords, textcoords='data',
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.0",
                                            color=color, linestyle='-', lw=1.5),
                            zorder=2
                        )
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_aspect('equal', adjustable='box') # Keep aspect ratio for circular layout

        plt.tight_layout() # Adjust layout to prevent overlaps
        plt.savefig("four_separate_solutions_circular_centered_depot.png", dpi=300)
        plt.show()

    else:
        print("No feasible solutions found to plot individual routes.")
        # Retrieve the last stored Pareto front for animation if the final result has no solutions
        if res.F is None or len(res.F) == 0:
            if len(pareto_history) > 0:
                last_valid_front, _ = pareto_history[-1]
                if last_valid_front.shape[0] > 0:
                    res.F = last_valid_front
            else:
                print("No Pareto history to create animation.")
                exit()


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    # Adjust max_x and max_y based on actual F values
    all_F_values = []
    for front_tuple in pareto_history:
        if front_tuple[0].shape[0] > 0:
            all_F_values.extend(front_tuple[0].tolist())
    all_F_values = np.array(all_F_values)

    max_x = all_F_values[:,0].max() * 1.1 if all_F_values.shape[0] > 0 else n + 1
    max_y = all_F_values[:,1].max() * 1.1 if all_F_values.shape[0] > 0 else 5000
    min_x = all_F_values[:,0].min() * 0.9 if all_F_values.shape[0] > 0 else -1
    min_y = all_F_values[:,1].min() * 0.9 if all_F_values.shape[0] > 0 else -1

    max_hv = max(hv_history)*1.05 if len(hv_history) > 0 else 1.0


    def update(frame):
        ax1.clear()
        ax2.clear()

        # Pareto front
        ax1.set_xlim(min_x, max_x)
        ax1.set_ylim(min_y, max_y)
        ax1.set_xlabel("Number of less preferred slots")
        ax1.set_ylabel("Total route time")
        ax1.set_title(f"Pareto Front - Generation {frame+1}")
        current_front, _ = pareto_history[frame]
        if len(current_front) > 0:
            ax1.scatter(current_front[:,0], current_front[:,1], color='red')
        else:
            ax1.text(0.5, 0.5, "No feasible solutions", transform=ax1.transAxes, ha='center')

        # HV progress
        ax2.set_xlim(0, len(hv_history))
        ax2.set_ylim(0, max_hv)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Hypervolume")
        ax2.set_title("HV Progress")
        if frame < len(hv_history):
            ax2.plot(hv_history[:frame+1], color='blue')
            ax2.scatter([frame], [hv_history[frame]], color='red')
        else:
            ax2.text(0.5, 0.5, "No HV data", transform=ax2.transAxes, ha='center')
        return []

    n_frames = len(pareto_history)
    if n_frames > 0:
        interval_ms = 10000 / n_frames  # 10 seconds total duration
        ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=False)
        ani.save("pareto_with_hv.mp4", writer='ffmpeg', dpi=200)
        print("Saved animation as pareto_with_hv.mp4")
    else:
        print("Not enough frames for animation.")