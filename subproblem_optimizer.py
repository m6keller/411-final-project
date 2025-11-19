from ortools.sat.python import cp_model

def generate_daily_schedule(
    day,
    surgeries,
    durations,
    surgeons,
    surgeon_of,
    deadline,
    cleaning_matrix,          # OL[i][i']
    surgeon_max_time,         # A_l^d
    max_room_minutes,
    max_positions,
    dual_pi_23,               # π_i^(2,3)
    dual_pi_5,                # π_l^d^(5)
    dual_pi_6,                # π_ltd^(6)  indexed as [surgeon][t]
    dual_pi_4,                # π_d^(4)
    time_slots                # discrete times, e.g. range(0,480)
):
    """
    Solve the subproblem for a single day and produce the best Schedule column
    with positive reduced cost.
    """

    model = cp_model.CpModel()

    # ----- Domains -----
    # W_p ∈ surgeries ∪ {0}
    W = [
        model.NewIntVarFromDomain(
            cp_model.Domain.FromValues(surgeries + [0]),
            f"W_{p}"
        ) for p in range(max_positions)
    ]

    # V_p ∈ time slots
    V = [
        model.NewIntVar(min(time_slots), max(time_slots), f"V_{p}")
        for p in range(max_positions)
    ]

    # O_i ∈ {0,1}
    O = {i: model.NewBoolVar(f"O_{i}") for i in surgeries}

    # ----- Constraint (10): no holes -----
    for p in range(max_positions - 1):
        model.Add(W[p] == 0).OnlyEnforceIf(W[p+1] == 0)

    # ----- Constraint (11): sequencing start times -----
    for p in range(max_positions - 1):
        i = W[p]
        j = W[p+1]

        # Use element constraints: V[p+1] >= V[p] + dur[i] + cleaning[i][j]
        duration_i = [durations[s] for s in surgeries + [0]]
        clean_ij = [[cleaning_matrix[a][b] for b in surgeries + [0]]
                                        for a in surgeries + [0]]

        # element(duration_i, W[p])
        d_i = model.NewIntVar(0, max_room_minutes, f"d_{p}")
        model.AddElement(W[p], duration_i, d_i)

        # cleaning time
        c_ij = model.NewIntVar(0, max_room_minutes, f"c_{p}")
        model.AddElement(W[p] * len(duration_i) + W[p+1],  # flatten pair index
                         [clean_ij[a][b] for a in range(len(duration_i))
                                         for b in range(len(duration_i))],
                         c_ij)

        model.Add(V[p+1] >= V[p] + d_i + c_ij)

    # ----- Constraint (12)-(13): count surgery appearances -----
    for i in surgeries:
        occ_list = []
        for p in range(max_positions):
            b = model.NewBoolVar(f"is_{i}_at_{p}")
            model.Add(W[p] == i).OnlyEnforceIf(b)
            model.Add(W[p] != i).OnlyEnforceIf(b.Not())
            occ_list.append(b)
        model.Add(O[i] == sum(occ_list))
        model.Add(O[i] <= 1)

    # ----- Constraint (14): surgeon max time -----
    for l in surgeons:
        model.Add(
            sum(
                O[i] * durations[i]
                for i in surgeries
                if surgeon_of[i] == l
            ) <= surgeon_max_time[l]
        )

    # ----- Constraint (15): position empty → propagate start time -----
    for p in range(1, max_positions):
        is_zero = model.NewBoolVar(f"W{p}_zero")
        model.Add(W[p] == 0).OnlyEnforceIf(is_zero)
        model.Add(W[p] != 0).OnlyEnforceIf(is_zero.Not())

        duration_prev = model.NewIntVar(0, max_room_minutes, f"durprev_{p}")
        model.AddElement(W[p-1], [durations[s] for s in surgeries + [0]], duration_prev)

        model.Add(V[p] == V[p-1] + duration_prev).OnlyEnforceIf(is_zero)

    # ================================================================
    # OBJECTIVE: Eq. (17)
    # ================================================================
    # Precompute G_i for each surgery
    G = {}
    for i in surgeries:
        ell = surgeon_of[i]
        G[i] = durations[i] - dual_pi_23[i] - durations[i] * dual_pi_5[ell]
    G[0] = 0

    # Precompute π* for each surgery i at each time t
    pi_star = {i: {} for i in surgeries}
    for i in surgeries:
        ell = surgeon_of[i]
        for t in time_slots:
            pi_star[i][t] = dual_pi_6[ell][t]

    # Build objective terms
    total_G = []
    total_pi_star = []

    for p in range(max_positions):
        # G[W[p]]
        g_p = model.NewIntVar(-10**6, 10**6, f"G_{p}")
        model.AddElement(W[p], [G[s] for s in surgeries + [0]], g_p)
        total_G.append(g_p)

        # π*(W[p]) at V[p] → element(pi_star[i], V[p])
        # create matrix: index by (surgery i, time t)
        flattened = []
        for s in surgeries + [0]:
            for t in time_slots:
                flattened.append(pi_star.get(s, {t: 0}).get(t, 0))

        idx = model.NewIntVar(0, len(flattened)-1, f"piidx_{p}")
        model.Add(idx == W[p] * len(time_slots) + (V[p] - min(time_slots)))

        pi_val = model.NewIntVar(0, 10**6, f"pi_{p}")
        model.AddElement(idx, flattened, pi_val)
        total_pi_star.append(pi_val)

    # Objective = sum(G) - sum(pi_star) - π4
    obj = (
        sum(total_G) -
        sum(total_pi_star) -
        dual_pi_4
    )
    model.Maximize(obj)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    # Reduced cost must be ≥ 0
    if solver.Value(obj) < 0:
        return None

    # Build Schedule object
    chosen = []
    surgeon_minutes = {}
    busy_times = {}

    for p in range(max_positions):
        i = solver.Value(W[p])
        if i == 0:
            continue
        chosen.append(i)

        s = surgeon_of[i]
        surgeon_minutes[s] = surgeon_minutes.get(s, 0) + durations[i]

        start = solver.Value(V[p])
        for t in range(start, start + durations[i]):
            busy_times[(s, day, t)] = 1

    # Total "profit" = Σ durations
    B_j = sum(durations[i] for i in chosen)

    schedule_id = f"{day}_col"

    return Schedule(
        schedule_id=schedule_id,
        B_j=B_j,
        day=day,
        surgeries=chosen,
        surgeon_work=surgeon_minutes,
        surgeon_busy_times=busy_times
    )
