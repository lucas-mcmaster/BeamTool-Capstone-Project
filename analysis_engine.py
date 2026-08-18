import numpy as np

class BeamAnalysis:
    """
    Performs structural analysis using Singularity Functions (Macaulay's Method).
    Returns Shear, Moment, and Deflection arrays for plotting.
    """
    def __init__(self, length, E, I):
        self.L = length
        self.E = E
        self.I = I
        self.loads = []
        self.supports = []

    def add_load(self, type, magnitude, start, end=None):
        """
        type: 'point', 'distributed', 'moment'
        magnitude: Positive (+) is DOWN for forces, CCW for moments.
        """
        self.loads.append({'type': type, 'mag': magnitude, 'a': start, 'b': end})

    def add_support(self, type, location):
        """type: 'pin', 'roller', 'fixed'"""
        self.supports.append({'type': type, 'loc': location})

    def macaulay(self, x, a, n):
        """
        Macaulay bracket function <x-a>^n.
        Handles the discontinuity of loads.
        """
        return np.where(x >= a, (x - a)**n, 0)

    def solve_reactions(self):
        """
        Solves for unknown reactions using a system of linear equations (Ax = B).
        Unknowns: Vertical forces (Ry) and Moments (M) at supports, plus integration constants C1, C2.
        Equations: Fy=0, M=0, Boundary Conditions (deflection/slope at supports).
        """
        # Identify Unknowns
        unknowns = []
        for i, sup in enumerate(self.supports):
            if sup['type'] in ['pin', 'roller', 'fixed']:
                unknowns.append(('Ry', i))
            if sup['type'] == 'fixed':
                unknowns.append(('M_sup', i))

        num_reactions = len(unknowns)
        total_vars = num_reactions + 2 # We also have 2 integration constants C1 (slope) and C2 (deflection)

        # Matrix A (Coefficients) and Vector B (Constants)
        A = np.zeros((total_vars, total_vars))
        B = np.zeros(total_vars)

        # --- Equilibrium Equations ---
        # --- EQ 1: Sum of Vertical Forces = 0 ---
        # Sum(Ry) = Sum(External Loads)
        row = 0
        for idx, (u_type, _) in enumerate(unknowns):
            if u_type == 'Ry': A[row, idx] = 1

        # Calculating sum of external loads (move to RHS)
        load_force_sum = 0
        for load in self.loads:
            if load['type'] == 'point': load_force_sum += load['mag']
            elif load['type'] == 'distributed':
                load_force_sum += load['mag'] * (load['b'] - load['a'])
        B[row] = load_force_sum

        #--- EQ 2: Sum of Moments about x=0 = 0 ---
        # Sum(Ry * x) + Sum(M_reaction) = Sum(Load Moments)
        row = 1
        for idx, (u_type, sup_idx) in enumerate(unknowns):
            loc = self.supports[sup_idx]['loc']
            if u_type == 'Ry': A[row, idx] = loc
        # We solve for INTERNAL Moment at the support to ensure correct diagram shape.
        # Coefficient -1 ensures M_sup represents the Internal Restraining Moment.
            elif u_type == 'M_sup': A[row, idx] = -1

        # Calculating sum of external load moments
        load_moment_sum = 0
        for load in self.loads:
            if load['type'] == 'point':
                load_moment_sum += load['mag'] * load['a']
            elif load['type'] == 'distributed':
                w = load['mag']
                length = load['b'] - load['a']
                centroid = load['a'] + length/2
                load_moment_sum += (w * length) * centroid
            elif load['type'] == 'moment':
                load_moment_sum += load['mag']
        B[row] = load_moment_sum

        # --- Boundary Conditions ---
        # Deflection y = 0 at pins/rollers/fixed. Slope theta = 0 at fixed.
        current_row = 2
        for i, sup in enumerate(self.supports):
            loc = sup['loc']

            # BC 1: Deflection y(loc) = 0 (Applicable to Pin, Roller, Fixed)
            # EI*y = [Reaction Terms] + [Load Terms] + C1*x + C2
            # 0 = [Reaction Terms] + [Load Terms] + C1*x + C2
            # [Reaction Terms] + C1*x + C2 = -[Load Terms]
            A[current_row, num_reactions] = loc  # coefficient for C1
            A[current_row, num_reactions+1] = 1  # coefficient for C2

            # Filling Reaction coefficients (Singularity function terms for deflection)
            for idx, (u_type, sup_idx) in enumerate(unknowns):
                sup_loc = self.supports[sup_idx]['loc']
                if u_type == 'Ry':
                    #(Ry / 6) <x-a>^3
                    A[current_row, idx] = (1/6) * self.macaulay(loc, sup_loc, 3)
                elif u_type == 'M_sup':
                    #(M / 2) <x-a>^2
                    A[current_row, idx] = (1/2) * self.macaulay(loc, sup_loc, 2)

            #Calculating Load terms (RHS)
            rhs_val = 0
            for load in self.loads:
                if load['type'] == 'point':
                    # (P/6) <x-a>^3
                    rhs_val += (load['mag']/6) * self.macaulay(loc, load['a'], 3)
                elif load['type'] == 'distributed':
                    #(w/24) <x-a>^4 + (w/24) <x-b>^4
                    rhs_val += (load['mag']/24) * self.macaulay(loc, load['a'], 4)
                    rhs_val -= (load['mag']/24) * self.macaulay(loc, load['b'], 4)
                elif load['type'] == 'moment':
                    #(M/2)<x-a>^2
                    rhs_val += (load['mag']/2) * self.macaulay(loc, load['a'], 2)
            B[current_row] = rhs_val
            current_row += 1

            # BC 2: Slope = 0 (Fixed supports only)
            # EI*theta = [Reaction Terms] - [Load Terms] + C1 = 0
            if sup['type'] == 'fixed':
                A[current_row, num_reactions] = 1 # C1 coeff
                A[current_row, num_reactions+1] = 0 # C2 coeff

                for idx, (u_type, sup_idx) in enumerate(unknowns):
                    sup_loc = self.supports[sup_idx]['loc']
                    if u_type == 'Ry': A[current_row, idx] = (1/2) * self.macaulay(loc, sup_loc, 2)
                    elif u_type == 'M_sup': A[current_row, idx] = 1 * self.macaulay(loc, sup_loc, 1)

                rhs_val = 0
                for load in self.loads:
                    if load['type'] == 'point':
                        rhs_val += (load['mag']/2) * self.macaulay(loc, load['a'], 2)
                    elif load['type'] == 'distributed':
                        rhs_val += (load['mag']/6) * self.macaulay(loc, load['a'], 3)
                        rhs_val -= (load['mag']/6) * self.macaulay(loc, load['b'], 3)
                    elif load['type'] == 'moment':
                        rhs_val += load['mag'] * self.macaulay(loc, load['a'], 1)
                B[current_row] = rhs_val
                current_row += 1

        try:
            x_sol = np.linalg.solve(A, B)
            return x_sol, unknowns
        except np.linalg.LinAlgError:
            return None, None

    def analyze(self, num_points=2001):
        sol, unknown_map = self.solve_reactions()
        if sol is None: return {"error": "Unstable Structure"}

        C1 = sol[-2]
        C2 = sol[-1]
        x = np.linspace(0, self.L, num_points)

        # Initialize arrays
        shear = np.zeros_like(x)
        moment = np.zeros_like(x)
        deflection = np.zeros_like(x) + C1*x + C2

        # 1. Add Reaction Effects
        for idx, (u_type, sup_idx) in enumerate(unknown_map):
            val = sol[idx]
            a = self.supports[sup_idx]['loc']

            if u_type == 'Ry':
                # Shear: Ry <x-a>^0
                shear += val * self.macaulay(x, a, 0)
                # Moment: Ry <x-a>^1
                moment += val * self.macaulay(x, a, 1)
                # Deflection: (Ry/6) <x-a>^3
                deflection += (val/6) * self.macaulay(x, a, 3)

            elif u_type == 'M_sup':
                # Shear: Moment reaction does NOT affect shear
                # Moment: +M_sup <x-a>^0
                moment += val * self.macaulay(x, a, 0)
                # Deflection: (M_sup/2) <x-a>^2
                deflection += (val/2) * self.macaulay(x, a, 2)

        # 2. Add Load Effects (Subtracting)
        for load in self.loads:
            a = load['a']
            mag = load['mag']

            if load['type'] == 'point':
                # Shear: -P <x-a>^0
                shear -= mag * self.macaulay(x, a, 0)
                # Moment: -P <x-a>^1
                moment -= mag * self.macaulay(x, a, 1)
                # Deflection: -(P/6) <x-a>^3
                deflection -= (mag/6) * self.macaulay(x, a, 3)

            elif load['type'] == 'distributed':
                b = load['b']
                # Shear: -w<x-a>^1 + w<x-b>^1
                shear -= mag * self.macaulay(x, a, 1)
                shear += mag * self.macaulay(x, b, 1)

                # Moment: -(w/2)<x-a>^2 + (w/2)<x-b>^2
                moment -= (mag/2) * self.macaulay(x, a, 2)
                moment += (mag/2) * self.macaulay(x, b, 2)

                # Deflection
                deflection -= (mag/24) * self.macaulay(x, a, 4)
                deflection += (mag/24) * self.macaulay(x, b, 4)

            elif load['type'] == 'moment':
                # Shear: No effect
                # Moment: -M_app <x-a>^0
                moment -= mag * self.macaulay(x, a, 0)
                deflection -= (mag/2) * self.macaulay(x, a, 2)

        # Final Scaling
        deflection = deflection / (self.E * self.I)

        return {
            "x": x.tolist(),
            "shear": shear.tolist(),
            "moment": moment.tolist(),
            "deflection": deflection.tolist(),
            "max_shear": np.max(np.abs(shear)),
            "max_moment": np.max(np.abs(moment)),
            "max_deflection": np.max(np.abs(deflection))
        }

def analyze_custom_beam(user_input):
    analyzer = BeamAnalysis(user_input['length'], user_input['E'], user_input['I'])
    # Add Self-Weight
    gravity = 9.81
    self_weight_w = user_input.get('beam_weight', 0) * gravity
    if self_weight_w > 0:
        analyzer.add_load('distributed', self_weight_w, 0, user_input['length'])
    
    for load in user_input['loads']:
        analyzer.add_load(load['type'], load['mag'], load['a'], load.get('b'))
    for sup in user_input['supports']:
        analyzer.add_support(sup['type'], sup['loc'])
        
    return analyzer.analyze()