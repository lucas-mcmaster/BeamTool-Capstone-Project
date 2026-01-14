import pandas as pd
import ast
from analysis_engine import BeamAnalysis

class ComparatorEngine:
    def __init__(self, csv_file_path):
        """
        Initializes the engine by loading beam data from a CSV file.
        
        :param csv_file_path: Path to the .csv file containing beam data.
        """
        # 1. Load Data
        self.beam_library = pd.read_csv(csv_file_path)
        
        # 2. Data Cleaning
        # The CSV stores lists as strings (e.g. "['Indoor', 'Seismic']").
        # We must convert them back to actual Python lists for the logic to work.
        if 'env_suitability' in self.beam_library.columns:
            self.beam_library['env_suitability'] = self.beam_library['env_suitability'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

    def recommend(self, user_reqs):
        """
        user_reqs: Dict containing mandatory and optional inputs.
        MANDATORY: 'length', 'loads', 'supports', 'environment'
        OPTIONAL: 'desired_material', 'max_cost', 'desired_depth', 'max_weight'
        """
        
        candidates = []
        
        # --- STEP 1: HARD FILTERING & ANALYSIS ---
        for index, beam in self.beam_library.iterrows():
            
            # 1. Environment Filter
            # Check if user's environment is in the beam's suitable list
            if user_reqs['environment'] not in beam['env_suitability']:
                continue 

            # 2. Run Analysis Engine
            # Note: We pass E and I_xx from the CSV row
            analyzer = BeamAnalysis(user_reqs['length'], beam['E'], beam['I_xx'])
            
            # Add Self Weight
            gravity = 9.81
            self_weight = beam['Density'] * beam['Area'] * gravity
            analyzer.add_load('distributed', self_weight, 0, user_reqs['length'])
            
            for load in user_reqs['loads']:
                analyzer.add_load(load['type'], load['mag'], load['a'], load.get('b'))
            
            for sup in user_reqs['supports']:
                analyzer.add_support(sup['type'], sup['loc'])
                
            res = analyzer.analyze()
            
            if "error" in res: continue # Skip unstable
            
            # 3. Safety Factor Check
            # Sigma = M*y/I. Assuming symmetric beam, y = Depth/2
            y_extreme = beam['Depth'] / 2
            max_stress = (res['max_moment'] * y_extreme) / beam['I_xx']
            
            # Avoid divide by zero
            safety_factor = (beam['Yield_Stress'] / max_stress) if max_stress > 0.000001 else 100.0
            
            if safety_factor < 1.5: # Hard Constraint: Safety must be >= 1.5
                continue 
                
            # Calculate Derived Metrics
            total_cost = beam['Cost_Per_m'] * user_reqs['length']
            total_weight = beam['Area'] * user_reqs['length'] * beam['Density']
            
            # Append to candidates
            candidates.append({
                'beam_id': beam['beam_id'],
                'material': beam['material'],
                'type': beam['type'],
                'designation': beam.get('Designation', 'Unknown'), # CSV has 'Designation'
                'total_cost': total_cost,
                'total_weight': total_weight,
                'safety_factor': safety_factor,
                'depth': beam['Depth'],
                'max_deflection': res['max_deflection'],
                'original_data': beam.to_dict()
            })

        if not candidates:
            return {"error": "No designs meet safety/environment requirements."}
        
        df = pd.DataFrame(candidates)

        # --- STEP 2: DYNAMIC WEIGHTED SCORING ---
        max_cost = df['total_cost'].max()
        max_weight = df['total_weight'].max()
        max_sf = df['safety_factor'].max()

        # Normalize (0 to 1)
        df['n_cost'] = 1 - (df['total_cost'] / max_cost) if max_cost > 0 else 1
        df['n_weight'] = 1 - (df['total_weight'] / max_weight) if max_weight > 0 else 1
        df['n_safety'] = (df['safety_factor'] / max_sf) if max_sf > 0 else 0

        # Define Base Weights
        weights = {
            'cost': 0.40,
            'weight': 0.30,
            'safety': 0.30,
            'material': 0.0,
            'depth': 0.0
        }

        # --- DYNAMIC ADJUSTMENT ---
        
        # Case A: User specifies 'desired_material'
        if 'desired_material' in user_reqs and user_reqs['desired_material']:
            df['score_mat'] = df['material'].apply(
                lambda x: 1.0 if str(x).lower() == user_reqs['desired_material'].lower() else 0.0
            )
            weights['material'] = 0.50
            weights['cost'] = 0.20
            weights['weight'] = 0.15
            weights['safety'] = 0.15
        else:
            df['score_mat'] = 0.0

        # Case B: User specifies 'max_cost'
        if 'max_cost' in user_reqs and user_reqs['max_cost']:
            # Soft penalty: If over budget, cut score significantly
            over_budget_mask = df['total_cost'] > user_reqs['max_cost']
            df.loc[over_budget_mask, 'n_cost'] -= 0.5 
            
            # Boost cost importance
            weights['cost'] += 0.20 
            total = sum(weights.values())
            for k in weights: weights[k] /= total

        # Case C: User specifies 'desired_depth'
        if 'desired_depth' in user_reqs and user_reqs['desired_depth']:
            target = user_reqs['desired_depth']
            max_diff = max(df['depth'].max() - target, target)
            df['score_depth'] = 1 - (abs(df['depth'] - target) / max_diff) if max_diff > 0 else 1.0
            
            weights['depth'] = 0.25
            total = sum(weights.values())
            for k in weights: weights[k] /= total
        else:
            df['score_depth'] = 0.0

        # Calculate Final Score
        df['final_score'] = (
            (df['n_cost'] * weights['cost']) +
            (df['n_weight'] * weights['weight']) +
            (df['n_safety'] * weights['safety']) +
            (df['score_mat'] * weights['material']) +
            (df['score_depth'] * weights['depth'])
        )

        # Sort
        df_sorted = df.sort_values(by='final_score', ascending=False)
        best_match = df_sorted.iloc[0]
        matches = df_sorted.head(3)

        return {
            "recommended": best_match['original_data'],
            "score": best_match['final_score'],
            "matches": matches['original_data'].tolist(),
            "analysis_results": {
                "max_deflection": best_match['max_deflection'],
                "safety_factor": best_match['safety_factor'],
                "total_cost": best_match['total_cost']
            }
        }