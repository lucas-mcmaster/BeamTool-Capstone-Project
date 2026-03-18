import pandas as pd
import numpy as np
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
        OPTIONAL: 'desired_material', 'max_cost', 'desired_depth', 'max_weight', 'design_priority'
        """
        
        candidates = []
        rejected_beams = []
        
        # --- STEP 1: HARD FILTERING & ANALYSIS ---
        for index, beam in self.beam_library.iterrows():
            
            beam_name = beam.get('Designation', 'Unknown')

            # 1. Environment Filter
            # Check if user's environment is in the beam's suitable list
            if user_reqs['environment'] not in beam['env_suitability']:
                rejected_beams.append({
                    "designation": beam_name,
                    "reason": f"Environment mismatch. User requires '{user_reqs['environment']}'."
                    
                })
                continue 
            
            # 2.a. Slender Beam (Euler-Bernoulli) Check (L/d >= 10)
            span_to_depth_ratio = user_reqs['length'] / beam['Depth']
            if span_to_depth_ratio < 10:
                rejected_beams.append({
                    "designation": beam_name,
                    "reason": f"Failed slender beam check. Span-to-depth ratio is {span_to_depth_ratio:.1f} (must be >= 10)."
                })
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
            
            if "error" in res: # Skip unstable
                rejected_beams.append({
                    "designation": beam_name,
                    "reason": "Unstable structure configuration."
                })
                continue
            
            #3. Deflection check to ensure following slender beam theory - using L/240 or L/360 as requirement-stipulated by Canada building code CSAS16 for steel 
            allowable_deflection = user_reqs['length'] / 240
            actual_max_deflection = abs(res['max_deflection'])

            if actual_max_deflection > allowable_deflection:
                rejected_beams.append({
                    "designation": beam_name,
                    "reason": f"SERVICEABILITY FAILURE: Max deflection is {actual_max_deflection*1000:.2f}mm, "
                              f"exceeding the limit of {allowable_deflection*1000:.2f}mm (L/240).",
                    "type": "deflection_failure"
                })
                continue

            # 4. Safety Factor Check
            #Von mises stress analysis
        
            #For calculating bending stress Sigma = M*y/I. Assuming symmetric beam, y = Depth/2
            y_extreme = beam['Depth'] / 2
            max_stress = (res['max_moment'] * y_extreme) / beam['I_xx']
            vm_bending = max_stress #for comparing to shear vm

            #calculating shear stress using a shape factor and area of beam since we do not always have width of section or moment of area
            max_shear= (res['max_shear']/beam['Area'])*1.5 #using a shape factor of 1.5 as it covers almost all beams
            vm_shear = np.sqrt(3)*max_shear

            vm_max = max(vm_bending, vm_shear)
            
            # Avoid divide by zero
            safety_factor = (beam['Yield_Stress'] / vm_max) if vm_max > 0.000001 else 100.0
            
            user_safety_factor= user_reqs.get('min_safety_factor', 1.0) #user selected SF. Forced to 1 if not given

            if safety_factor < user_safety_factor: # Hard Constraint: Safety must be greater than user selection
                if safety_factor < 1:
                    rejected_beams.append({
                        "designation": beam_name,
                        "reason": f"MANDATORY FAILURE: Beam would physically yield. Ssafety Factor = {safety_factor:.2f} (Minimum required 1.0).",
                        "type": "structural_failure"
                    })
                else:
                    rejected_beams.append({
                        "designation": beam_name,
                        "reason": f"USER CONSTRAINT: Beam is safe but below user's custom margin. Safety Factor = {safety_factor:.2f} (User requested >= {user_safety_factor}).",
                        "type": "user_constraint",
                        "total_cost": beam['Cost_Per_m'] * user_reqs['length'] #AI can use this for cost-benefit discussion
                    })
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
            'cost': 0.50,
            'weight': 0.40,
            'safety': 0.10,
            'material': 0.0,
            'depth': 0.0,
            'section':0.0
        }

        # --- DYNAMIC ADJUSTMENT ---
        #Starting with selected user priority
        user_priority = user_reqs.get('design_priority')
        if user_priority in ['cost', 'weight', 'safety']:
            weights[user_priority] += 0.50  # Significant boost to the primary goal
        
        # Case A: User specifies 'desired_material'
        if 'desired_material' in user_reqs and user_reqs['desired_material']:
            df['score_mat'] = df['material'].apply(
                lambda x: 1.0 if str(x).lower() == user_reqs['desired_material'].lower() else 0.0
            )
            weights['material'] += 0.40  #adds importance to material type 
        else:
            df['score_mat'] = 0.0

        #Case B: User specifies a 'desired_section'
        if user_reqs.get('desired_section'):
            # Checks the 'type' column from CSV
            df['score_section'] = df['type'].apply(
                lambda x: 1.0 if str(x).lower() == user_reqs['desired_section'].lower() else 0.0
            )
            weights['section'] += 0.35  # Add importance to cross-section match
        else:
            df['score_section'] = 0.0

        # Case C: User specifies 'max_cost'
        if 'max_cost' in user_reqs and user_reqs['max_cost']:
            # Soft penalty: If over budget, cut score significantly
            over_budget_mask = df['total_cost'] > user_reqs['max_cost']
            df.loc[over_budget_mask, 'n_cost'] -= 0.5 
            
            # Boost cost importance since budget also exists
            weights['cost'] += 0.20 

        # Case D: User specifies 'desired_depth'
        if 'desired_depth' in user_reqs and user_reqs['desired_depth']:
            target = user_reqs['desired_depth']
            max_diff = max(df['depth'].max() - target, target)
            df['score_depth'] = 1 - (abs(df['depth'] - target) / max_diff) if max_diff > 0 else 1.0
            
            weights['depth'] += 0.3 #making depth have some value
        else:
            df['score_depth'] = 0.0

        #normalizing weightings
        total_weight_points = sum(weights.values())
        for k in weights:
            weights[k] /= total_weight_points

        #Calculate Final Score
        df['final_score'] = (
            (df['n_cost'] * weights['cost']) +
            (df['n_weight'] * weights['weight']) +
            (df['n_safety'] * weights['safety']) +
            (df['score_mat'] * weights['material']) +
            (df['score_depth'] * weights['depth']) +
            (df['score_section'] * weights['section'])
        )

        # Sort
        df_sorted = df.sort_values(by='final_score', ascending=False)
        best_match = df_sorted.iloc[0]
        
        #code to send data for top 3 matches
        matches_data = []
        for _, row in df_sorted.head(3).iterrows():
            matches_data.append({
                "Designation": row['designation'],
                "material": row['material'],
                "type": row['type'],
                "Depth": row['depth'],
                "score": row['final_score'],
                "total_cost": row['total_cost'],
                "safety_factor": row['safety_factor'],
                "max_deflection": row['max_deflection']
            })

        #identifying high value tradeoffs where safety factor choice impacts best beam
        high_value_tradeoffs = []
        recommended_cost = best_match['total_cost']

        for rej in rejected_beams:
            # Only look at beams rejected for USER constraints, not structural failure
            if rej.get('type') == "user_constraint":
                # Check if this rejected beam is significantly cheaper (e.g., > 15% savings)
                if rej['total_cost'] < (recommended_cost * 0.85):
                    savings = recommended_cost - rej['total_cost']
                    rej['is_high_value'] = True
                    rej['potential_savings'] = savings
                    high_value_tradeoffs.append(rej)

        return {
            "recommended": best_match['original_data'],
            "score": best_match['final_score'],
            "matches": matches_data,
            "analysis_results": {
                "max_deflection": best_match['max_deflection'],
                "safety_factor": best_match['safety_factor'],
                "total_cost": best_match['total_cost']
            },
            "rejected_beams" : rejected_beams, #returns the dict of rejected beams for tracking by AI
            "tradeoffs": high_value_tradeoffs  #New field for the AI to prioritize
        }