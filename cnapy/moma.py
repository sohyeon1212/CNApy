
import cobra
from typing import Dict
from optlang.symbolics import Add

def linear_moma(model: cobra.Model, reference_fluxes: Dict[str, float]) -> cobra.Solution:
    with model as model:
        # We need to minimize sum(|v - v_ref|)
        # v - v_ref = delta_pos - delta_neg
        # minimize sum(delta_pos + delta_neg)
        # delta_pos, delta_neg >= 0
        
        objective_terms = []
        vars_to_add = []
        cons_to_add = []
        
        for rid, v_ref in reference_fluxes.items():
            if rid not in model.reactions:
                continue
            
            rxn = model.reactions.get_by_id(rid)
            
            # Skip if reaction is fixed to 0 in both reference and current model (optimization)
            if v_ref == 0 and rxn.lower_bound == 0 and rxn.upper_bound == 0:
                continue

            pos = model.problem.Variable(f"{rid}_pos_slack", lb=0)
            neg = model.problem.Variable(f"{rid}_neg_slack", lb=0)
            
            vars_to_add.extend([pos, neg])
            objective_terms.extend([pos, neg])
            
            # Constraint: v - pos + neg = v_ref
            # => v - pos + neg - v_ref = 0
            # In optlang: Constraint(expression, lb=v_ref, ub=v_ref) where expression is v - pos + neg
            
            expression = rxn.flux_expression - pos + neg
            constraint = model.problem.Constraint(expression, lb=v_ref, ub=v_ref)
            cons_to_add.append(constraint)
            
        model.add_cons_vars(vars_to_add)
        model.add_cons_vars(cons_to_add)
        
        model.objective = model.problem.Objective(Add(*objective_terms), direction='min')
        
        solution = model.optimize()
        return solution
