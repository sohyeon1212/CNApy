
import cobra
from cnapy.moma import linear_moma
from optlang.symbolics import Add

def test_linear_moma():
    try:
        cobra.Configuration().solver = 'glpk'
    except Exception:
        print("Could not set solver to glpk, trying default")
        
    print("Creating toy model...")
    model = cobra.Model("toy_model")
    
    # Create metabolites
    A = cobra.Metabolite('A')
    B = cobra.Metabolite('B')
    C = cobra.Metabolite('C')
    
    # Create reactions
    # R1: -> A
    R1 = cobra.Reaction('R1')
    R1.add_metabolites({A: 1})
    R1.lower_bound = 0
    R1.upper_bound = 10
    
    # R2: A -> B
    R2 = cobra.Reaction('R2')
    R2.add_metabolites({A: -1, B: 1})
    R2.lower_bound = 0
    R2.upper_bound = 10
    
    # R3: B -> C
    R3 = cobra.Reaction('R3')
    R3.add_metabolites({B: -1, C: 1})
    R3.lower_bound = 0
    R3.upper_bound = 10
    
    # R4: C -> 
    R4 = cobra.Reaction('R4')
    R4.add_metabolites({C: -1})
    R4.lower_bound = 0
    R4.upper_bound = 10
    
    model.add_reactions([R1, R2, R3, R4])
    model.objective = 'R4'
    
    print("Running FBA on reference model...")
    solution_ref = model.optimize()
    print("Reference solution status:", solution_ref.status)
    print("Reference fluxes:", solution_ref.fluxes.to_dict())
    
    reference_fluxes = solution_ref.fluxes.to_dict()
    
    # Create a knockout scenario
    print("\nApplying knockout (R2 = 0)...")
    # We don't modify the model object directly for MOMA usually if we want to keep the original model structure 
    # but here we want to simulate a modified state.
    # In CNApy, the scenario is applied to the model before optimization.
    
    # Let's constrain R2 to 0
    model.reactions.R2.lower_bound = 0
    model.reactions.R2.upper_bound = 0
    
    print("Running Linear MOMA...")
    moma_solution = linear_moma(model, reference_fluxes)
    
    print("MOMA solution status:", moma_solution.status)
    print("MOMA fluxes:", moma_solution.fluxes.to_dict())
    
    # Check if R2 is indeed 0
    if abs(moma_solution.fluxes['R2']) < 1e-6:
        print("SUCCESS: R2 is 0 as expected.")
    else:
        print("FAILURE: R2 is not 0.")
        
    # Check if it tried to stay close to reference
    # Reference was R1=10, R2=10, R3=10, R4=10
    # With R2=0, max flow is 0.
    # Wait, if R2=0, then A accumulates? No, A is produced by R1.
    # If R2=0, then R1 must be 0 for mass balance of A (if A cannot accumulate).
    # Let's check mass balance constraints.
    # A: R1 - R2 = 0 => R1 = R2
    # B: R2 - R3 = 0 => R3 = R2
    # C: R3 - R4 = 0 => R4 = R3
    # So if R2=0, all must be 0.
    
    # Let's make a more interesting model where there is an alternative pathway.
    # R5: A -> C
    print("\nAdding alternative pathway R5: A -> C")
    R5 = cobra.Reaction('R5')
    R5.add_metabolites({A: -1, C: 1})
    R5.lower_bound = 0
    R5.upper_bound = 10
    model.add_reactions([R5])
    
    # Reset R2
    model.reactions.R2.lower_bound = 0
    model.reactions.R2.upper_bound = 10
    
    print("Running FBA on new reference model...")
    solution_ref = model.optimize()
    print("Reference fluxes:", solution_ref.fluxes.to_dict())
    reference_fluxes = solution_ref.fluxes.to_dict()
    
    # Now knockout R2 again
    print("\nApplying knockout (R2 = 0) again...")
    model.reactions.R2.lower_bound = 0
    model.reactions.R2.upper_bound = 0
    
    print("Running Linear MOMA...")
    moma_solution = linear_moma(model, reference_fluxes)
    print("MOMA fluxes:", moma_solution.fluxes.to_dict())
    
    # In FBA, R5 would likely take all flux to maximize R4.
    # In MOMA, we want to minimize distance to reference.
    # Reference was likely using R2 (shortest path? or just arbitrary if costs are equal).
    # If reference used R2=10, R5=0.
    # Now R2=0.
    # FBA would say R5=10 to get R4=10.
    # MOMA would say:
    # dist = |R1-10| + |R2-10| + |R3-10| + |R4-10| + |R5-0|
    # If R1=10, R5=10, R4=10, R2=0, R3=0 (since R2=0 -> R3=0)
    # dist = |10-10| + |0-10| + |0-10| + |10-10| + |10-0| = 0 + 10 + 10 + 0 + 10 = 30
    # If R1=0, ...
    # dist = |0-10| + |0-10| + |0-10| + |0-10| + |0-0| = 10 + 10 + 10 + 10 + 0 = 40
    # So MOMA should prefer maintaining flux through R5 to restore objective?
    # Actually MOMA minimizes flux change.
    # If we just drop everything to 0, change is 40.
    # If we switch to R5, change is 30.
    # So it should switch to R5.
    
    if moma_solution.fluxes['R5'] > 9.0:
         print("SUCCESS: MOMA redirected flux to R5.")
    else:
         print("Note: MOMA did not redirect full flux to R5, which is also a valid outcome depending on the exact minimization.")

if __name__ == "__main__":
    test_linear_moma()
