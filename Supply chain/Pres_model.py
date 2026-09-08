import joblib

from pyomo.environ import (
    ConcreteModel,
    Var,
    NonNegativeIntegers,
    Constraint,
    Objective,
    minimize,
    SolverFactory,
    value
)


# ============================================================
# 1. LOAD TRAINED PREDICTIVE MODEL
# ============================================================

model_data = joblib.load(
    "demand_model.pkl"
)

predictive_model = model_data["model"]
features = model_data["features"]


print("Predictive model loaded successfully.")


# ============================================================
# 2. CREATE INPUT DATA FOR NEXT WEEK
# ============================================================

# We want to predict demand for week 105.

next_week = {
    "week": 105,
    "week_of_year": 1,
    "month": 1,

    # Expected business conditions
    "price": 9.50,
    "promotion": 1,
    "holiday": 0,
    "temperature": 18,

    # Demand observed in week 104
    "previous_week_demand": 79.2396
}


# ============================================================
# 3. PREDICT NEXT WEEK'S DEMAND
# ============================================================

prediction_input = [
    [
        next_week[feature]
        for feature in features
    ]
]

predicted_demand = predictive_model.predict(
    prediction_input
)[0]


print("\nPREDICTIVE MODEL")
print("----------------")
print(
    f"Predicted demand: "
    f"{predicted_demand:.0f} units"
)


# ============================================================
# 4. CURRENT INVENTORY
# ============================================================

current_inventory = 40

# Desired safety stock
safety_stock = 20


# ============================================================
# 5. PRESCRIPTIVE MODEL
# ============================================================

pyomo_model = ConcreteModel()


# Decision variable:
# How many units should we order?
pyomo_model.order = Var(
    domain=NonNegativeIntegers
)


# ============================================================
# 6. BUSINESS PARAMETERS
# ============================================================

order_cost = 4

supplier_capacity = 200


# ============================================================
# 7. INVENTORY CONSTRAINT
# ============================================================

def inventory_constraint(model):

    return (
        current_inventory
        + model.order
        >=
        predicted_demand
        + safety_stock
    )


pyomo_model.inventory_constraint = Constraint(
    rule=inventory_constraint
)


# ============================================================
# 8. SUPPLIER CAPACITY
# ============================================================

pyomo_model.capacity_constraint = Constraint(
    expr=pyomo_model.order
    <= supplier_capacity
)


# ============================================================
# 9. OBJECTIVE FUNCTION
# ============================================================

# Minimize ordering cost

pyomo_model.objective = Objective(
    expr=order_cost * pyomo_model.order,
    sense=minimize
)


# ============================================================
# 10. SOLVE
# ============================================================

solver = SolverFactory("glpk")

results = solver.solve(
    pyomo_model
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

optimal_order = value(
    pyomo_model.order
)

final_inventory = (
    current_inventory
    + optimal_order
)

print("\nPRESCRIPTIVE MODEL")
print("------------------")

print(
    f"Current inventory: "
    f"{current_inventory:.0f}"
)

print(
    f"Predicted demand: "
    f"{predicted_demand:.0f}"
)

print(
    f"Safety stock: "
    f"{safety_stock:.0f}"
)

print(
    f"Optimal order: "
    f"{optimal_order:.0f}"
)

print(
    f"Inventory after order: "
    f"{final_inventory:.0f}"
)