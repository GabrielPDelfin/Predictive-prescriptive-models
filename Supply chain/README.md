Simple supply-chain optimization problem.

Given historical weekly data on the demand of a specific product, we want to predict next week's demand.
This prediction is used in a MO model to compute the optimal number of product that we must order to meet the demand.

The number of ordered product is subject to capacity and demand constraints.

The features used in the ML model are:
- week: general week number
- week_of_year: Week number relative to the current year
- month: Month Number
- price: The product's selling price
- promotion: Binary. Indicates whether the product is beaing promoted or not
- holiday: Binary. Indicates whether the week belongs to a holiday week or not
- temperature: Indicates the average temperature of the week
- previous_week_demand: How much product was sold the previous week