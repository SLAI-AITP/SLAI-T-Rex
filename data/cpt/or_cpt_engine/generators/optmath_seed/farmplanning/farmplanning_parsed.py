import gurobipy as gp
from gurobipy import GRB
import random

from or_cpt_engine.generators.numeric import uniform_rounded

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Farm Planning optimization problem.

        Parameters:
            See original code.
        """
        self.problem_type = "farm_planning"
        self.mathematical_formulation = r"""
        ### Mathematical Model for Farm Resource Planning

        Sets:
        - \(C\): crops
        - \(T\): months
        - \(B\): family-consumption bundles

        Decision variables:
        - \(x_c \ge 0\): area planted with crop \(c\)
        - \(s_c \ge 0\): amount of crop \(c\) sold
        - \(q_b \ge 0\): fraction of family-consumption bundle \(b\)
        - \(h^F \ge 0\): family labor level
        - \(h^P \ge 0\): permanent labor level
        - \(h^T_t \ge 0\): temporary labor in month \(t\)

        Objective:
        Maximize crop sales revenue minus family/permanent/temporary labor cost
        and water cost.

        Core constraints:
        - monthly land occupation does not exceed available land
        - monthly labor requirement is covered by family, permanent, and temporary labor
        - monthly water use does not exceed monthly water limit
        - annual water use does not exceed annual water availability
        - each crop yield is split between family consumption bundles and sales
        - family-consumption bundle fractions sum to one

        This is a continuous farm planning LP. It does not contain binary crop
        selection, fixed planting setup costs, inventory carryover, or crop
        rotation constraints.
        """
        
        default_parameters = {
            "n_crops": (3, 5),
            "n_months": (8, 10),
            "n_consumption_bundles": (2, 4),
            "yield_range": (1, 10),
            "price_range": (80, 200),
            "land_available": (100, 200),
            "labor_required_range": (1, 10),
            "water_requirement_range": (0.1, 1.0),
            "water_limit": (50, 80),
            "annual_water_available": (500, 1000),
            "wage_rates": {
                "family": 500,
                "permanent": 800,
                "temporary": 5
            },
            "price_of_water": 1,  # Added Price of Water (dollars per cubic km)
        }
        
        if parameters is None or not parameters:
            parameters = default_parameters
        else:
            parameters = {**default_parameters, **dict(parameters)}
            
        for key, value in parameters.items():
            setattr(self, key, value)
        self.parameters = dict(parameters)
        
        self.seed = seed
        if self.seed is not None:
            random.seed(seed)

    def generate_instance(self):
        """
        Generate a Farm Planning problem instance and create its corresponding Gurobi model.
        """
        # Randomly select number of crops, months, and consumption bundles
        self.n_crops = random.randint(*self.n_crops)
        self.n_months = random.randint(*self.n_months)
        self.n_consumption_bundles = random.randint(*self.n_consumption_bundles)
        self.land_available = random.randint(*self.land_available)
        self.water_limit = random.randint(*self.water_limit)
        self.annual_water_available = random.randint(*self.annual_water_available)

        # Generate sets
        crops = [f"crop_{i}" for i in range(self.n_crops)]
        months = [f"month_{i}" for i in range(self.n_months)]
        consumption_bundles = [f"bundle_{i}" for i in range(self.n_consumption_bundles)]

        # Generate parameters
        yield_c = {crop: random.randint(*self.yield_range) for crop in crops}
        price_c = {crop: random.randint(*self.price_range) for crop in crops}
        labor_required = {(month, crop): random.randint(*self.labor_required_range) for month in months for crop in crops}
        water_requirement = {(month, crop): uniform_rounded(*self.water_requirement_range) for month in months for crop in crops}
        amount_in_bundle = {(crop, bundle): uniform_rounded(0.1, 1.0) for crop in crops for bundle in consumption_bundles}
        fraction_occupies_land = {(month, crop): uniform_rounded(0.1, 1.0) for month in months for crop in crops}
        working_hours = 160  # Example: 160 hours per month per laborer

        # Create Gurobi model
        model = gp.Model("FarmPlanning")
        model.Params.OutputFlag = 0  # Suppress Gurobi output

        # Decision variables
        amount_planted = model.addVars(crops, vtype=GRB.CONTINUOUS, name="AmountPlanted")
        permanent_labor_hired = model.addVar(vtype=GRB.CONTINUOUS, name="PermanentLaborHired")
        temporary_labor_hired = model.addVars(months, vtype=GRB.CONTINUOUS, name="TemporaryLaborHired")
        family_labor_available = model.addVar(vtype=GRB.CONTINUOUS, name="FamilyLaborAvailable")
        sales = model.addVars(crops, vtype=GRB.CONTINUOUS, name="Sales")
        fraction_consumed = model.addVars(consumption_bundles, vtype=GRB.CONTINUOUS, name="FractionConsumed")

        self.parameters.update(
            {
                "crops": crops,
                "months": months,
                "consumption_bundles": consumption_bundles,
                "yield_per_area": yield_c,
                "crop_price": price_c,
                "land_available": self.land_available,
                "monthly_water_limit": self.water_limit,
                "annual_water_available": self.annual_water_available,
                "wage_rates": self.wage_rates,
                "price_of_water": self.price_of_water,
                "working_hours_per_labor_unit": working_hours,
                "labor_required": {f"{month}|{crop}": labor_required[month, crop] for month in months for crop in crops},
                "water_requirement": {f"{month}|{crop}": water_requirement[month, crop] for month in months for crop in crops},
                "amount_in_bundle": {f"{crop}|{bundle}": amount_in_bundle[crop, bundle] for crop in crops for bundle in consumption_bundles},
                "fraction_occupies_land": {f"{month}|{crop}": fraction_occupies_land[month, crop] for month in months for crop in crops},
                "compact_farm_planning_tables": {
                    "sets": {
                        "crops": crops,
                        "months": months,
                        "consumption_bundles": consumption_bundles,
                    },
                    "crop_table": {
                        "columns": ["crop", "yield_per_area", "selling_price"],
                        "rows": [
                            [crop, yield_c[crop], price_c[crop]]
                            for crop in crops
                        ],
                    },
                    "monthly_crop_resource_table": {
                        "columns": [
                            "month",
                            "crop",
                            "land_occupation_fraction",
                            "labor_required_per_area",
                            "water_required_per_area",
                        ],
                        "rows": [
                            [
                                month,
                                crop,
                                fraction_occupies_land[month, crop],
                                labor_required[month, crop],
                                water_requirement[month, crop],
                            ]
                            for month in months
                            for crop in crops
                        ],
                    },
                    "consumption_bundle_table": {
                        "columns": ["crop", "bundle", "crop_amount_in_bundle"],
                        "rows": [
                            [crop, bundle, amount_in_bundle[crop, bundle]]
                            for crop in crops
                            for bundle in consumption_bundles
                        ],
                    },
                    "resource_and_cost_table": {
                        "land_available_each_month": self.land_available,
                        "monthly_water_limit": self.water_limit,
                        "annual_water_available": self.annual_water_available,
                        "working_hours_per_labor_unit": working_hours,
                        "wage_rates": self.wage_rates,
                        "price_of_water": self.price_of_water,
                    },
                    "required_decision_layers": [
                        "AmountPlanted[c] continuous planted area",
                        "Sales[c] continuous crop sales",
                        "FractionConsumed[b] continuous family-consumption bundle fraction",
                        "FamilyLaborAvailable, PermanentLaborHired, TemporaryLaborHired[t] continuous labor levels",
                    ],
                },
                "decision_variables": {
                    "AmountPlanted[c]": "continuous planted area for crop c",
                    "Sales[c]": "continuous amount of crop c sold",
                    "FractionConsumed[b]": "continuous fraction of consumption bundle b used by the family",
                    "FamilyLaborAvailable": "continuous family labor level",
                    "PermanentLaborHired": "continuous permanent labor level",
                    "TemporaryLaborHired[t]": "continuous temporary labor in month t",
                },
                "objective_terms": [
                    "crop_price_times_sales",
                    "family_labor_wage_cost",
                    "permanent_labor_wage_cost",
                    "temporary_labor_wage_cost",
                    "water_use_cost",
                ],
                "required_constraints": [
                    "monthly_land_occupation_capacity",
                    "monthly_labor_requirement_coverage",
                    "monthly_water_limit",
                    "annual_water_availability",
                    "crop_yield_split_between_family_consumption_and_sales",
                    "family_consumption_bundle_fraction_sum",
                ],
                "required_parameter_presentation": [
                    "list crops, months, and consumption bundles",
                    "list crop yield and crop selling price",
                    "list monthly land occupation coefficient for each crop",
                    "list monthly labor requirement for each crop",
                    "list monthly water requirement for each crop",
                    "list bundle composition amount for every crop-bundle pair",
                    "list land availability, monthly water limit, and annual water availability",
                    "list family, permanent, temporary labor wage rates and water price",
                    "state that this is a continuous LP with no binary crop-selection variables",
                ],
                "structured_problem_data": {
                    "sets": {
                        "crops": crops,
                        "months": months,
                        "consumption_bundles": consumption_bundles,
                    },
                    "parameters": {
                        "yield_per_area": yield_c,
                        "crop_price": price_c,
                        "land_available": self.land_available,
                        "monthly_water_limit": self.water_limit,
                        "annual_water_available": self.annual_water_available,
                        "wage_rates": self.wage_rates,
                        "price_of_water": self.price_of_water,
                    },
                    "constraints": {
                        "land": "for each month, sum fraction_occupies_land[t,c] * AmountPlanted[c] <= land_available",
                        "labor": "for each month, crop labor requirement is covered by family, permanent, and temporary labor",
                        "monthly_water": "for each month, water use is at most monthly_water_limit",
                        "annual_water": "total water use across all months and crops is at most annual_water_available",
                        "yield_split": "for each crop, yield * planted area equals family consumption plus sales",
                        "bundle_fraction": "consumption bundle fractions sum to one",
                    },
                    "objective": "maximize crop sales revenue minus labor and water costs",
                },
                "business_interpretation_guardrails": [
                    "This is a continuous farm planning LP, not a binary crop-selection or fixed-charge model.",
                    "Do not add crop rotation, inventory carryover, minimum planting, or fixed setup constraints.",
                    "Do not omit family consumption bundle balance.",
                    "Do not omit labor cost, water cost, monthly land, monthly water, or annual water constraints.",
                    "Do not change the objective to minimize resource use; it maximizes net farm earnings.",
                ],
            }
        )

        # Objective function
        model.setObjective(
            gp.quicksum(price_c[crop] * sales[crop] for crop in crops) -
            self.wage_rates["family"] * family_labor_available -
            self.wage_rates["permanent"] * permanent_labor_hired -
            self.wage_rates["temporary"] * gp.quicksum(temporary_labor_hired[month] for month in months) -
            self.price_of_water * gp.quicksum(water_requirement[(month, crop)] * amount_planted[crop] for month in months for crop in crops),
            GRB.MAXIMIZE
        )

        # Constraints
        # Updated Land Limitation to include FractionOccupiesLand
        model.addConstrs(
            (
                gp.quicksum(fraction_occupies_land[(month, crop)] * amount_planted[crop] for crop in crops) <= self.land_available
                for month in months
            ),
            name="MonthlyLandCapacity",
        )

        # Labor requirements
        model.addConstrs(
            (
                gp.quicksum(labor_required[(month, crop)] * amount_planted[crop] for crop in crops)
                <= working_hours * (family_labor_available + permanent_labor_hired) + temporary_labor_hired[month]
                for month in months
            ),
            name="MonthlyLaborCoverage",
        )

        # Water requirements 1
        model.addConstrs(
            (
                gp.quicksum(water_requirement[(month, crop)] * amount_planted[crop] for crop in crops) <= self.water_limit
                for month in months
            ),
            name="MonthlyWaterLimit",
        )

        # Water requirements 2
        model.addConstr(
            gp.quicksum(water_requirement[(month, crop)] * amount_planted[crop] for month in months for crop in crops) <= self.annual_water_available,
            name="AnnualWaterAvailability"
        )

        # Family consumption 1
        model.addConstrs(
            (
                yield_c[crop] * amount_planted[crop]
                == gp.quicksum(amount_in_bundle[(crop, bundle)] * fraction_consumed[bundle] for bundle in consumption_bundles) + sales[crop]
                for crop in crops
            ),
            name="CropYieldSplit",
        )

        # Family consumption 2
        model.addConstr(
            gp.quicksum(fraction_consumed[bundle] for bundle in consumption_bundles) == 1,
            name="ConsumptionBundleNormalization"
        )

        return model

if __name__ == '__main__':
    import time
    def test_generator():
        generator = Generator()
        model = generator.generate_instance()

        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time

        model.write("farm_planning.lp")
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
    test_generator()
