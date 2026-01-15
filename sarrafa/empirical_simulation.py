# Empirical Simulation: Capital-Intensive vs Labor-Intensive Techniques
# Based on the paper: "Reswitching as a Non‑Robust Phenomenon"

import numpy as np
import matplotlib.pyplot as plt

# Project lifespan
PROJECT_LIFESPAN = 20  # Years

# CPI inflation rate (1.6% per year)
CPI_RATE = 0.016

# Configuration for Technique A (Capital-Intensive)
# Units: Ten Thousand Yuan (10,000 Yuan)
TECHNIQUE_A = {
    "name": "Technique A (Capital-Intensive)",
    "initial_investment": 10.0,  # 100,000 Yuan
    "annual_fuel_cost": 5.5,     # 55,000 Yuan per year
    "annual_wage": 24.0,         # 240,000 Yuan per year for operator
    "special_costs": {
        5: 1.5,   # Maintenance at year 5: 15,000 Yuan
        12: 2.5,  # Component replacement at year 12: 25,000 Yuan
        20: 1.0   # Scrap disposal at year 20: 10,000 Yuan
    },
    "has_cpi_perturbation": False  # No CPI perturbation for Technique A
}

# Calculate base annual operating cost for Technique A
TECHNIQUE_A["base_annual_cost"] = TECHNIQUE_A["annual_fuel_cost"] + TECHNIQUE_A["annual_wage"]

# Configuration for Technique B (Labor-Intensive)
# Units: Ten Thousand Yuan (10,000 Yuan)
WORKERS_B = 25                     # Number of workers
WAGE_PER_WORKER_B = 12.0           # 120,000 Yuan per worker per year
SHOVELS_PER_WORKER_B = 6           # 6 shovels per worker per year
COST_PER_SHOVEL_B = 25.0 / 10000   # 25 Yuan per shovel, converted to Ten Thousand Yuan

TECHNIQUE_B = {
    "name": "Technique B (Labor-Intensive)",
    "initial_investment": 0.0,    # No initial investment
    "workers": WORKERS_B,
    "wage_per_worker": WAGE_PER_WORKER_B,
    "shovels_per_worker": SHOVELS_PER_WORKER_B,
    "cost_per_shovel": COST_PER_SHOVEL_B,
    "special_costs": {},          # No special costs
    "has_cpi_perturbation": True  # Apply CPI perturbation for Technique B
}

# Calculate base annual operating cost for Technique B
TECHNIQUE_B["annual_wage_total"] = WORKERS_B * WAGE_PER_WORKER_B
TECHNIQUE_B["annual_shovel_cost"] = WORKERS_B * SHOVELS_PER_WORKER_B * COST_PER_SHOVEL_B
TECHNIQUE_B["base_annual_cost"] = TECHNIQUE_B["annual_wage_total"] + TECHNIQUE_B["annual_shovel_cost"]



def calculate_accumulated_cost(technique, interest_rate):
    """
    Calculate accumulated cost over time using simple interest (only on initial investment)
    
    Args:
        technique: Dictionary containing technique configuration
        interest_rate: Annual interest rate (decimal, e.g., 0.05 for 5%)
        
    Returns:
        list: Accumulated cost for each year (0 to 20 years)
    """
    accumulated_costs = [technique["initial_investment"]]  # Year 0: only initial investment
    total_cost = technique["initial_investment"]
    
    for year in range(1, PROJECT_LIFESPAN + 1):
        # Simple interest: only on initial investment, not compounded
        interest = technique["initial_investment"] * interest_rate
        
        # Base annual operating cost
        annual_cost = technique["base_annual_cost"]
        
        # Add special costs if applicable for this year
        if year in technique["special_costs"]:
            annual_cost += technique["special_costs"][year]
        
        # Apply CPI perturbation if enabled
        if technique["has_cpi_perturbation"]:
            # Add 1.6% random perturbation (normal distribution around 1.6%)
            # Mean: 1.6%, Standard Deviation: 0.5% (reasonable for inflation variability)
            cpi_perturbation = np.random.normal(CPI_RATE, CPI_RATE * 0.33)  # 33% coefficient of variation
            annual_cost *= (1 + cpi_perturbation)
        
        # Update total accumulated cost
        total_cost += interest + annual_cost
        accumulated_costs.append(total_cost)
    
    return accumulated_costs



def calculate_linear_regression(years, costs):
    """
    Perform linear regression on accumulated costs over years
    and calculate residual standard deviation
    
    Args:
        years: List of years (independent variable)
        costs: List of accumulated costs (dependent variable)
        
    Returns:
        tuple: (slope, intercept, r_squared, residual_std)
            slope: Slope of the regression line
            intercept: Intercept of the regression line
            r_squared: R-squared value
            residual_std: Residual standard deviation
    """
    # Convert to numpy arrays
    x = np.array(years)
    y = np.array(costs)
    
    # Perform linear regression
    slope, intercept = np.polyfit(x, y, 1)
    
    # Calculate predicted values
    y_pred = slope * x + intercept
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Calculate R-squared
    y_mean = np.mean(y)
    total_variance = np.sum((y - y_mean) ** 2)
    explained_variance = np.sum((y_pred - y_mean) ** 2)
    r_squared = explained_variance / total_variance
    
    # Calculate residual standard deviation
    residual_std = np.std(residuals)
    
    return slope, intercept, r_squared, residual_std



def plot_technique_comparison():
    """
    Plot accumulated costs for both techniques at 5% interest rate (simple interest)
    with special cost markers for Technique A
    """
    interest_rate = 0.05  # 5% interest rate only
    years = list(range(0, PROJECT_LIFESPAN + 1))  # 0 to 20 years
    
    plt.figure(figsize=(14, 8))
    
    # Calculate accumulated costs for both techniques
    costs_a = calculate_accumulated_cost(TECHNIQUE_A, interest_rate)
    costs_b = calculate_accumulated_cost(TECHNIQUE_B, interest_rate)
    
    # Plot Technique A
    plt.plot(years, costs_a, 'b-', label=TECHNIQUE_A["name"], linewidth=2)
    
    # Add markers for Technique A special cost years
    for special_year, special_cost in TECHNIQUE_A["special_costs"].items():
        cost_at_year = costs_a[special_year]
        plt.plot(special_year, cost_at_year, 'bo', markersize=8)
        plt.text(special_year + 0.2, cost_at_year, f' +{special_cost}k', 
                 fontsize=10, verticalalignment='center', color='blue')
    
    # Plot Technique B
    plt.plot(years, costs_b, 'r-', label=TECHNIQUE_B["name"], linewidth=2)
    
    # Highlight special years for Technique A with vertical lines
    for special_year in TECHNIQUE_A["special_costs"].keys():
        plt.axvline(x=special_year, color='gray', linestyle=':', alpha=0.7)
    
    plt.title('Accumulated Costs Comparison: Capital-Intensive vs Labor-Intensive')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Accumulated Cost (Ten Thousand Yuan)', fontsize=12)
    
    # Add legend
    plt.legend(loc='upper left', fontsize=10)
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.8)
    
    # Set y-axis limits to accommodate both techniques
    min_cost = min(min(costs_a), min(costs_b)) * 0.9
    max_cost = max(max(costs_a), max(costs_b)) * 1.05
    plt.ylim(min_cost, max_cost)
    
    # Add text box with CPI information
    cpi_text = f"CPI Inflation Rate: {CPI_RATE*100:.2f}% per year (applied to Technique B only)"
    plt.text(0.02, 0.02, cpi_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('technique_comparison_costs.png', dpi=300)
    plt.close()
    print("Plot saved as 'technique_comparison_costs.png'")
    
    # Calculate linear regression for Technique A
    slope_a, intercept_a, r_squared_a, residual_std_a = calculate_linear_regression(years, costs_a)
    
    return slope_a, intercept_a, r_squared_a, residual_std_a



def calculate_zone_of_ambiguity():
    """
    Calculate the Zone of Ambiguity between two techniques using the formula from the paper
    
    Returns:
        tuple: (t_start, t_end, width, sa, sb)
            t_start: Start year of the ambiguity zone
            t_end: End year of the ambiguity zone
            width: Width of the ambiguity zone in years
            sa: Standard deviation of Technique A residuals (confidence band width)
            sb: Standard deviation of Technique B simulation results (confidence band width)
    """
    interest_rate = 0.05
    
    # Run multiple simulations to get confidence bands
    num_simulations = 1000
    
    # Store results for multiple simulations
    all_costs_b = []
    
    # Get Technique A costs (deterministic)
    costs_a = calculate_accumulated_cost(TECHNIQUE_A, interest_rate)
    
    # Run multiple simulations for Technique B (stochastic CPI perturbation)
    for i in range(num_simulations):
        costs_b = calculate_accumulated_cost(TECHNIQUE_B, interest_rate)
        all_costs_b.append(costs_b)
    
    # Calculate confidence band for Technique A (residuals from linear regression)
    years = np.array(range(0, PROJECT_LIFESPAN + 1))
    slope_a, intercept_a, r_squared_a, residual_std_a = calculate_linear_regression(years, costs_a)
    sa = residual_std_a
    
    # Calculate confidence band for Technique B (standard deviation of simulation results)
    all_costs_b_array = np.array(all_costs_b)
    # Calculate standard deviation across simulations for each year
    sb_yearly = np.std(all_costs_b_array, axis=0)
    # Use average standard deviation as confidence band width
    sb = np.mean(sb_yearly)
    
    # Paper formula parameters
    Ia = TECHNIQUE_A["initial_investment"]
    Ib = TECHNIQUE_B["initial_investment"]
    ma = TECHNIQUE_A["base_annual_cost"]
    mb = TECHNIQUE_B["base_annual_cost"]
    
    # Calculate numerator and denominator for the formula
    numerator = Ib - Ia
    denominator = ma - mb
    sa_plus_sb = sa + sb
    
    # Calculate t_start and t_end
    t_start = (numerator - sa_plus_sb) / denominator
    t_end = (numerator + sa_plus_sb) / denominator
    
    # Calculate width
    width = t_end - t_start
    
    return t_start, t_end, width, sa, sb


def main():
    """
    Main function to run the simulation
    """
    interest_rate = 0.05  # 5% interest rate
    
    print("Simulating Technique Comparison")
    print("=" * 60)
    
    # Print Technique A details
    print(f"{TECHNIQUE_A['name']}:")
    print(f"  Initial Investment: {TECHNIQUE_A['initial_investment']} Ten Thousand Yuan")
    print(f"  Annual Wage: {TECHNIQUE_A['annual_wage']} Ten Thousand Yuan")
    print(f"  Annual Fuel Cost: {TECHNIQUE_A['annual_fuel_cost']} Ten Thousand Yuan")
    print(f"  Base Annual Cost: {TECHNIQUE_A['base_annual_cost']} Ten Thousand Yuan")
    print("  Special Costs:")
    for year, cost in TECHNIQUE_A['special_costs'].items():
        print(f"    Year {year}: {cost} Ten Thousand Yuan")
    print(f"  CPI Perturbation: {'Enabled' if TECHNIQUE_A['has_cpi_perturbation'] else 'Disabled'}")
    print()
    
    # Print Technique B details
    print(f"{TECHNIQUE_B['name']}:")
    print(f"  Initial Investment: {TECHNIQUE_B['initial_investment']} Ten Thousand Yuan")
    print(f"  Number of Workers: {TECHNIQUE_B['workers']}")
    print(f"  Wage per Worker: {TECHNIQUE_B['wage_per_worker']} Ten Thousand Yuan per year")
    print(f"  Total Annual Wage: {TECHNIQUE_B['annual_wage_total']} Ten Thousand Yuan")
    print(f"  Shovels per Worker per Year: {TECHNIQUE_B['shovels_per_worker']}")
    print(f"  Cost per Shovel: {TECHNIQUE_B['cost_per_shovel']*10000:.0f} Yuan")
    print(f"  Total Annual Shovel Cost: {TECHNIQUE_B['annual_shovel_cost']:.4f} Ten Thousand Yuan")
    print(f"  Base Annual Cost: {TECHNIQUE_B['base_annual_cost']:.2f} Ten Thousand Yuan")
    print(f"  CPI Perturbation: {'Enabled' if TECHNIQUE_B['has_cpi_perturbation'] else 'Disabled'}")
    print()
    
    print(f"Interest Rate: {interest_rate*100:.0f}% (Simple Interest on Initial Investment Only)")
    print(f"CPI Inflation Rate: {CPI_RATE*100:.2f}% per year")
    print()
    
    # Calculate detailed annual costs for Technique A and print
    print("Annual Cost Breakdown for Technique A (Ten Thousand Yuan)")
    print("Year | Initial | Wage | Fuel | Special | Interest | Total | Cumulative")
    print("-" * 80)
    
    cumulative_cost_a = TECHNIQUE_A["initial_investment"]
    print(f"   0 |   {TECHNIQUE_A['initial_investment']:4.1f} |    0 |    0 |      0 |       0 |  {TECHNIQUE_A['initial_investment']:4.1f} |      {cumulative_cost_a:6.1f}")
    
    for year in range(1, PROJECT_LIFESPAN + 1):
        # Calculate interest (simple interest: only on initial investment)
        interest = TECHNIQUE_A["initial_investment"] * interest_rate
        
        # Base annual costs
        wage = TECHNIQUE_A["annual_wage"]
        fuel = TECHNIQUE_A["annual_fuel_cost"]
        
        # Special costs if applicable
        special = TECHNIQUE_A["special_costs"].get(year, 0)
        
        # Total annual cost
        annual_total = wage + fuel + special + interest
        
        # Update cumulative cost
        cumulative_cost_a += annual_total
        
        print(f"{year:4d} |      0 | {wage:4.1f} | {fuel:4.1f} |   {special:4.1f} |  {interest:6.1f} | {annual_total:5.1f} |      {cumulative_cost_a:6.1f}")
    
    # Generate and save the comparison plot, get regression results for Technique A
    slope, intercept, r_squared, residual_std = plot_technique_comparison()
    
    # Print regression results
    print()
    print("=" * 60)
    print(f"Linear Regression Analysis for {TECHNIQUE_A['name']}")
    print("=" * 60)
    print(f"Regression Equation: Cost = {slope:.2f} * Year + {intercept:.2f}")
    print(f"R-squared: {r_squared:.4f} ({r_squared*100:.2f}% of variance explained)")
    print(f"Residual Standard Deviation: {residual_std:.2f} Ten Thousand Yuan")
    
    # Calculate average cost for Technique A
    costs_a = calculate_accumulated_cost(TECHNIQUE_A, interest_rate)
    avg_cost_a = np.mean(costs_a)
    print(f"Average Cost: {avg_cost_a:.2f} Ten Thousand Yuan")
    print(f"Coefficient of Variation for Residuals: {(residual_std / avg_cost_a)*100:.2f}%")
    print()
    
    # Print final cost comparison (single simulation)
    costs_b = calculate_accumulated_cost(TECHNIQUE_B, interest_rate)
    print("Final Cost Comparison (Year 20) - Single Simulation")
    print("=" * 50)
    print(f"{TECHNIQUE_A['name']}: {costs_a[-1]:.2f} Ten Thousand Yuan")
    print(f"{TECHNIQUE_B['name']}: {costs_b[-1]:.2f} Ten Thousand Yuan")
    print(f"Difference (B - A): {costs_b[-1] - costs_a[-1]:.2f} Ten Thousand Yuan")
    print(f"{(TECHNIQUE_B['name'])} is {'more' if costs_b[-1] > costs_a[-1] else 'less'} expensive by {abs(costs_b[-1] - costs_a[-1]):.2f} Ten Thousand Yuan")
    print()
    
    # Calculate Zone of Ambiguity
    print("=" * 60)
    print("Calculating Zone of Ambiguity")
    print("=" * 60)
    t_start, t_end, width, sa, sb = calculate_zone_of_ambiguity()
    
    print("Zone of Ambiguity Parameters:")
    print(f"  Sa (Technique A confidence band width): {sa:.2f} Ten Thousand Yuan")
    print(f"  Sb (Technique B confidence band width): {sb:.2f} Ten Thousand Yuan")
    print(f"  Ia - Ib: {TECHNIQUE_A['initial_investment'] - TECHNIQUE_B['initial_investment']:.2f} Ten Thousand Yuan")
    print(f"  ma - mb: {TECHNIQUE_A['base_annual_cost'] - TECHNIQUE_B['base_annual_cost']:.2f} Ten Thousand Yuan/year")
    print()
    
    print("Zone of Ambiguity Results:")
    print(f"  Start Year (t_start): {t_start:.2f} years")
    print(f"  End Year (t_end): {t_end:.2f} years")
    print(f"  Width (W): {width:.2f} years")
    print()
    
    if width > 0 and t_start > 0 and t_end < PROJECT_LIFESPAN:
        print("Interpretation:")
        print(f"  The Zone of Ambiguity exists between year {t_start:.1f} and {t_end:.1f}.")
        print(f"  Within this {width:.1f}-year window, the choice between techniques is uncertain.")
        print(f"  Outside this window, one technique is clearly superior to the other.")
    else:
        print("Interpretation:")
        print(f"  No meaningful Zone of Ambiguity exists within the project lifespan.")
        print(f"  One technique is clearly superior to the other throughout the entire {PROJECT_LIFESPAN}-year period.")
    
    print()
    print("Simulation completed. Check 'technique_comparison_costs.png' for the plot.")



if __name__ == "__main__":
    main()
