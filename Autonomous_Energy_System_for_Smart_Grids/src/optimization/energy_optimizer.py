import pandas as pd
import numpy as np
from src import config

def get_rate_for_hour(hour: float) -> float:
    """
    Get the tariff rate (£ per kWh) for a given hour of the day (0-24).
    
    Tariff Rules:
    - Off-peak: 00:00 - 07:00 (hour >= 0 and hour < 7)
    - Peak: 16:00 - 20:00 (hour >= 16 and hour < 20)
    - Standard: All other times
    """
    if 0.0 <= hour < 7.0:
        return config.TOU_OFFPEAK_RATE
    elif 16.0 <= hour < 20.0:
        return config.TOU_PEAK_RATE
    else:
        return config.TOU_STANDARD_RATE

def calculate_flat_cost(energy_sum: float) -> float:
    """Calculate energy cost under standard flat rate tariff."""
    return energy_sum * config.STANDARD_RATE

def calculate_tou_cost(profile: pd.Series) -> dict:
    """
    Calculate cost under Time-of-Use (ToU) tariff.
    
    Parameters:
    - profile: pd.Series indexable by half-hour index (0-47) or hour index (0-23)
               representing energy consumption (in kWh) in that interval.
               
    Returns a dict with:
    - total_cost: Total cost in £
    - peak_cost: Cost incurred during peak hours
    - standard_cost: Cost incurred during standard hours
    - offpeak_cost: Cost incurred during off-peak hours
    - peak_energy: Energy consumed during peak hours
    - standard_energy: Energy consumed during standard hours
    - offpeak_energy: Energy consumed during off-peak hours
    """
    # Ensure profile index is integer type
    profile = profile.copy()
    profile.index = profile.index.astype(int)
    
    n_intervals = len(profile)
    interval_duration_hours = 24.0 / n_intervals
    
    total_cost = 0.0
    peak_cost = 0.0
    standard_cost = 0.0
    offpeak_cost = 0.0
    
    peak_energy = 0.0
    standard_energy = 0.0
    offpeak_energy = 0.0
    
    for idx, val in profile.items():
        # Represent time as hour of day (e.g. half-hourly indices 0-47 -> hours 0.0 to 23.5)
        hour = idx * interval_duration_hours
        rate = get_rate_for_hour(hour)
        cost = val * rate
        
        total_cost += cost
        if rate == config.TOU_PEAK_RATE:
            peak_cost += cost
            peak_energy += val
        elif rate == config.TOU_OFFPEAK_RATE:
            offpeak_cost += cost
            offpeak_energy += val
        else:
            standard_cost += cost
            standard_energy += val
            
    return {
        "total_cost": total_cost,
        "peak_cost": peak_cost,
        "standard_cost": standard_cost,
        "offpeak_cost": offpeak_cost,
        "peak_energy": peak_energy,
        "standard_energy": standard_energy,
        "offpeak_energy": offpeak_energy
    }

def simulate_peak_shift(profile: pd.Series, shift_pct: float) -> dict:
    """
    Simulate shifting a percentage of peak energy consumption to off-peak hours.
    
    Parameters:
    - profile: pd.Series indexable by half-hour index (0-47) or hour index (0-23)
               representing energy consumption (in kWh).
    - shift_pct: Float between 0.0 and 1.0 (e.g., 0.15 for 15% shift).
    
    Returns a dict with:
    - shifted_profile: pd.Series representing the modified energy profile
    - original_cost_details: dict from calculate_tou_cost of original profile
    - new_cost_details: dict from calculate_tou_cost of shifted profile
    - savings: Float representing cost savings in £
    - shifted_energy_kWh: Amount of energy shifted in kWh
    """
    # Ensure profile index is integer type
    profile = profile.copy()
    profile.index = profile.index.astype(int)
    
    n_intervals = len(profile)
    interval_duration_hours = 24.0 / n_intervals
    
    # Identify indices that correspond to peak and off-peak periods
    peak_indices = []
    offpeak_indices = []
    
    for idx in range(n_intervals):
        hour = idx * interval_duration_hours
        rate = get_rate_for_hour(hour)
        if rate == config.TOU_PEAK_RATE:
            peak_indices.append(idx)
        elif rate == config.TOU_OFFPEAK_RATE:
            offpeak_indices.append(idx)
            
    # Calculate original cost
    orig_details = calculate_tou_cost(profile)
    
    # Peak energy to shift
    peak_energy_total = orig_details["peak_energy"]
    shifted_energy_kWh = peak_energy_total * shift_pct
    
    # Create shifted profile
    shifted_profile = profile.copy()
    
    if peak_energy_total > 0 and len(peak_indices) > 0 and len(offpeak_indices) > 0:
        # Subtract from peak proportional to existing usage at each peak interval
        for idx in peak_indices:
            ratio = profile.loc[idx] / peak_energy_total
            shifted_profile.loc[idx] -= shifted_energy_kWh * ratio
            
        # Distribute the shifted energy to off-peak hours
        # Option A: Distribute evenly among off-peak intervals
        # Option B: Distribute proportional to existing off-peak consumption
        offpeak_total = orig_details["offpeak_energy"]
        if offpeak_total > 0:
            for idx in offpeak_indices:
                ratio = profile.loc[idx] / offpeak_total
                shifted_profile.loc[idx] += shifted_energy_kWh * ratio
        else:
            # If off-peak consumption is 0, distribute evenly
            energy_per_interval = shifted_energy_kWh / len(offpeak_indices)
            for idx in offpeak_indices:
                shifted_profile.loc[idx] += energy_per_interval
                
    # Calculate new cost
    new_details = calculate_tou_cost(shifted_profile)
    savings = orig_details["total_cost"] - new_details["total_cost"]
    
    return {
        "shifted_profile": shifted_profile,
        "original_cost_details": orig_details,
        "new_cost_details": new_details,
        "savings": savings,
        "shifted_energy_kWh": shifted_energy_kWh
    }

def generate_efficiency_insights(profile_summary: dict, cluster_name: str) -> list:
    """Generate personalized optimization recommendations based on usage patterns."""
    recommendations = []
    
    if "Eco-Saver" in cluster_name:
        recommendations.extend([
            "Your household has excellent base efficiency! Keep up the good work.",
            "Consider small behavioral adjustments such as washing clothes during standard or off-peak hours to save even more.",
            "Look into smart thermostats to automate heating/cooling schedules based on room occupancy."
        ])
    elif "High Load" in cluster_name:
        recommendations.extend([
            "Your household falls under the High Load consumer category. Transitioning to a Time-of-Use (ToU) tariff can offer substantial savings if you shift laundry, dishwashing, or EV charging.",
            "Heavy loads like electric vehicle (EV) charging or immersion heaters should be scheduled strictly during off-peak hours (00:00 - 07:00) when rates drop to £0.05/kWh.",
            "Consider upgrading older home appliances to energy star certified equivalents to lower baseline consumption.",
            "Install smart plugs and monitor standby energy usage ('vampire loads') on large entertainment centers or home office setups."
        ])
    elif "Comfort Consumer" in cluster_name:
        recommendations.extend([
            "Your usage shows peaky demand patterns, likely concentrated in the evening hours.",
            "A shift of just 15% of your evening kitchen or laundry load to off-peak hours can decrease your bill on a ToU plan.",
            "Investigate thermal energy storage solutions or pre-heating/cooling your living spaces before the 16:00 - 20:00 peak period starts."
        ])
    else: # Standard Saver
        recommendations.extend([
            "Your consumption is moderate and relatively stable. A Time-of-Use tariff may offer mild savings over a flat rate plan.",
            "Verify if your water heater or main heating runs on a timer. Restrict usage during peak evening hours (16:00 - 20:00) where possible.",
            "Replace standard light bulbs with energy-efficient LED alternatives across frequently used rooms."
        ])
        
    return recommendations
