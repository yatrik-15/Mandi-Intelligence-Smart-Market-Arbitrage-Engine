"""
Distance Calculator for Gujarat Mandis
======================================

Calculates distances between any two locations in Gujarat.
Uses a distance matrix of major cities and mandis.
"""

import math
from typing import Dict, Tuple

# Distance matrix: distances in km between major Gujarat locations
# Based on actual road distances via major highways
DISTANCE_MATRIX = {
    # Format: (city1, city2): distance_km
    ('Gandhinagar', 'Ahmedabad'): 26,
    ('Gandhinagar', 'Mehsana'): 62,
    ('Gandhinagar', 'Rajkot'): 237,
    ('Gandhinagar', 'Surat'): 273,
    ('Gandhinagar', 'Anand'): 98,
    ('Gandhinagar', 'Bharuch'): 211,
    ('Gandhinagar', 'Amreli'): 277,
    ('Gandhinagar', 'Vadodara'): 100,
    ('Gandhinagar', 'Bhavnagar'): 200,
    ('Gandhinagar', 'Jamnagar'): 330,
    ('Gandhinagar', 'Junagadh'): 330,
    
    ('Ahmedabad', 'Mehsana'): 64,
    ('Ahmedabad', 'Rajkot'): 216,
    ('Ahmedabad', 'Surat'): 263,
    ('Ahmedabad', 'Anand'): 89,
    ('Ahmedabad', 'Bharuch'): 192,
    ('Ahmedabad', 'Vadodara'): 110,
    ('Ahmedabad', 'Bhavnagar'): 189,
    ('Ahmedabad', 'Jamnagar'): 315,
    ('Ahmedabad', 'Amreli'): 255,
    
    ('Rajkot', 'Surat'): 296,
    ('Rajkot', 'Jamnagar'): 92,
    ('Rajkot', 'Bhavnagar'): 165,
    ('Rajkot', 'Amreli'): 117,
    ('Rajkot', 'Junagadh'): 104,
    
    ('Surat', 'Vadodara'): 145,
    ('Surat', 'Bharuch'): 66,
    ('Surat', 'Anand'): 176,
    
    ('Vadodara', 'Anand'): 46,
    ('Vadodara', 'Bharuch'): 66,
    ('Vadodara', 'Mehsana'): 148,
    
    # Add more as needed
}

# City aliases and standardization
CITY_ALIASES = {
    'gandhinagar': 'Gandhinagar',
    'ahmedabad': 'Ahmedabad',
    'amd': 'Ahmedabad',
    'mehsana': 'Mehsana',
    'mahesana': 'Mehsana',
    'rajkot': 'Rajkot',
    'surat': 'Surat',
    'anand': 'Anand',
    'bharuch': 'Bharuch',
    'vadodara': 'Vadodara',
    'baroda': 'Vadodara',
    'bhavnagar': 'Bhavnagar',
    'jamnagar': 'Jamnagar',
    'amreli': 'Amreli',
    'junagadh': 'Junagadh',
}


def standardize_location(location: str) -> str:
    """Standardize location name"""
    if not location:
        return 'Gandhinagar'
    
    location_lower = location.lower().strip()
    return CITY_ALIASES.get(location_lower, location.title())


def get_distance(from_location: str, to_location: str) -> float:
    """
    Get distance between two locations in km.
    
    Args:
        from_location: Starting location
        to_location: Destination location
        
    Returns:
        Distance in kilometers
    """
    from_loc = standardize_location(from_location)
    to_loc = standardize_location(to_location)
    
    # Same location
    if from_loc == to_loc:
        return 0.0
    
    # Try direct lookup
    key1 = (from_loc, to_loc)
    key2 = (to_loc, from_loc)
    
    if key1 in DISTANCE_MATRIX:
        return DISTANCE_MATRIX[key1]
    elif key2 in DISTANCE_MATRIX:
        return DISTANCE_MATRIX[key2]
    
    # If not found, try to estimate using triangulation
    # Find common city and calculate via that route
    estimated = estimate_distance_via_hub(from_loc, to_loc)
    if estimated:
        return estimated
    
    # Fallback: use approximate estimation based on coordinates
    return estimate_distance_fallback(from_loc, to_loc)


def estimate_distance_via_hub(from_loc: str, to_loc: str) -> float:
    """
    Estimate distance by routing through a hub city (Ahmedabad or Gandhinagar).
    """
    hubs = ['Gandhinagar', 'Ahmedabad', 'Vadodara']
    
    for hub in hubs:
        # Try to find route: from_loc -> hub -> to_loc
        dist1_key1 = (from_loc, hub)
        dist1_key2 = (hub, from_loc)
        dist2_key1 = (hub, to_loc)
        dist2_key2 = (to_loc, hub)
        
        dist1 = DISTANCE_MATRIX.get(dist1_key1) or DISTANCE_MATRIX.get(dist1_key2)
        dist2 = DISTANCE_MATRIX.get(dist2_key1) or DISTANCE_MATRIX.get(dist2_key2)
        
        if dist1 and dist2:
            # Route via hub (add 10% overhead for non-direct route)
            return (dist1 + dist2) * 1.1
    
    return None


def estimate_distance_fallback(from_loc: str, to_loc: str) -> float:
    """
    Fallback distance estimation.
    Uses average distance to Gandhinagar as reference.
    """
    # Get distances to Gandhinagar for both locations
    from_to_gn = get_distance_to_gandhinagar(from_loc)
    to_to_gn = get_distance_to_gandhinagar(to_loc)
    
    # Rough estimate using triangle inequality
    # This is very approximate but better than nothing
    return abs(from_to_gn - to_to_gn) if from_to_gn and to_to_gn else 150.0


def get_distance_to_gandhinagar(location: str) -> float:
    """Get distance from any location to Gandhinagar"""
    loc = standardize_location(location)
    
    if loc == 'Gandhinagar':
        return 0.0
    
    # Check direct distance
    key1 = ('Gandhinagar', loc)
    key2 = (loc, 'Gandhinagar')
    
    return DISTANCE_MATRIX.get(key1) or DISTANCE_MATRIX.get(key2) or None


def calculate_distances_from_location(farmer_location: str, mandis: list) -> Dict[str, float]:
    """
    Calculate distances from farmer's location to all mandis.
    
    Args:
        farmer_location: Farmer's current location
        mandis: List of mandi names
        
    Returns:
        Dictionary mapping mandi_name -> distance in km
    """
    distances = {}
    
    for mandi in mandis:
        distances[mandi] = get_distance(farmer_location, mandi)
    
    return distances


# Example usage
if __name__ == "__main__":
    print("Distance Calculator Test")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        ('Gandhinagar', 'Ahmedabad'),
        ('Ahmedabad', 'Rajkot'),
        ('Surat', 'Rajkot'),
        ('Ahmedabad', 'Surat'),
        ('Gandhinagar', 'Rajkot'),
    ]
    
    for from_loc, to_loc in test_cases:
        distance = get_distance(from_loc, to_loc)
        print(f"{from_loc} → {to_loc}: {distance:.1f} km")
    
    print("\n" + "=" * 50)
    print("Distances from Ahmedabad to all mandis:")
    mandis = ['Rajkot', 'Surat', 'Mehsana', 'Anand', 'Vadodara']
    distances = calculate_distances_from_location('Ahmedabad', mandis)
    for mandi, dist in distances.items():
        print(f"  {mandi}: {dist:.1f} km")
