"""
Keypoint mapping utilities for 32-keypoint system.

This module provides utilities to convert between the full 57-keypoint system
and the reduced 32-keypoint system used in the project.

Usage:
    from keypoint_mapping_32 import convert_57_to_32, convert_32_to_57
    
    # Convert detected keypoints from 57-system to 32-system
    kp_dict_57 = {1: {...}, 5: {...}, 28: {...}, ...}  # Original detection
    kp_dict_32 = convert_57_to_32(kp_dict_57)  # Now: {6: {...}, 13: {...}, 1: {...}, ...}
"""

import numpy as np
from typing import Dict, Optional, List


# Mapping: new_keypoint_id (1-32) -> original_keypoint_id (1-57)
# CORRECTED: Fixed horizontal flip based on user testing (verified Dec 1, 2025)
KEYPOINT_32_TO_57 = {
    1: 1,   2: 4,  3: 8,   4: 20,  5: 24,  6: 28,  7: 9,  8: 21,
    9: 45,  10: 5,  11: 31, 12: 34, 13: 25, 14: 2,  15: 32, 16: 35,
    17: 29, 18: 6,  19: 33, 20: 36, 21: 26, 22: 57, 23: 10, 24: 22,
    25: 3,  26: 7,  27: 11, 28: 23, 29: 27, 30: 30, 31: 50, 32: 52
}

# Reverse mapping: original_keypoint_id (1-57) -> new_keypoint_id (1-32)
# CORRECTED: Auto-generated from forward mapping (verified Dec 1, 2025)
# Note: Original ID 20 maps to new ID 4 (positions 4 and 5 both map to original 20)
KEYPOINT_57_TO_32 = {
    1: 1,   2: 14,  3: 25, 4: 2, 5: 10,  6: 18,  7: 26,  8: 3, 9: 7,  
    10: 23, 11: 27, 14: 2,  19: 7,  20: 4,  21: 8,  22: 24, 23: 28, 24: 5, 25: 13,
    26: 21, 27: 29, 28: 6,  29: 17, 30: 30, 31: 11, 32: 15, 33: 19,
    34: 12, 35: 16, 36: 20, 45: 9,  50: 31, 52: 32, 57: 22
}


def convert_57_to_32(kp_dict_57: Dict[int, dict]) -> Dict[int, dict]:
    """
    Convert keypoint dictionary from 57-keypoint system to 32-keypoint system.
    
    Args:
        kp_dict_57: Dictionary with keypoint IDs from 57-system as keys
                   Example: {1: {'x': 100, 'y': 200, 'p': 0.9}, ...}
    
    Returns:
        Dictionary with keypoint IDs from 32-system as keys
        Only includes keypoints that are in the 32-keypoint system
        
    Example:
        >>> kp_57 = {1: {'x': 0, 'y': 0, 'p': 0.9}, 28: {'x': 0, 'y': 68, 'p': 0.95}}
        >>> kp_32 = convert_57_to_32(kp_57)
        >>> print(kp_32)
        {6: {'x': 0, 'y': 0, 'p': 0.9}, 1: {'x': 0, 'y': 68, 'p': 0.95}}
    """
    kp_dict_32 = {}
    
    for orig_id, kp_data in kp_dict_57.items():
        # Check if this keypoint is in the 32-keypoint system
        new_id = KEYPOINT_57_TO_32.get(orig_id)
        
        if new_id is not None:
            # Copy keypoint data with new ID
            kp_dict_32[new_id] = kp_data.copy()
            
            # Optionally store original ID for reference
            kp_dict_32[new_id]['original_id'] = orig_id
    
    return kp_dict_32


def convert_32_to_57(kp_dict_32: Dict[int, dict]) -> Dict[int, dict]:
    """
    Convert keypoint dictionary from 32-keypoint system to 57-keypoint system.
    
    Args:
        kp_dict_32: Dictionary with keypoint IDs from 32-system as keys
                   Example: {1: {'x': 100, 'y': 200, 'p': 0.9}, ...}
    
    Returns:
        Dictionary with keypoint IDs from 57-system as keys
        
    Example:
        >>> kp_32 = {1: {'x': 0, 'y': 68, 'p': 0.95}, 6: {'x': 0, 'y': 0, 'p': 0.9}}
        >>> kp_57 = convert_32_to_57(kp_32)
        >>> print(kp_57)
        {28: {'x': 0, 'y': 68, 'p': 0.95}, 1: {'x': 0, 'y': 0, 'p': 0.9}}
    """
    kp_dict_57 = {}
    
    for new_id, kp_data in kp_dict_32.items():
        # Get original ID
        orig_id = KEYPOINT_32_TO_57.get(new_id)
        
        if orig_id is not None:
            # Copy keypoint data with original ID
            kp_dict_57[orig_id] = kp_data.copy()
            
            # Remove 'original_id' field if it exists (cleanup)
            kp_dict_57[orig_id].pop('original_id', None)
    
    return kp_dict_57


def filter_to_32_keypoints(kp_dict_57: Dict[int, dict], 
                           renumber: bool = True) -> Dict[int, dict]:
    """
    Filter keypoints to only include those in the 32-keypoint system.
    
    Args:
        kp_dict_57: Dictionary with keypoint IDs from 57-system
        renumber: If True, renumber to 32-system. If False, keep original IDs.
    
    Returns:
        Filtered dictionary (with new or original IDs based on renumber)
        
    Example:
        >>> kp_57 = {1: {...}, 5: {...}, 12: {...}, 28: {...}}  # 12 not in 32-system
        >>> filtered = filter_to_32_keypoints(kp_57, renumber=False)
        >>> print(list(filtered.keys()))
        [1, 5, 28]  # 12 removed
    """
    if renumber:
        return convert_57_to_32(kp_dict_57)
    else:
        # Keep only keypoints that are in the 32-system, but keep original IDs
        return {k: v for k, v in kp_dict_57.items() if k in KEYPOINT_57_TO_32}


def get_missing_keypoints_32(kp_dict_32: Dict[int, dict]) -> List[int]:
    """
    Get list of missing keypoint IDs (from 32-keypoint system).
    
    Args:
        kp_dict_32: Dictionary with detected keypoints (32-system IDs)
    
    Returns:
        List of missing keypoint IDs (1-32)
        
    Example:
        >>> kp_32 = {1: {...}, 5: {...}, 10: {...}}
        >>> missing = get_missing_keypoints_32(kp_32)
        >>> print(missing)
        [2, 3, 4, 6, 7, 8, 9, 11, 12, ..., 32]
    """
    all_ids = set(range(1, 33))
    detected_ids = set(kp_dict_32.keys())
    missing_ids = sorted(all_ids - detected_ids)
    
    return missing_ids


def get_detection_rate_32(kp_dict_32: Dict[int, dict]) -> float:
    """
    Calculate detection rate for 32-keypoint system.
    
    Args:
        kp_dict_32: Dictionary with detected keypoints (32-system IDs)
    
    Returns:
        Detection rate as percentage (0-100)
        
    Example:
        >>> kp_32 = {1: {...}, 5: {...}, 10: {...}}  # 3 detected
        >>> rate = get_detection_rate_32(kp_32)
        >>> print(f"{rate:.1f}%")
        9.4%  # 3/32 = 9.375%
    """
    detected = len(kp_dict_32)
    total = 32
    return (detected / total) * 100


def is_keypoint_in_32_system(orig_id: int) -> bool:
    """
    Check if an original keypoint ID (1-57) is in the 32-keypoint system.
    
    Args:
        orig_id: Original keypoint ID (1-57)
    
    Returns:
        True if keypoint is in 32-system, False otherwise
        
    Example:
        >>> is_keypoint_in_32_system(1)
        True
        >>> is_keypoint_in_32_system(12)  # Goal post - not in 32-system
        False
    """
    return orig_id in KEYPOINT_57_TO_32


def get_32_keypoint_info(new_id: int) -> Optional[Dict]:
    """
    Get information about a keypoint in the 32-system.
    
    Args:
        new_id: New keypoint ID (1-32)
    
    Returns:
        Dictionary with info about the keypoint, or None if invalid ID
        
    Example:
        >>> info = get_32_keypoint_info(1)
        >>> print(info)
        {'new_id': 1, 'original_id': 28, 'description': 'Top-left corner'}
    """
    if new_id not in KEYPOINT_32_TO_57:
        return None
    
    orig_id = KEYPOINT_32_TO_57[new_id]
    
    # Descriptions (subset from the full list)
    descriptions = {
        1: "Bottom-left corner", 2: "Bottom center", 3: "Bottom-right corner",
        4: "Left goal line, penalty area bottom", 5: "Left penalty area bottom-left",
        6: "Right penalty area bottom-left", 7: "Right goal line, penalty area bottom",
        8: "Left goal line, goal area bottom", 9: "Left goal area bottom-left",
        10: "Right goal area bottom-left", 11: "Right goal line, goal area bottom",
        20: "Left goal line, goal area top", 21: "Left goal area top-left",
        22: "Right goal area top-left", 23: "Right goal line, goal area top",
        24: "Left goal line, penalty area top", 25: "Left penalty area top-left",
        26: "Right penalty area top-left", 27: "Right goal line, penalty area top",
        28: "Top-left corner", 29: "Top center", 30: "Top-right corner",
        31: "Left penalty arc - bottom", 32: "Center circle - bottom",
        33: "Right penalty arc - bottom", 34: "Left penalty arc - top",
        35: "Center circle - top", 36: "Right penalty arc - top",
        45: "Left penalty spot", 50: "Center - left", 52: "Center - right",
        57: "Right penalty spot"
    }
    
    return {
        'new_id': new_id,
        'original_id': orig_id,
        'description': descriptions.get(orig_id, f"Keypoint {orig_id}")
    }


def print_conversion_example():
    """Print example of conversion between systems."""
    print("\n" + "="*80)
    print("KEYPOINT CONVERSION EXAMPLE")
    print("="*80)
    
    # Example: Detected keypoints in 57-system
    kp_57 = {
        1: {'x': 0.0, 'y': 0.0, 'p': 0.95},
        28: {'x': 0.0, 'y': 68.0, 'p': 0.92},
        45: {'x': 11.0, 'y': 34.0, 'p': 0.88},
        12: {'x': 0.0, 'y': 30.34, 'p': 0.75}  # Goal post - NOT in 32-system
    }
    
    print("\nOriginal detections (57-system):")
    for orig_id, data in kp_57.items():
        in_32 = "✓" if is_keypoint_in_32_system(orig_id) else "✗"
        print(f"  {in_32} ID {orig_id:2d}: ({data['x']:6.2f}, {data['y']:6.2f}) confidence={data['p']:.2f}")
    
    # Convert to 32-system
    kp_32 = convert_57_to_32(kp_57)
    
    print("\nConverted to 32-system:")
    for new_id, data in kp_32.items():
        orig_id = data.get('original_id', '?')
        print(f"  New ID {new_id:2d} (was {orig_id:2d}): ({data['x']:6.2f}, {data['y']:6.2f}) confidence={data['p']:.2f}")
    
    print(f"\nDetection rate: {get_detection_rate_32(kp_32):.1f}% ({len(kp_32)}/32 keypoints)")
    
    missing = get_missing_keypoints_32(kp_32)
    print(f"Missing keypoints: {missing[:10]}..." if len(missing) > 10 else f"Missing keypoints: {missing}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Print example
    print_conversion_example()
    
    # Print summary
    print("KEYPOINT MAPPING SUMMARY")
    print("="*80)
    print(f"Total keypoints in full system: 57")
    print(f"Total keypoints in reduced system: 32")
    print(f"Unused keypoints: {57 - 32}")
    print(f"\nTo use in your code:")
    print("  from keypoint_mapping_32 import convert_57_to_32")
    print("  kp_32 = convert_57_to_32(kp_dict_57)")
    print("="*80)

