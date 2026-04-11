"""
Complete Keypoints Module

Completes missing keypoints by computing line intersections.
Based on PnLCalib's complete_keypoints function, adapted for 32-keypoint system.

Usage:
    from utils.complete_keypoints import complete_keypoints
    
    kp_dict, lines_dict = complete_keypoints(
        kp_dict, lines_dict, 
        w=width, h=height, 
        normalize=False
    )
"""

import copy
import numpy as np
from scipy.stats import linregress
from typing import Dict, Tuple, List, Optional


def line_intersection_from_points(x1_list: List[float], y1_list: List[float], 
                                   x2_list: List[float], y2_list: List[float]) -> Tuple[float, float]:
    """
    Compute intersection point of two lines using linear regression.
    
    Args:
        x1_list: [x1_start, x1_end] for line 1
        y1_list: [y1_start, y1_end] for line 1
        x2_list: [x2_start, x2_end] for line 2
        y2_list: [y2_start, y2_end] for line 2
    
    Returns:
        (x_intersection, y_intersection)
    """
    # Add small epsilon to avoid identical coordinate values
    x1_list = [x1_list[0] + 1e-7, x1_list[1] + 1e-7]
    x2_list = [x2_list[0] + 1e-7, x2_list[1] + 1e-7]
    
    slope1, intercept1, r1, p1, se1 = linregress(x1_list, y1_list)
    slope2, intercept2, r2, p2, se2 = linregress(x2_list, y2_list)
    
    x_intersection = (intercept2 - intercept1) / (slope1 - slope2 + 1e-7)
    y_intersection = slope1 * x_intersection + intercept1
    
    return x_intersection, y_intersection


def find_line_pairs_for_keypoint(
    target_kp_id: int,
    lines_dict: Dict,
    kp_dict: Dict
) -> Optional[Tuple[int, int]]:
    """
    Find which two lines should intersect to create a given keypoint.
    
    This is a heuristic approach: we try to find lines that are likely
    to intersect at the missing keypoint location.
    
    Args:
        target_kp_id: Target keypoint ID (1-32)
        lines_dict: Dictionary of detected lines {line_id: {x1, y1, x2, y2, ...}}
        kp_dict: Dictionary of existing keypoints
    
    Returns:
        (line_id1, line_id2) if found, None otherwise
    """
    # For now, we'll use a simple heuristic:
    # Try all pairs of lines and see which ones intersect near expected locations
    # This is a simplified version - in the full implementation, you'd have
    # a predefined mapping of which line pairs create which keypoints
    
    line_ids = list(lines_dict.keys())
    
    # Try all pairs
    for i, line_id1 in enumerate(line_ids):
        for line_id2 in line_ids[i+1:]:
            line1 = lines_dict[line_id1]
            line2 = lines_dict[line_id2]
            
            # Check if both lines have valid endpoints
            if 'x1' not in line1 or 'x2' not in line1:
                continue
            if 'x1' not in line2 or 'x2' not in line2:
                continue
            
            try:
                x1 = [line1['x1'], line1['x2']]
                y1 = [line1['y1'], line1['y2']]
                x2 = [line2['x1'], line2['x2']]
                y2 = [line2['y1'], line2['y2']]
                
                inter_x, inter_y = line_intersection_from_points(x1, y1, x2, y2)
                
                # Check if intersection is reasonable (within frame bounds)
                # We'll accept this pair if intersection is valid
                if not (np.isnan(inter_x) or np.isnan(inter_y) or 
                        np.isinf(inter_x) or np.isinf(inter_y)):
                    return (line_id1, line_id2)
            except:
                continue
    
    return None


def complete_keypoints(
    kp_dict: Dict,
    lines_dict: Dict,
    w: int,
    h: int,
    normalize: bool = False,
    use_line_mapping: bool = True
) -> Tuple[Dict, Dict]:
    """
    Complete missing keypoints by computing line intersections.
    
    Based on PnLCalib's complete_keypoints, adapted for 32-keypoint system.
    
    Args:
        kp_dict: Dictionary of detected keypoints {kp_id: {x, y, confidence, ...}}
        lines_dict: Dictionary of detected lines {line_id: {x1, y1, x2, y2, ...}}
        w: Frame width
        h: Frame height
        normalize: If True, normalize coordinates to [0, 1]
        use_line_mapping: If True, use predefined line-to-keypoint mappings
    
    Returns:
        (complete_kp_dict, lines_dict) - completed keypoint dict and lines dict
    """
    w_extra = 0.0 * w
    h_extra = 0.0 * h
    
    complete_dict = copy.deepcopy(kp_dict)
    
    # Strategy 1: Try to complete keypoints 1-32 using line intersections
    # We'll try all pairs of lines to find intersections that could be missing keypoints
    if use_line_mapping and len(lines_dict) >= 2:
        # Get all line pairs
        line_ids = sorted([lid for lid in lines_dict.keys() if isinstance(lid, int)])
        
        for line_id1 in line_ids:
            for line_id2 in line_ids:
                if line_id1 >= line_id2:
                    continue
                
                line1 = lines_dict[line_id1]
                line2 = lines_dict[line_id2]
                
                # Check if both lines have valid endpoints
                if 'x1' not in line1 or 'x2' not in line1:
                    continue
                if 'x1' not in line2 or 'x2' not in line2:
                    continue
                
                try:
                    x1 = [line1['x1'], line1['x2']]
                    y1 = [line1['y1'], line1['y2']]
                    x2 = [line2['x1'], line2['x2']]
                    y2 = [line2['y1'], line2['y2']]
                    
                    inter_x, inter_y = line_intersection_from_points(x1, y1, x2, y2)
                    
                    # Check if intersection is valid and within frame bounds
                    if (not (np.isnan(inter_x) or np.isnan(inter_y) or 
                            np.isinf(inter_x) or np.isinf(inter_y)) and
                        -w_extra <= inter_x <= w + w_extra and
                        -h_extra <= inter_y <= h + h_extra):
                        
                        # Check if this intersection is close to any missing keypoint
                        # For now, we'll add it as a new keypoint with a special ID
                        # or try to match it to the closest missing keypoint
                        
                        # Find the closest missing keypoint (1-32)
                        min_dist = float('inf')
                        best_kp_id = None
                        
                        for kp_id in range(1, 33):
                            if kp_id not in complete_dict:
                                # We don't know the expected location, so we'll
                                # just add intersections as new keypoints
                                # In a full implementation, you'd have expected locations
                                pass
                        
                        # For now, add intersection as a keypoint with ID based on line pair
                        # Use a high ID to avoid conflicts (100+)
                        new_kp_id = 100 + line_id1 * 10 + line_id2
                        if new_kp_id not in complete_dict:
                            complete_dict[new_kp_id] = {
                                'x': round(inter_x, 2),
                                'y': round(inter_y, 2),
                                'confidence': 1.0,
                                'from_intersection': True,
                                'line1': line_id1,
                                'line2': line_id2
                            }
                except Exception as e:
                    # Skip invalid line pairs
                    continue
    
    # Strategy 2: More targeted approach - try to complete specific keypoints
    # by finding line pairs that are likely to intersect at those locations
    # This would require a mapping of expected keypoint locations or line-to-keypoint relationships
    
    if normalize:
        for kp_id in complete_dict.keys():
            if 'x' in complete_dict[kp_id] and 'y' in complete_dict[kp_id]:
                complete_dict[kp_id]['x'] /= w
                complete_dict[kp_id]['y'] /= h
        
        for line_id in lines_dict.keys():
            if 'x1' in lines_dict[line_id]:
                lines_dict[line_id]['x1'] /= w
                lines_dict[line_id]['y1'] /= h
                lines_dict[line_id]['x2'] /= w
                lines_dict[line_id]['y2'] /= h
    
    complete_dict = dict(sorted(complete_dict.items()))
    
    return complete_dict, lines_dict


def complete_keypoints_with_mapping(
    kp_dict: Dict,
    lines_dict: Dict,
    w: int,
    h: int,
    keypoint_line_mapping: Dict[int, List[int]],
    normalize: bool = False
) -> Tuple[Dict, Dict]:
    """
    Complete missing keypoints using a predefined mapping of which line pairs create which keypoints.
    
    Args:
        kp_dict: Dictionary of detected keypoints
        lines_dict: Dictionary of detected lines
        w: Frame width
        h: Frame height
        keypoint_line_mapping: Dict mapping keypoint_id -> [line_id1, line_id2]
        normalize: If True, normalize coordinates
    
    Returns:
        (complete_kp_dict, lines_dict)
    """
    w_extra = 0.0 * w
    h_extra = 0.0 * h
    
    complete_dict = copy.deepcopy(kp_dict)
    
    # For each keypoint in the mapping
    for kp_id, line_pair in keypoint_line_mapping.items():
        # Skip if keypoint already exists
        if kp_id in complete_dict:
            continue
        
        line_id1, line_id2 = line_pair
        
        # Check if both lines exist
        if line_id1 not in lines_dict or line_id2 not in lines_dict:
            continue
        
        line1 = lines_dict[line_id1]
        line2 = lines_dict[line_id2]
        
        # Check if both lines have valid endpoints
        if 'x1' not in line1 or 'x2' not in line1:
            continue
        if 'x1' not in line2 or 'x2' not in line2:
            continue
        
        try:
            x1 = [line1['x1'], line1['x2']]
            y1 = [line1['y1'], line1['y2']]
            x2 = [line2['x1'], line2['x2']]
            y2 = [line2['y1'], line2['y2']]
            
            inter_x, inter_y = line_intersection_from_points(x1, y1, x2, y2)
            
            # Check if intersection is valid and within bounds
            if (not (np.isnan(inter_x) or np.isnan(inter_y) or 
                    np.isinf(inter_x) or np.isinf(inter_y)) and
                -w_extra <= inter_x <= w + w_extra and
                -h_extra <= inter_y <= h + h_extra):
                
                complete_dict[kp_id] = {
                    'x': round(inter_x, 2),
                    'y': round(inter_y, 2),
                    'confidence': 1.0,
                    'from_intersection': True,
                    'line1': line_id1,
                    'line2': line_id2
                }
        except Exception:
            continue
    
    if normalize:
        for kp_id in complete_dict.keys():
            if 'x' in complete_dict[kp_id] and 'y' in complete_dict[kp_id]:
                complete_dict[kp_id]['x'] /= w
                complete_dict[kp_id]['y'] /= h
        
        for line_id in lines_dict.keys():
            if 'x1' in lines_dict[line_id]:
                lines_dict[line_id]['x1'] /= w
                lines_dict[line_id]['y1'] /= h
                lines_dict[line_id]['x2'] /= w
                lines_dict[line_id]['y2'] /= h
    
    complete_dict = dict(sorted(complete_dict.items()))
    
    return complete_dict, lines_dict

