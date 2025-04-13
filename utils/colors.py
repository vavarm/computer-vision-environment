def generate_color_from_id(id):
    # A predefined set of colors based on the ID (just a few examples)
    color_map = {
        1: (0.1, 0.2, 0.5),  # Blue
        2: (0.7, 0.2, 0.1),  # Red
        3: (0.1, 0.7, 0.1),  # Green
        4: (0.9, 0.7, 0.1),  # Yellow
        5: (0.4, 0.4, 0.8),  # Light Blue
        6: (0.9, 0.3, 0.7),  # Pink
        7: (0.2, 0.9, 0.4),  # Light Green
        8: (0.8, 0.8, 0.8)  # Gray
    }

    # Return a color based on the ID, defaulting to a gray if the ID isn't in the map
    return color_map.get(id, (0.5, 0.5, 0.5))  # Default to gray if ID not found