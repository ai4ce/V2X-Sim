class_to_rgb = {
    0: [255, 255, 255],  # Unlabeled
    1: [71, 141, 230],  # Vehicles
    2: [122, 217, 209],  # Sidewalk
    3: [145, 171, 100],  # Ground / Terrain
    4: [231, 136, 101],  # Road / Traffic light / Pole
    5: [142, 80, 204],  # Buildings
    6: [224, 8, 50],  # Pedestrian
    7: [106, 142, 34]  # Vegetation
    # 7: [102, 102, 156],  # Walls
    # 0: [55, 90, 80],  # Other
}

# Remap pixel values given by carla
classes_remap = {
    0: 0,  # Unlabeled (so that we don't forget this class)
    10: 1,  # Vehicles
    8: 2,  # Sidewalk
    14: 3,  # Ground (non-drivable)
    22: 3,  # Terrain (non-drivable)
    7: 4,  # Road
    6: 4,  # Road line
    18: 4,  # Traffic light
    5: 4,  # Pole
    1: 5,  # Building
    4: 6,  # Pedestrian
    9: 7,  # Vegetation
}

class_idx_to_name = {
    0: 'Unlabeled',
    1: 'Vehicles',
    2: 'Sidewalk',
    3: 'Ground & Terrain',
    4: 'Road',
    5: 'Buildings',
    6: 'Pedestrian',
    7: 'Vegetation'
}