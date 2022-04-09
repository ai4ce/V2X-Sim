import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import imageio
from sklearn.feature_extraction import image

vis_path = './logs/visualization'
scene_count = 20
agent_count = 6
frame_count = 100

for s in range(76, 77):
    images = np.array([])
    for a in range(agent_count):
        current_gif = f'{vis_path}/agent{a}/scene_{s}/gif/out.gif'
        print(current_gif)
        if not os.path.isfile(current_gif):
            images = np.append(images, None)
        else:
            images = np.append(images, imageio.get_reader(current_gif))
    new_gif = imageio.get_writer(f'./logs/visualization/all/scene_{s}.gif')


    for f in range(frame_count):
        frames = images.copy()
        for ii in range(agent_count):
            if images[ii] != None:
                frames[ii] = images[ii].get_next_data()
            else:
                frames[ii] = np.full((504, 811, 4), fill_value=255, dtype=np.uint8) # white
        
        r1 = np.hstack((frames[0], frames[1], frames[2]))
        r2 = np.hstack((frames[3], frames[4], frames[5]))

        new_image = np.vstack((r1, r2))
    
        new_gif.append_data(new_image)
