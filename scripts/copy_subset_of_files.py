import shutil
import os
import numpy as np

images_path = "/home/ubuntu/tamir/datasets/IL_weekend/64eef8f8ac243656c2bd2b76"
images_names = os.listdir(images_path)
print(len(images_names))

chosen_images_names = np.random.choice(images_names, 500, replace=False)
print(len(chosen_images_names))

dst_path = "/home/ubuntu/tamir/workspace/AB_AutoTagging_Lanes/tmp/single_frame/data_500"
for name in chosen_images_names:
    src_path = f"{images_path}/{name}"
    shutil.copy(src_path, dst_path)
    
print('Copied')