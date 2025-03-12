import numpy as np
import os
from data import patches_extraction
from multiprocessing import Pool


def preprocessing(dataset_npz, pc_path, filename, index):
    # perform preprocessing and save all data
    x_b, x_s = patches_extraction(filename = pc_path, number_patches = 16, points_per_patch = 14900, small_points_per_patch = 8192)
    np.savez(os.path.join(dataset_npz, filename), x_b=x_b, x_s=x_s)
    print("{}) {} saved ".format(index, os.path.join(dataset_npz, filename) ))

if __name__ == "__main__":
    dataset_ply_path = "data/WPC/distorted"
    destination_npz_path = "data/WPC/distorted_npz_16"
    list_pc = [os.path.join(dataset_ply_path, f) for f in os.listdir(dataset_ply_path)]
    filenames = [f.split("\\")[1][:-4] for f in list_pc]


    print(f'There are {len(list_pc)} files waiting for process... ')
    print(f'USE {12} CPU core to process the task in parallel.')

    # now multithread
    pool = Pool(12)  
    for i, pc_path in enumerate(list_pc):
        pool.apply_async(func=preprocessing, args=(destination_npz_path, pc_path, filenames[i], i))
    pool.close()     # close process pool
    pool.join()      # waits for all the child process      
