import torch
import numpy as np
import pandas as pd                   
from sklearn.neighbors import KDTree
import open3d as o3d
from os.path import join
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

###### UTILS
def xyz_1_2001(xyz):
    """
    Input:
        xyz: pointcloud data, [N, D], 
    Return:
        new_xyz: normalized from 1 to 2001 point cloud [N, D]
    """
    new_xyz = np.copy(xyz)
    new_xyz = new_xyz - new_xyz.min()
    scale = xyz.max() - xyz.min()    
    new_xyz = new_xyz / scale         
    new_xyz = new_xyz*2000 + 1        
    return new_xyz

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D], 
        npoint: number of samples (1024)
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N = point.shape[0]                  
    xyz = point[:,:3]                   
    centroids = np.zeros((npoint,))    
    distance = np.ones((N,)) * 1e10    
    farthest = np.random.randint(0, N)  
    for i in range(npoint):             
        centroids[i] = farthest       
        centroid = xyz[farthest, :]     
        dist = np.sum((xyz - centroid) ** 2, -1)   
        mask = dist < distance                     
        distance[mask] = dist[mask]                
        farthest = np.argmax(distance, -1)         
    point = point[centroids.astype(np.int32)]      
    return point  

def patches_extraction(filename, number_patches, points_per_patch, small_points_per_patch):
    # read it
    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points)                         
    colors = np.asarray(pcd.colors)*255
    # pc definition
    pc = np.concatenate((points, colors), axis=1)
    # normalize xyz to 1-2001
    pc[:,0:3] = xyz_1_2001(pc[:,0:3])        
    points = pc[:,0:3]
    kd_tree = KDTree(points)
    centers = farthest_point_sample(pc, number_patches)
    # compute patches
    patches_big = []
    patches_small = []
    for center in centers:
        center = center[0:3]
        _, knn_idx_big = kd_tree.query(X = [center], k = points_per_patch)   
        _, knn_idx_small = kd_tree.query(X = [center], k = small_points_per_patch)   
        kpoint_big = pc[knn_idx_big[0]][:,:6]
        patches_big.append(kpoint_big)
        kpoint_small = pc[knn_idx_small[0]][:,:6]
        patches_small.append(kpoint_small)
    return np.array(patches_big), np.array(patches_small)

###### TORCH DATASETS #####

######### ABSTRACT CLASS #########################################################################################################################################################################

class Pointcloud_Dataset(Dataset):

    def __init__(self, pd_db:pd.DataFrame, path_ply:str, number_patches:int, points_structure:int, train:bool = False) -> None:
        self.pd_db = pd_db
        self.path_ply = path_ply
        self.number_patches = number_patches        # also known as "number of centers"
        self.points_structure = points_structure    # 1024
        self.train = train

    # abstract method
    def __getitem__(self, index) -> list[torch.tensor, torch.tensor, torch.tensor]:
        raise NotImplementedError

    def __len__(self)-> int:
        return self.pd_db.shape[0]

#################################################################################################################################################################################################
    
class WPC(Pointcloud_Dataset):

    def __init__(self, pd_db: pd.DataFrame, path_ply: str, number_patches: int, points_structure: int, train:bool = False) -> None:
        super().__init__(pd_db, path_ply, number_patches, points_structure, train)
        self.max_mos = 100
        self.min_mos = 0
        

    def __getitem__(self, index) -> list[torch.tensor, torch.tensor, torch.tensor]:
        # locate the ply file
        filename_ply = self.pd_db.iloc[index, 1]
        filename_ply = filename_ply[:-1] # to remove the tick at the end of the filename field in the csv
        mos = self.pd_db.iloc[index, 2]
        features = np.load(join(self.path_ply + "_npz_" + str(self.number_patches), filename_ply[:-3] + "npz"))
        patches_big, patches_small = features['x_b'], features['x_s']
        index_b = np.arange(patches_big.shape[1])
        if self.train:
            index_big = np.random.choice(index_b, self.points_structure, replace=False)
            for patch in patches_small:
                patch = np.random.shuffle(patch)
        else:
            np.random.seed(0)
            index_big = np.random.choice(index_b, self.points_structure, replace=False)
        patches_big = patches_big[:, index_big]
        return torch.tensor(np.array(patches_big)).float(), torch.tensor(np.array(patches_small)).float(), torch.tensor(mos/self.max_mos).float()
    
    def __len__(self) -> int:
        return super().__len__()
   

class SIAT_PCQD(Pointcloud_Dataset): 

    def __init__(self, pd_db: pd.DataFrame, path_ply: str, number_patches: int, points_structure: int, train:bool = False) -> None:
        super().__init__(pd_db, path_ply, number_patches, points_structure, train) 
        self.max_mos = 5

    def __getitem__(self, index) -> list[torch.tensor, torch.tensor, torch.tensor]:
        # locate the ply file
        filename_ply = "{}_r{:02d}.ply".format(self.pd_db.iloc[index, 1], self.pd_db.iloc[index, 2])
        mos = self.pd_db.iloc[index, -1]
        features = np.load(join(self.path_ply + "_npz_" + str(self.number_patches), filename_ply[:-3] + "npz"))
        patches_big, patches_small = features['x_b'], features['x_s']
        index_b = np.arange(patches_big.shape[1])
        if self.train:
            index_big = np.random.choice(index_b, self.points_structure, replace=False)
            for patch in patches_small:
                patch = np.random.shuffle(patch)
        else:
            np.random.seed(0)
            index_big = np.random.choice(index_b, self.points_structure, replace=False)
        patches_big = patches_big[:, index_big]
        return torch.tensor(np.array(patches_big)).float(), torch.tensor(np.array(patches_small)).float(), torch.tensor(mos/self.max_mos).float()
    
    
    def __len__(self) -> int:
        return super().__len__()
    
class SJTU_PCQA(Pointcloud_Dataset): 

    def __init__(self, pd_db: pd.DataFrame, path_ply: str, number_patches: int, points_structure: int, train:bool = False) -> None:
        super().__init__(pd_db, path_ply, number_patches, points_structure, train)
        self.number_distortions = 42
        self.max_mos = 10
        
    def __getitem__(self, index) -> list[torch.tensor, torch.tensor, torch.tensor]:
        # locate the ply file
        name_pc = self.pd_db.iloc[index , 1]
        filename_ply = name_pc + "_{}.ply".format(self.pd_db.iloc[index, 2])
        mos = self.pd_db.iloc[index, 0]
        features = np.load(join(self.path_ply + "_npz_" + str(self.number_patches), filename_ply[:-3] + "npz"))
        patches_big, patches_small = features['x_b'], features['x_s']
        index_b = np.arange(patches_big.shape[1])
        if self.train:
            index_big = np.random.choice(index_b, self.points_structure, replace=False)
            for patch in patches_small:
                patch = np.random.shuffle(patch)
        else:
            np.random.seed(0)
            index_big = np.random.choice(index_b, self.points_structure, replace=False)
        patches_big = patches_big[:, index_big]
        return torch.tensor(np.array(patches_big)).float(), torch.tensor(np.array(patches_small)).float(), torch.tensor(mos/self.max_mos).float()
    
    def __len__(self) -> int:
        return super().__len__()
    
class LS_PCQA(Pointcloud_Dataset):

    def __init__(self, pd_db: pd.DataFrame, path_ply: str, number_patches: int, points_texture: int, train:bool = False) -> None:
        super().__init__(pd_db, path_ply, number_patches, points_texture, train)
        self.max_mos = 100

    def __getitem__(self, index) -> list[torch.tensor, torch.tensor, torch.tensor]:
        # locate the ply file
        filename_ply = self.pd_db.iloc[index, 0]
        mos = self.pd_db.iloc[index, 1]
        features = np.load(join(self.path_ply + "_npz_" + str(self.number_patches), filename_ply[:-3] + "npz"))
        patches_big, patches_small = features['x_b'], features['x_s']
        index_b = np.arange(patches_big.shape[1])
        if self.train:
            index_big = np.random.choice(index_b, self.points_structure, replace=False)
            for patch in patches_small:
                patch = np.random.shuffle(patch)
        else:
            np.random.seed(0)
            index_big = np.random.choice(index_b, self.points_structure, replace=False)
        patches_big = patches_big[:, index_big]
        return torch.tensor(np.array(patches_big)).float(), torch.tensor(np.array(patches_small)).float(), torch.tensor(mos/self.max_mos).float()
    
    def __len__(self) -> int:
        return super().__len__()
    

class WPC2(Pointcloud_Dataset):

    def __init__(self, pd_db: pd.DataFrame, path_ply: str, number_patches: int, points_structure: int, train:bool = False) -> None:
        super().__init__(pd_db, path_ply, number_patches, points_structure, train)
        self.max_mos = 100
        self.min_mos = 0

    def __getitem__(self, index) -> list[torch.tensor, torch.tensor, torch.tensor]:
        # locate the ply file
        filename_ply = self.pd_db.iloc[index, 1]
        mos = self.pd_db.iloc[index, 4]
        features = np.load(join(self.path_ply + "_npz_" + str(self.number_patches), filename_ply[:-3] + "npz"))
        patches_big, patches_small = features['x_b'], features['x_s']
        index_b = np.arange(patches_big.shape[1])
        if self.train:
            index_big = np.random.choice(index_b, self.points_structure, replace=False)
            for patch in patches_small:
                patch = np.random.shuffle(patch)
        else:
            np.random.seed(0)
            index_big = np.random.choice(index_b, self.points_structure, replace=False)
        patches_big = patches_big[:, index_big]
        return torch.tensor(np.array(patches_big)).float(), torch.tensor(np.array(patches_small)).float(), torch.tensor(mos/self.max_mos).float()
    
    def __len__(self) -> int:
        return super().__len__()
    
##### LIGHTNINGDATAMODULES ######
    

class WPCModule(LightningDataModule):

    def __init__(self,  pd_db: pd.DataFrame, path_ply: str, number_patches: int, points_structure: int, batch_size:int) -> None:
        super().__init__()
        self.pd_db = pd_db
        self.path_ply = path_ply
        self.number_patches = number_patches
        self.points_structure = points_structure
        self.batch_size = batch_size
        self.training_pc = ["bag", "biscuits", "cake", "flowerpot", "glasses_case", "honeydew_melon", "house", "litchi", "pen_container",
                             "ping-pong_bat", "puer_tea", "pumpkin", "ship", "statue", "stone", "tool_box"]
        self.test_pc = ["banana", "pumpkin", "mushroom", "pineapple"] 
        self.max_mos = 100

        # train-test splitting according to original paper (random 10 % of the training data is used as validation set)
        self.pd_db_train, self.pd_db_val = train_test_split(self.pd_db[self.pd_db['Content'].isin(self.training_pc)], test_size=0.1, random_state=42)
        self.pd_db_test = self.pd_db[self.pd_db['Content'].isin(self.test_pc)]

    def prepare_data(self):
        pass
    
    def train_dataloader(self):
        return DataLoader(WPC(self.pd_db_train, self.path_ply, self.number_patches, self.points_structure, train = True), batch_size = self.batch_size, shuffle = True, drop_last = False)
    
    def val_dataloader(self):
        return DataLoader(WPC(self.pd_db_val, self.path_ply, self.number_patches, self.points_structure), batch_size = self.batch_size, drop_last = False)
    
    def test_dataloader(self):
        return DataLoader(WPC(self.pd_db_test, self.path_ply, self.number_patches, self.points_structure), batch_size = self.batch_size, drop_last = False)
    
class SIAT_PCQDModule(LightningDataModule):

    def __init__(self, pd_db: pd.DataFrame, path_ply: str, number_patches: int, points_structure: int, batch_size:int, fold:int) -> None:
        super().__init__()
        self.pd_db = pd_db
        self.path_ply = path_ply
        self.number_patches = number_patches
        self.points_structure = points_structure
        self.batch_size = batch_size
        self.fold = fold
        self.name_pcs = ["Longdress", "Andrew", "UlliWegner", "RedAndBlack", "Ricardo", "The20sMaria",
                         "Phil", "Loot", "Sarah", "Soldier", "Grass", "Biplane", "Banana", "AngelSeated", 
                         "RomanOillamp", "Facade", "Bush", "House", "Nike", "ULBUnicorn"]
        self.name_pc_test = [self.name_pcs[self.fold]] 
        self.name_pc_training = self.name_pcs
        self.name_pc_training.remove(self.name_pc_test[0])
        print("Training point clouds: {}".format(self.name_pc_training))
        print("Test point cloud: {}".format(self.name_pc_test))
        # train-test splitting according to original paper (random 10 % of the training data is used as validation set)
        self.pd_db_train, self.pd_db_val = train_test_split(self.pd_db[self.pd_db['Sequences'].isin(self.name_pc_training)], test_size=0.05, random_state=3)
        self.pd_db_test = self.pd_db[self.pd_db['Sequences'].isin(self.name_pc_test)]
        self.max_mos = 1

    def prepare_data(self) -> None:
        pass
    
    def train_dataloader(self):
        return DataLoader(SIAT_PCQD(pd_db = self.pd_db_train, path_ply = self.path_ply, number_patches = self.number_patches, points_structure = self.points_structure, train = True), batch_size = self.batch_size, shuffle = True)
    
    def val_dataloader(self):
        return DataLoader(SIAT_PCQD(pd_db = self.pd_db_val, path_ply = self.path_ply, number_patches = self.number_patches, points_structure = self.points_structure, train = False), batch_size = self.batch_size, shuffle = False)
    
    def test_dataloader(self):
        return DataLoader(SIAT_PCQD(pd_db = self.pd_db_test, path_ply = self.path_ply, number_patches = self.number_patches, points_structure = self.points_structure, train = False), batch_size = self.batch_size, shuffle = False)
    
class SJTU_PCQAModule(LightningDataModule):

    def __init__(self, pd_db: pd.DataFrame, path_ply: str, number_patches: int, points_structure: int, batch_size:int, fold:int) -> None:
        super().__init__()
        self.pd_db = pd_db.transpose().stack().reset_index().drop(columns=["level_0", "level_1"])
        self.path_ply = path_ply
        self.number_patches = number_patches
        self.points_structure = points_structure
        self.batch_size = batch_size
        self.fold = fold
        self.name_pcs = ['redandblack','ULB unicorn','loot','soldier','Romanoillamp','longdress','statue','shiva','hhi']
        self.max_mos = 10
        
        # create a column for remembering the name of the pristine pointcloud
        point_cloud_names = []
        for pc_name in self.name_pcs:
            point_cloud_names.extend(([pc_name] * 42))

        self.pd_db['Content'] = point_cloud_names
        self.pd_db['Index_distortion'] = list(range(42)) * 9
        self.name_pc_test = [self.name_pcs[self.fold]] 
        self.name_pc_training = self.name_pcs
        self.name_pc_training.remove(self.name_pc_test[0])
        print("Training point clouds: {}".format(self.name_pc_training))
        print("Test point cloud: {}".format(self.name_pc_test))

        # train-test splitting according to original paper (random 10 % of the training data is used as validation set)
        self.pd_db_train, self.pd_db_val = train_test_split(self.pd_db[self.pd_db['Content'].isin(self.name_pc_training)], test_size=0.1, random_state=52)
        self.pd_db_test = self.pd_db[self.pd_db['Content'].isin(self.name_pc_test)]

    def prepare_data(self) -> None:
        pass
    
    def train_dataloader(self):
        return DataLoader(SJTU_PCQA(pd_db = self.pd_db_train, path_ply = self.path_ply, number_patches = self.number_patches, points_structure = self.points_structure, train = True), batch_size = self.batch_size, shuffle = True)
    
    def val_dataloader(self):
        return DataLoader(SJTU_PCQA(pd_db = self.pd_db_val, path_ply = self.path_ply, number_patches = self.number_patches, points_structure = self.points_structure, train = False), batch_size = self.batch_size, shuffle = False)
    
    def test_dataloader(self):
        return DataLoader(SJTU_PCQA(pd_db = self.pd_db_test, path_ply = self.path_ply, number_patches = self.number_patches, points_structure = self.points_structure, train = False), batch_size = self.batch_size, shuffle = False)
    

###################################### TESTS ###############
if __name__ == "__main__": # test functions
    # WPC
        info_file = pd.read_excel("data/WPC/mos.xls")
        path_ply = "data/WPC/distorted"
        dataset = WPC(info_file, path_ply, 16, 1024)
        p_t, p_s, mos = dataset[np.random.randint(low = 0, high = len(dataset))]
        print("*******WPC*******")
        print("MOS: {}".format(mos))
        print("Texture patches: {}".format(p_t.shape))
        print("Structure patches: {}".format(p_s.shape))
        print(len(dataset))
        
        # WPC2.0
        info_file = pd.read_excel("data/WPC2.0/mos.xlsx")
        path_ply = "data/WPC2.0/distorted"
        dataset = WPC2(info_file, path_ply, 16, 14900, 8192)
        p_t, p_s, mos = dataset[np.random.randint(low = 0, high = len(dataset))]
        print("*******WPC2.0*******")
        print("MOS: {}".format(mos))
        print("Texture patches: {}".format(p_t.shape))
        print("Structure patches: {}".format(p_s.shape))
        print(len(dataset))

        # SIAT-PCQD
        info_file = pd.read_excel("data/SIAT-PCQD/DMOS.xlsx").ffill(axis = 0)
        path_ply = "data/SIAT-PCQD/distorted"
        dataset = SIAT_PCQD(info_file, path_ply, 16, 14900, 8192)
        p_t, p_s, mos = dataset[np.random.randint(low = 0, high = len(dataset))]
        print("*******SIAT-PCQD*******")
        print("MOS: {}".format(mos))
        print("Texture patches: {}".format(p_t.shape))
        print("Structure patches: {}".format(p_s.shape))
        print(len(dataset))

        # SJTU-PCQA
        info_file = pd.read_excel("data/SJTU-PCQA/MOS.xlsx", header = None)
        path_ply = "data/SJTU-PCQA/distorted"
        dataset = SJTU_PCQA(info_file, path_ply, 16, 14900, 8192)
        p_t, p_s, mos = dataset[np.random.randint(low = 0, high = len(dataset))]
        print("*******SJTU-PCQA*******")
        print("MOS: {}".format(mos))
        print("Texture patches: {}".format(p_t.shape))
        print("Structure patches: {}".format(p_s.shape))
        print(len(dataset))

        # LS-PCQA
        info_file = pd.read_excel("data/LS-PCQA/MOS.xlsx")
        path_ply = "data/LS-PCQA/distorted"
        dataset = LS_PCQA(info_file, path_ply, 16, 14900, 8192)
        p_t, p_s, mos = dataset[np.random.randint(low = 0, high = len(dataset))]
        print("*******LS-PCQA*******")
        print("MOS: {}".format(mos))
        print("Texture patches: {}".format(p_t.shape))
        print("Structure patches: {}".format(p_s.shape))
        print(len(dataset))
        

        ###### TEST LIGHTINING DATAMODULE
        # SIAT-PCQD
        
        print("******************* SIAT-PCQD test *****************")
        for k in range(20):
            print("Fold {}".format(k))
            info_file = pd.read_excel("data/SIAT-PCQD/DMOS.xlsx").ffill(axis = 0)
            path_ply = "data/SIAT-PCQD/distorted"
            datamodule = SIAT_PCQDModule(info_file, path_ply, 16, 1024, 4, k) # 16 batch size fold 0
            training_dataloder = datamodule.train_dataloader()
            batch = next(iter(training_dataloder))
            x_b, x_s, mos = batch
            print("Dataloader size: {}".format(len(training_dataloder)))
            print("x_b shape : {}".format(x_b.shape))
            print("x_s shape : {}".format(x_s.shape))
            print("mos shape : {}".format(mos.shape))

        # SJTU-PCQA
        print("****************** SJTU-PCQA test ***************")
        for k in range(9):
            print("Fold {}".format(k))
            info_file = pd.read_excel("data/SJTU-PCQA/MOS.xlsx", header = None)
            path_ply = "data/SJTU-PCQA/distorted"
            datamodule = SJTU_PCQAModule(info_file, path_ply, 16, 1024, 4, k) # 16 batch size fold 0
            training_dataloder = datamodule.train_dataloader()
            batch = next(iter(training_dataloder))
            x_b, x_s, mos = batch
            print("Dataloader size: {}".format(len(training_dataloder)))
            print("x_b shape : {}".format(x_b.shape))
            print("x_s shape : {}".format(x_s.shape))
            print("mos shape : {}".format(mos.shape))
        
        # WPC
        print("******** WPC test *********")
        info_file = pd.read_excel("data/WPC/mos.xls")
        path_ply = "data/WPC/distorted"
        datamodule = WPCModule(info_file, path_ply, 16, 1024, 4) # 16 batch size
        training_dataloder = datamodule.train_dataloader()
        batch = next(iter(training_dataloder))
        x_b, x_s, mos = batch
        print("Dataloader size: {}".format(len(training_dataloder)))
        print("x_b shape : {}".format(x_b.shape))
        print("x_s shape : {}".format(x_s.shape))
        print("mos shape : {}".format(mos.shape))
        

    


