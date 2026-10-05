
import pandas as pd                   
from pytorch_lightning import LightningDataModule
from torch.utils.data import  DataLoader
from sklearn.model_selection import train_test_split
from data import WPC, SJTU_PCQA

class WPCModuleCross(LightningDataModule):

    def __init__(self,  pd_db: pd.DataFrame, path_ply: str, number_patches: int, points_structure: int, batch_size:int):
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
        self.pd_db_train, self.pd_db_val = train_test_split(self.pd_db, test_size=0.1, random_state=42)
        self.pd_db_test = self.pd_db

    def prepare_data(self):
        pass
    
    def train_dataloader(self):
        return DataLoader(WPC(self.pd_db_train, self.path_ply, self.number_patches, self.points_structure, train = True), batch_size = self.batch_size, shuffle = True, drop_last = False)
    
    def val_dataloader(self):
        return DataLoader(WPC(self.pd_db_val, self.path_ply, self.number_patches, self.points_structure), batch_size = self.batch_size, drop_last = False)
    
    def test_dataloader(self):
        return DataLoader(WPC(self.pd_db_test, self.path_ply, self.number_patches, self.points_structure), batch_size = self.batch_size, drop_last = False)
    
class SJTU_PCQAModuleCross(LightningDataModule):

    def __init__(self, pd_db: pd.DataFrame, path_ply: str, number_patches: int, points_structure: int, batch_size:int):
        super().__init__()
        self.pd_db = pd_db.transpose().stack().reset_index().drop(columns=["level_0", "level_1"])
        self.path_ply = path_ply
        self.number_patches = number_patches
        self.points_structure = points_structure
        self.batch_size = batch_size
        self.name_pcs = ['redandblack','ULB unicorn','loot','soldier','Romanoillamp','longdress','statue','shiva','hhi']
        self.max_mos = 10
        
        # create a column for remembering the name of the pristine pointcloud
        point_cloud_names = []
        for pc_name in self.name_pcs:
            point_cloud_names.extend(([pc_name] * 42))

        self.pd_db['Content'] = point_cloud_names
        self.pd_db['Index_distortion'] = list(range(42)) * 9

        # train-test splitting according to original paper (random 10 % of the training data is used as validation set)
        self.pd_db_train, self.pd_db_val = train_test_split(self.pd_db, test_size=0.1, random_state=52)
        self.pd_db_test = self.pd_db

    def prepare_data(self):
        pass
    
    def train_dataloader(self):
        return DataLoader(SJTU_PCQA(pd_db = self.pd_db_train, path_ply = self.path_ply, number_patches = self.number_patches, points_structure = self.points_structure, train = True), batch_size = self.batch_size, shuffle = True)
    
    def val_dataloader(self):
        return DataLoader(SJTU_PCQA(pd_db = self.pd_db_val, path_ply = self.path_ply, number_patches = self.number_patches, points_structure = self.points_structure, train = False), batch_size = self.batch_size, shuffle = False)
    
    def test_dataloader(self):
        return DataLoader(SJTU_PCQA(pd_db = self.pd_db_test, path_ply = self.path_ply, number_patches = self.number_patches, points_structure = self.points_structure, train = False), batch_size = self.batch_size, shuffle = False)