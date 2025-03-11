import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from util import sample_and_group

###### TORCH MODULES #########
class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, groups=4, bias=False)  # Grouped Convolution
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, groups=4, bias=False)  # Grouped Convolution
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, _ = x.size()
        x1 = F.elu(self.bn1(self.conv1(x)))
        x2 = F.elu(self.bn2(self.conv2(x1)))
        x3 = F.adaptive_max_pool1d(x2, 1)
        x4 = x3.view(batch_size, -1)
        x_res = x4.reshape(b, n, -1).permute(0, 2, 1)
        return x_res

class S_TFE(nn.Module):
    def __init__(self, point_num):
        super(S_TFE, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 1024, kernel_size=1, stride=int(point_num/256), bias=False)
        self.bn2 = nn.BatchNorm1d(1024)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.conv_fuse1 = nn.Sequential(nn.Conv1d(1280, 512, kernel_size=1, bias=False),
                            nn.BatchNorm1d(512),
                            nn.ELU())

    def forward(self, x):
        xyz = x[:, 0:3, :].permute(0, 2, 1)
        x = F.elu(self.bn1(self.conv1(x)))
        x_skip = F.elu(self.bn2(self.conv2(x)))
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, neighbor=32, xyz=xyz, feature=x)
        feature_0 = self.gather_local_0(new_feature)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, neighbor=32, xyz=new_xyz, feature=feature_0)
        feature_1 = self.gather_local_1(new_feature)
        feature_1 = torch.cat((feature_1, x_skip), dim=1)
        res = self.conv_fuse1(feature_1)
        return res
    
class GlobalVariancePooling(nn.Module):
    def __init__(self):
        super(GlobalVariancePooling, self).__init__()

    def forward(self, x):
        # x shape: (batch_size, patch, features)
        
        # Compute the mean over the spatial dimensions
        mean = x.mean(dim=[2], keepdim=True)
        
        # Variance = E[(X - E[X])^2]
        variance = ((x - mean) ** 2).mean(dim=[2])
        
        return variance

class PST_PCQA(nn.Module):
    def __init__(self, points_texture:int, points_structure:int, dropout:float, patches:int):
        super(PST_PCQA, self).__init__()
        self.points_texture = points_texture
        self.points_structure = points_structure
        self.dropout = dropout

        self.Net_b = S_TFE(point_num = self.points_structure)
        self.Net_s = S_TFE(point_num = self.points_texture)
        self.CBR = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, groups=4, bias=False),  # Grouped Convolution
                        nn.BatchNorm1d(512),
                        nn.ELU())
        self.linear1 = nn.Linear(512, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(patches)    # <-------- number of patches here

        self.LBED = nn.Sequential(
            nn.Linear(in_features = 256, out_features = 1, bias = False),
            nn.BatchNorm1d(patches), # <-------- number of patches here
            nn.ELU(),
            nn.Dropout(self.dropout)
        )

        self.linear_mos_per_patch = nn.Linear(in_features = 256, out_features = 1)
        self.GVP = GlobalVariancePooling()


    def forward(self, x_b, x_s):
        batch_size, patches, _, features = x_b.size()
        x_b = x_b.view(-1, x_b.shape[2], features)
        x_s = x_s.view(-1, x_s.shape[2], features)
        x_b = x_b.permute(0, 2, 1)
        x_s = x_s.permute(0, 2, 1)
        x_b = self.Net_b(x_b)
        x_s = self.Net_s(x_s)
        x = torch.cat((x_b, x_s), dim=1)
        # x = x_s # only for ablation study
        x = self.CBR(x)
        # x = F.adaptive_avg_pool1d(x, 1)
        # x = F.adaptive_max_pool1d(x, 1)
        # x = torch.cat((F.adaptive_max_pool1d(x, 1), self.GVP(x).unsqueeze(-1)), dim = -1)
        x = self.GVP(x).unsqueeze(-1)
        # x = torch.cat((F.adaptive_avg_pool1d(x, 1), self.GVP(x).unsqueeze(-1)), dim = -1)
        x = x.view(batch_size, patches, -1) # features per patch [batch_size, patch, features = 1024]
        x = F.elu(self.bn1(self.linear1(x))) # [batch_size, patch_ features = 256]


        # CREATE HERE TWO BRANCHES, MOS AND WEIGHTS PREDICTION
        logits_patch = self.LBED(x).squeeze(-1)
        mos_per_patch = self.linear_mos_per_patch(x).squeeze(-1)
        mos = torch.mean(logits_patch * mos_per_patch, -1)
        return mos, mos_per_patch
    

###### LIGHTNING MODULE ######
    
class PST_PCQAModule(LightningModule):

    def __init__(self, points_texture:int, points_structure:int, dropout:float, patches:int, lr:float):
        super().__init__()
        self.points_texture = points_texture
        self.points_structure = points_structure
        self.dropout = dropout
        self.lr = lr
        self.model = PST_PCQA(self.points_texture, self.points_structure, self.dropout, patches)

        # loss function
        self.loss_function = torch.nn.MSELoss()

        # save data for later assessment of metrics
        self.true_mos = []
        self.predicted_mos = []

    def forward(self, x_b, x_s):
        return self.model(x_b, x_s)
    
    def training_step(self, x, ind_x):
        x_b, x_s, mos = x
        pred, pred_per_patch= self.forward(x_b, x_s)
        loss = self.loss_function(pred, mos)
        self.log("train/loss_mse", loss, prog_bar = True, on_step = False, on_epoch = True)
        loss_per_patch = self.loss_function(pred_per_patch, mos.unsqueeze(-1) * torch.ones(x_b.size()[0], x_b.size()[1], device = self.device))
        self.log("train/loss_per_patch", loss_per_patch, on_step = False, on_epoch = True )
        final_loss = loss + loss_per_patch # now it is 1 and 1 but we can balance them
        return final_loss

    def validation_step(self, x, ind_x):
        x_b, x_s, mos = x
        pred, pred_per_patch= self.forward(x_b, x_s)
        loss = self.loss_function(pred, mos)
        self.log("val/loss_mse", loss, prog_bar = True, on_step = False, on_epoch = True)
        loss_per_patch = self.loss_function(pred_per_patch, mos.unsqueeze(-1) * torch.ones(x_b.size()[0], x_b.size()[1], device = self.device))
        self.log("val/loss_per_patch", loss_per_patch, on_step = False, on_epoch = True )
        final_loss = loss + loss_per_patch
        return final_loss
    
    def test_step(self, x, ind_x):
        x_b, x_s, mos = x
        pred, _ = self.forward(x_b, x_s)
        loss = self.loss_function(pred, mos)
        self.log("test/loss_mse", loss, prog_bar = True, on_step = False, on_epoch = True)
        self.true_mos.append(mos)
        self.predicted_mos.append(pred)
        return loss
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = 1e-5)
        # return opt
        return {
           "optimizer": opt,
           "lr_scheduler": {
               "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=400, eta_min=0.1*float(self.lr))
                           },
              }   
    

if __name__ == "__main__":
    # test the model 
    model = PST_PCQAModule(8192, 1024, 0.5, 16, 0.0001)
    dummy_b, dummy_s = torch.rand(2, 16, 1024, 6), torch.rand(2, 16, 8192, 6)
    mos, mos_per_patch = model(dummy_b, dummy_s)
    print(mos.shape)
    print(mos_per_patch.shape)
         
