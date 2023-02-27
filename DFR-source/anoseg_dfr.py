import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
#from extractors.feature import Extractor
from feature import Extractor
from torch.utils.data import DataLoader
import torch.optim as optim
#from data.MVTec import NormalDataset, TrainTestDataset,
from skimage import img_as_ubyte

import time
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave
from skimage import measure
from skimage.transform import resize
import pandas as pd

from feat_cae import FeatCAE
from ssim import ssim

import joblib
from sklearn.decomposition import PCA

from utils import *
import torch
import torch.nn as nn
import numpy as np
np.set_printoptions(threshold=np.inf)
#二分类
class FocalLoss(nn.Module):
 
    def __init__(self, gamma=2,alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha=alpha
    def forward(self, input, target):
        # input:size is M*2. M　is the batch　number
        # target:size is M.
        pt=torch.softmax(input,dim=1)
        p=pt[:,1]
        loss = -self.alpha*(1-p)**self.gamma*(target*torch.log(p))-\
               (1-self.alpha)*p**self.gamma*((1-target)*torch.log(1-p))
        return loss.mean()
class AnoSegDFR():
    """
    Anomaly segmentation model: DFR.
    """
    def __init__(self, cfg):
        super(AnoSegDFR, self).__init__()
        self.cfg = cfg
        self.path = cfg.save_path    # model and results saving path

        self.n_layers = len(cfg.cnn_layers)
        self.n_dim = cfg.latent_dim

        self.log_step = 10
        self.data_name = cfg.data_name

        self.img_size = cfg.img_size
        self.threshold = cfg.thred
        self.device = torch.device(cfg.device)

        # feature extractor
        self.extractor = Extractor(backbone=cfg.backbone,
                 cnn_layers=cfg.cnn_layers,
                 upsample=cfg.upsample,
                 is_agg=cfg.is_agg,
                 kernel_size=cfg.kernel_size,
                 stride=cfg.stride,
                 dilation=cfg.dilation,
                 featmap_size=cfg.featmap_size,
                 device=cfg.device).to(self.device)

        # datasest
        self.train_data_path = cfg.train_data_path
        self.test_data_path = cfg.test_data_path
        self.train_data = self.build_dataset(is_train=True)
        self.test_data = self.build_dataset(is_train=False)

        # dataloader
        self.train_data_loader = DataLoader(self.train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
        self.test_data_loader = DataLoader(self.test_data, batch_size=1, shuffle=False, num_workers=1)
        self.eval_data_loader = DataLoader(self.train_data, batch_size=10, shuffle=False, num_workers=2)


        # autoencoder classifier
        self.autoencoder, self.model_name = self.build_classifier()
        if cfg.model_name != "":
            self.model_name = cfg.model_name
        print("model name:", self.model_name)

        # optimizer
        self.lr = cfg.lr

        # saving paths
        self.subpath = self.data_name + "/" + self.model_name
        self.subpath = self.data_name + "/" + self.model_name
        self.model_path = os.path.join(self.path, self.subpath + "/model")
        # self.model_path = '/root/project/mypy/DFR/DFR-source/checkpoints/carpet-semi-5-5/dalunwen_reconstruction_only_good/model'

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.eval_path = os.path.join(self.path, self.subpath + "/eval")
        if not os.path.exists(self.eval_path):
            os.makedirs(self.eval_path)
        self.BCEwithLog = torch.nn.BCEWithLogitsLoss()
        # self.load_model()
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.lr, weight_decay=0)
        self.focal = FocalLoss()
    def build_classifier(self):
        # self.load_dim(self.model_path)
        if self.n_dim is None:
            print("Estimating one class classifier AE parameter...")
            feats = torch.Tensor()
            for i, (normal_img,_,_) in enumerate(self.eval_data_loader):
                i += 1
                if i > 1:
                    break
                normal_img = normal_img.to(self.device)
                feat = self.extractor.feat_vec(normal_img)
                feats = torch.cat([feats, feat.cpu()], dim=0)
            # to numpy
            feats = feats.detach().numpy()
            # estimate parameters for mlp
            pca = PCA(n_components=0.90)    # 0.9 here try 0.8
            pca.fit(feats)
            n_dim, in_feat = pca.components_.shape
            print("AE Parameter (in_feat, n_dim): ({}, {})".format(in_feat, n_dim))
            self.n_dim = n_dim
        else:
            for i, (normal_img,_,_) in enumerate(self.eval_data_loader):
                i += 1
                if i > 1:
                    break
                normal_img = normal_img.to(self.device)
                feat = self.extractor.feat_vec(normal_img)
            in_feat = feat.shape[1]

        print("BN?:", self.cfg.is_bn)
        print("AE Parameter (in_feat, n_dim): ({}, {})".format(in_feat, self.n_dim))

        autoencoder = FeatCAE(in_channels=in_feat, latent_dim=self.n_dim, is_bn=self.cfg.is_bn).to(self.device)
        model_name = "AnoSegDFR({})_{}_l{}_d{}_s{}_k{}_{}".format('BN' if self.cfg.is_bn else 'noBN',
                                                                self.cfg.backbone, self.n_layers,
                                                                self.n_dim, self.cfg.stride[0],
                                                                self.cfg.kernel_size[0], self.cfg.upsample)

        return autoencoder, model_name

    def build_dataset(self, is_train):
        from MVTec import NormalDataset, TestDataset,Train_abnormal_Dataset
        normal_data_path = self.train_data_path
        abnormal_data_path = self.test_data_path
        if is_train:
            # dataset = NormalDataset(normal_data_path, normalize=True)
            dataset = Train_abnormal_Dataset(normal_data_path, normalize=True)
        else:
            dataset = TestDataset(path=abnormal_data_path)
        return dataset

    def train(self):
        # if self.load_model():
        #     print(".......Model Loaded.......")
        #     return

        start_time = time.time()
        self.cfg.epochs=500
        # train
        iters_per_epoch = len(self.train_data_loader)  # total iterations every epoch
        epochs = self.cfg.epochs  # total epochs
        for epoch in range(1, epochs+1):
            self.extractor.train()
            self.autoencoder.train()
            losses = []
            for i, (normal_img,abnormal_img,mask) in enumerate(self.train_data_loader):
                normal_img = normal_img.to(self.device)
                abnormal_img = abnormal_img.to(self.device)
                mask = mask.to(self.device)
                # imsave('./{}.png'.format(epoch), ((normal_img[0].detach().cpu().float().permute(1,2,0).numpy() + 1.0) / 2.0 * 255.0).astype('uint8'))

                # forward and backward
                total_loss = self.optimize_step(normal_img,abnormal_img,mask)                
                # total_loss = self.optimize_dual_step(normal_img,abnormal_img,mask)
                # total_loss = self.optimize_step_semi_mode(normal_img,abnormal_img,mask)
                # total_loss = self.optimize_step_semi_mode_direct_to_target(normal_img,abnormal_img,mask)
                # statistics and logging
                loss = {}
                loss['total_loss'] = total_loss.data.item()
                
                # tracking loss
                losses.append(loss['total_loss'])

            if epoch % 2 == 0:
                #                 self.save_model()

                print('Epoch {}/{}'.format(epoch, epochs))
                print('-' * 10)
                elapsed = time.time() - start_time
                total_time = ((epochs * iters_per_epoch) - (epoch * iters_per_epoch + i)) * elapsed / (
                        epoch * iters_per_epoch + i + 1)
                epoch_time = (iters_per_epoch - i) * elapsed / (epoch * iters_per_epoch + i + 1)

                epoch_time = str(datetime.timedelta(seconds=epoch_time))
                total_time = str(datetime.timedelta(seconds=total_time))
                elapsed = str(datetime.timedelta(seconds=elapsed))

                log = "Elapsed {}/{} -- {} , Epoch [{}/{}], Iter [{}/{}]".format(
                    elapsed, epoch_time, total_time, epoch, epochs, i + 1, iters_per_epoch)

                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

            if epoch % 2 == 0:
                # save model
                print("save model")
                self.save_model()
                self.segment_evaluation()
                self.segment_evaluation_with_otsu_li()
                self.metrics_detecion()
                self.validation(epoch)
#             print("Cost total time {}s".format(time.time() - start_time))
#             print("Done.")
            self.tracking_loss(epoch, np.mean(np.array(losses)))
        # save model
        self.save_model()
        print("Cost total time {}s".format(time.time() - start_time))
        print("Done.")
        # self.metrics_evaluation()


    def tracking_loss(self, epoch, loss):
        out_file = os.path.join(self.eval_path, '{}_epoch_loss.csv'.format(self.model_name))
        if not os.path.exists(out_file):
            with open(out_file, mode='w') as f:
                f.write("Epoch" + ",loss" + "\n")
        with open(out_file, mode='a+') as f:
            f.write(str(epoch) + "," + str(loss) + "\n")
            
    def optimize_step(self, input_data,abnormal_img,mask):
        self.extractor.train()
        self.autoencoder.train()

        self.optimizer.zero_grad()

        # forward
        input_data = self.extractor(input_data)
        # abnormal_img = self.extractor(abnormal_img)

        # print(input_data.size())
        dec = self.autoencoder(input_data)
        # dec_defect = self.autoencoder(abnormal_img)
        # loss
        # print("dec feature: ",dec_defect.shape)
        # print("abnormal_img feature: ",abnormal_img.shape)
        # print("mask shape: ",mask.shape)

        total_loss = self.autoencoder.loss_function(dec, input_data.detach().data)
        # total_loss = self.autoencoder.loss_function(dec, input_data.detach().data)
        # print("total_loss.1shape: ", total_loss.shape)

        # ssim_loss= ssim(dec,input_data.detach().data)
        # total_loss_defect = self.autoencoder.loss_function(dec_defect, abnormal_img.detach().data)
        # total_loss_defect = self.autoencoder.loss_function(dec_defect, input_data.detach().data)
        # scores = self.autoencoder.compute_energy(dec_defect, abnormal_img.detach().data)
        # print("scores.1shape: ", scores.shape)
        
       
        # mask = nn.functional.interpolate(mask, size=dec_defect.shape[2], mode="bilinear", align_corners=True)
        # mask = torch.squeeze(mask,0)
        # mask[mask > 0.5] = 1
        # mask[mask <= 0.5] = 0
        # print("mask.1shape: ", mask.shape)

        # scores = torch.sigmoid(scores)
        # total_loss_defect = self.autoencoder.loss_function(scores, mask.detach().data)
        # total_loss_defect = self.focal(scores,mask.detach().data)
        # print("total_loss_defect.1shape: ", total_loss_defect.shape)

        # scores = scores.reshape((scores.shape[0], 1, self.extractor.out_size[0], self.extractor.out_size[1]))  # test batch size is 1.
        # print("scores.2shape: ", scores)
        # np.savetxt("./train_mask.txt", mask.data.cpu().numpy().squeeze(), fmt="%.2f")

        # total_loss_defect = self.BCEwithLog(scores,mask.squeeze())
        # total_loss_defect = torch.mean((scores-mask)**2)
        # print("total_loss feature: ",total_loss.shape)
        # print("mask feature: ",mask)
        # print("mask feature shape: ",mask.shape)
        # total_loss+=total_loss_defect
        # total_loss=total_loss_defect

        # total_loss+=ssim_loss
        # self.reset_grad()
        total_loss.backward()
        # total_loss_defect.backward()

        self.optimizer.step()
        # return total_loss
        return total_loss
    def optimize_supervised_step(self, input_data,abnormal_img,mask):
        self.extractor.train()
        self.autoencoder.train()

        self.optimizer.zero_grad()

        # forward
        # input_data = self.extractor(input_data)
        abnormal_img = self.extractor(abnormal_img)

        # print(input_data.size())
        # dec = self.autoencoder(input_data)
        dec_defect = self.autoencoder(abnormal_img)
        # loss
        # print("dec feature: ",dec_defect.shape)
        # print("abnormal_img feature: ",abnormal_img.shape)
        # print("mask shape: ",mask.shape)

        # total_loss = self.autoencoder.loss_function(dec, input_data.detach().data)
        # total_loss = self.autoencoder.loss_function(dec, input_data.detach().data)
        # print("total_loss.1shape: ", total_loss.shape)

        # ssim_loss= ssim(dec,input_data.detach().data)
        # total_loss_defect = self.autoencoder.loss_function(dec_defect, abnormal_img.detach().data)
        # total_loss_defect = self.autoencoder.loss_function(dec_defect, input_data.detach().data)
        scores = self.autoencoder.compute_energy(dec_defect, abnormal_img.detach().data)
        # print("scores.1shape: ", scores.shape)
        
       
        mask = nn.functional.interpolate(mask, size=dec_defect.shape[2], mode="bilinear", align_corners=True)
        # mask = torch.squeeze(mask,0)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        # print("mask.1shape: ", mask.shape)

        # scores = torch.sigmoid(scores)
        total_loss_defect = self.autoencoder.loss_function(scores, mask.detach().data)
        # total_loss_defect = self.focal(scores,mask.detach().data)
        # print("total_loss_defect.1shape: ", total_loss_defect.shape)

        # scores = scores.reshape((scores.shape[0], 1, self.extractor.out_size[0], self.extractor.out_size[1]))  # test batch size is 1.
        # print("scores.2shape: ", scores)
        # np.savetxt("./train_mask.txt", mask.data.cpu().numpy().squeeze(), fmt="%.2f")

        # total_loss_defect = self.BCEwithLog(scores,mask.squeeze())
        # total_loss_defect = torch.mean((scores-mask)**2)
        # print("total_loss feature: ",total_loss.shape)
        # print("mask feature: ",mask)
        # print("mask feature shape: ",mask.shape)
        # total_loss+=total_loss_defect
        total_loss=total_loss_defect

        # total_loss+=ssim_loss
        # self.reset_grad()
        total_loss.backward()
        # total_loss_defect.backward()

        self.optimizer.step()
        # return total_loss
        return total_loss
    def optimize_dual_step(self, input_data,abnormal_img,mask):
        self.extractor.train()
        self.autoencoder.train()

        self.optimizer.zero_grad()

        # forward
        input_data = self.extractor(input_data)
        abnormal_img = self.extractor(abnormal_img)

        # print(input_data.size())
        dec = self.autoencoder(input_data)
        dec_defect = self.autoencoder(abnormal_img)
        # loss
        # print("dec feature: ",dec_defect.shape)
        # print("abnormal_img feature: ",abnormal_img.shape)
        # print("mask shape: ",mask.shape)

        # total_loss = self.autoencoder.loss_function(dec, input_data.detach().data)
        total_loss = self.autoencoder.loss_function(dec, input_data.detach().data)
        # print("total_loss.1shape: ", total_loss.shape)

        # ssim_loss= ssim(dec,input_data.detach().data)
        # total_loss_defect = self.autoencoder.loss_function(dec_defect, abnormal_img.detach().data)
        # total_loss_defect = self.autoencoder.loss_function(dec_defect, input_data.detach().data)
        scores = self.autoencoder.compute_energy(dec_defect, abnormal_img.detach().data)
        # print("scores.1shape: ", scores.shape)
        
       
        mask = nn.functional.interpolate(mask, size=dec_defect.shape[2], mode="bilinear", align_corners=True)
        # mask = torch.squeeze(mask,0)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        # print("mask.1shape: ", mask.shape)

        scores = torch.sigmoid(scores)
        total_loss_defect = self.autoencoder.loss_function(scores, mask.detach().data)
        # total_loss_defect = self.focal(scores,mask.detach().data)
        # print("total_loss_defect.1shape: ", total_loss_defect.shape)

        # scores = scores.reshape((scores.shape[0], 1, self.extractor.out_size[0], self.extractor.out_size[1]))  # test batch size is 1.
        # print("scores.2shape: ", scores)
        # np.savetxt("./train_mask.txt", mask.data.cpu().numpy().squeeze(), fmt="%.2f")

        # total_loss_defect = self.BCEwithLog(scores,mask.squeeze())
        # total_loss_defect = torch.mean((scores-mask)**2)
        # print("total_loss feature: ",total_loss.shape)
        # print("mask feature: ",mask)
        # print("mask feature shape: ",mask.shape)
        total_loss+=total_loss_defect
        # total_loss=total_loss_defect

        # total_loss+=ssim_loss
        # self.reset_grad()
        total_loss.backward()
        # total_loss_defect.backward()

        self.optimizer.step()
        # return total_loss
        return total_loss
    
    
    def optimize_step_semi_mode(self, input_data,abnormal_img,mask):
        self.extractor.train()
        self.autoencoder.train()

        self.optimizer.zero_grad()

        # forward
        input_data = self.extractor(input_data)
        # abnormal_img = self.extractor(abnormal_img)

        # print(input_data.size())
        dec = self.autoencoder(input_data)
        # dec_defect = self.autoencoder(abnormal_img)
        # loss
        # print("dec feature: ",dec_defect.shape)
        # print("abnormal_img feature: ",abnormal_img.shape)
        # print("mask shape: ",mask.shape)

        # total_loss = self.autoencoder.loss_function(dec, input_data.detach().data)
        # total_loss = self.autoencoder.loss_function(dec, input_data.detach().data)
        # print("total_loss.1shape: ", total_loss.shape)

        # ssim_loss= ssim(dec,input_data.detach().data)
        # total_loss_defect = self.autoencoder.loss_function(dec_defect, abnormal_img.detach().data)
        # total_loss_defect = self.autoencoder.loss_function(dec_defect, input_data.detach().data)
        # print("input_data max: ",torch.max(input_data).cpu().detach().numpy())
        # print("input_data min: ",torch.min(input_data).cpu().detach().numpy())
        # print("dec max: ",torch.max(dec).cpu().detach().numpy())
        # print("dec min: ",torch.min(dec).cpu().detach().numpy())

        scores = self.autoencoder.compute_energy(dec, input_data.detach().data)
        scores = torch.sigmoid(scores)
        # print("scores.1shape: ", scores.shape)
        # print("scores max: ",torch.max(scores).cpu().detach().numpy())
        # print("scores min: ",torch.min(scores).cpu().detach().numpy())
        # print("mask max: ",torch.max(mask).cpu().detach().numpy())
        # print("mask min: ",torch.min(mask).cpu().detach().numpy())
        mask = nn.functional.interpolate(mask, size=dec.shape[2], mode="bilinear", align_corners=True)
        # mask = torch.squeeze(mask,0)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        # print("mask.1shape: ", mask.shape)

        # scores = torch.sigmoid(scores)
        total_loss_defect = self.autoencoder.loss_function(scores, mask.detach().data)
        # total_loss_defect = self.focal(scores,mask.detach().data)
        # print("total_loss_defect.1shape: ", total_loss_defect.shape)

        # scores = scores.reshape((scores.shape[0], 1, self.extractor.out_size[0], self.extractor.out_size[1]))  # test batch size is 1.
        # print("scores.2shape: ", scores)
        # if (torch.max(mask).cpu().numpy()==1.0): 
        #     np.savetxt("./train_mask1.txt", mask.data.cpu().numpy().squeeze(), fmt="%.2f")

        # total_loss_defect = self.BCEwithLog(scores,mask.squeeze())
        # total_loss_defect = torch.mean((scores-mask)**2)
        # print("total_loss feature: ",total_loss.shape)
        # print("mask feature: ",mask)
        # print("mask feature shape: ",mask.shape)
        # total_loss+=total_loss_defect
        # total_loss=total_loss_defect

        # total_loss+=ssim_loss
        # self.reset_grad()
        # total_loss.backward()
        total_loss_defect.backward()

        self.optimizer.step()
        # return total_loss
        return total_loss_defect
    
    def optimize_step_semi_mode_direct_to_target(self, input_data,abnormal_img,mask):
        self.extractor.train()
        self.autoencoder.train()

        self.optimizer.zero_grad()

        # forward
        input_data = self.extractor(input_data)
        # abnormal_img = self.extractor(abnormal_img)

        # print(input_data.size())
        dec = self.autoencoder(input_data)
        # dec_defect = self.autoencoder(abnormal_img)
        # loss
        # print("dec feature: ",dec_defect.shape)
        # print("abnormal_img feature: ",abnormal_img.shape)
        # print("mask shape: ",mask.shape)

        # total_loss = self.autoencoder.loss_function(dec, input_data.detach().data)
        # total_loss = self.autoencoder.loss_function(dec, input_data.detach().data)
        # print("total_loss.1shape: ", total_loss.shape)

        # ssim_loss= ssim(dec,input_data.detach().data)
        # total_loss_defect = self.autoencoder.loss_function(dec_defect, abnormal_img.detach().data)
        # total_loss_defect = self.autoencoder.loss_function(dec_defect, input_data.detach().data)
        # print("input_data max: ",torch.max(input_data).cpu().detach().numpy())
        # print("input_data min: ",torch.min(input_data).cpu().detach().numpy())
        # print("dec max: ",torch.max(dec).cpu().detach().numpy())
        # print("dec min: ",torch.min(dec).cpu().detach().numpy())

        # scores = self.autoencoder.compute_energy(dec, input_data.detach().data)
        # scores = torch.sigmoid(scores)
        scores = torch.mean(dec, dim=1)

        # print("scores.1shape: ", scores.shape)
        # print("scores max: ",torch.max(scores).cpu().detach().numpy())
        # print("scores min: ",torch.min(scores).cpu().detach().numpy())
        # print("mask max: ",torch.max(mask).cpu().detach().numpy())
        # print("mask min: ",torch.min(mask).cpu().detach().numpy())
        mask = nn.functional.interpolate(mask, size=dec.shape[2], mode="bilinear", align_corners=True)
        # mask = torch.squeeze(mask,0)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        # print("mask.1shape: ", mask.shape)

        # scores = torch.sigmoid(scores)
        total_loss_defect = self.autoencoder.loss_function(scores, mask.detach().data)
        # total_loss_defect = self.focal(scores,mask.detach().data)
        # print("total_loss_defect.1shape: ", total_loss_defect.shape)

        # scores = scores.reshape((scores.shape[0], 1, self.extractor.out_size[0], self.extractor.out_size[1]))  # test batch size is 1.
        # print("scores.2shape: ", scores)
        # if (torch.max(mask).cpu().numpy()==1.0): 
        #     np.savetxt("./train_mask1.txt", mask.data.cpu().numpy().squeeze(), fmt="%.2f")

        # total_loss_defect = self.BCEwithLog(scores,mask.squeeze())
        # total_loss_defect = torch.mean((scores-mask)**2)
        # print("total_loss feature: ",total_loss.shape)
        # print("mask feature: ",mask)
        # print("mask feature shape: ",mask.shape)
        # total_loss+=total_loss_defect
        # total_loss=total_loss_defect

        # total_loss+=ssim_loss
        # self.reset_grad()
        # total_loss.backward()
        total_loss_defect.backward()

        self.optimizer.step()
        # return total_loss
        return total_loss_defect

    def score_origin(self, input):
        """
        Args:
            input: image with size of (img_size_h, img_size_w, channels)
        Returns:
            score map with shape (img_size_h, img_size_w)
        """
        self.extractor.eval()
        self.autoencoder.eval()
        # print("input.shape: ",input.shape)

        input = self.extractor(input)
        dec = self.autoencoder(input)

        # sample energy
        scores = self.autoencoder.compute_energy(dec, input)
        scores = torch.sigmoid(scores)
      
        # print("extractor input.shape: ",input.shape)
        # print("dec.shape: ",dec.shape)
        # print("scores.1shape: ",scores.shape)
        scores = scores.reshape((1, 1, self.extractor.out_size[0], self.extractor.out_size[1]))    # test batch size is 1.
        # print("scores.2shape: ",scores.shape)

        scores = nn.functional.interpolate(scores, size=self.img_size, mode="bilinear", align_corners=True).squeeze()
        # print("scores.3shape: ",scores.shape)

        # print("score shape:", scores.shape)
        return scores
    
    def score(self, input):
        """
        Args:score_direct_to_target
            input: image with size of (img_size_h, img_size_w, channels)
        Returns:
            score map with shape (img_size_h, img_size_w)
        """
        self.extractor.eval()
        self.autoencoder.eval()
        # print("input.shape: ",input.shape)

        input = self.extractor(input)
        dec = self.autoencoder(input)

        # sample energy
        scores = self.autoencoder.compute_energy(dec, input)
        scores = torch.sigmoid(scores)
        # scores = torch.mean(dec, dim=1)
        # scores = torch.sigmoid(scores)
        # print("extractor input.shape: ",input.shape)
        # print("dec.shape: ",dec.shape)
        # print("scores.1shape: ",scores.shape)
        scores = scores.reshape((1, 1, self.extractor.out_size[0], self.extractor.out_size[1]))    # test batch size is 1.
        # print("scores.2shape: ",scores.shape)

        scores = nn.functional.interpolate(scores, size=self.img_size, mode="bilinear", align_corners=True).squeeze()
        # print("scores.3shape: ",scores.shape)

        # print("score shape:", scores.shape)
        return scores

    def segment(self, input, threshold=0.5):
        """
        Args:
            input: image with size of (img_size_h, img_size_w, channels)
        Returns:
            score map and binary score map with shape (img_size_h, img_size_w)
        """
        # predict
        scores = self.score(input).data.cpu().numpy()

        # binary score
        # print("threshold:", threshold)
        binary_scores = np.zeros_like(scores)    # torch.zeros_like(scores)
        binary_scores[scores <= threshold] = 0
        binary_scores[scores > threshold] = 1

        return scores, binary_scores

    def segment_evaluation(self):
        i = 0
        metrics = []
        Mean_AUROC=0
        time_start = time.time()
        for i, (img, mask, name) in enumerate(self.test_data_loader):    # batch size is 1.
            i += 1

            # segment
            img = img.to(self.device)
            scores, binary_scores = self.segment(img, threshold=self.threshold)

            # show something
            #     plt.figure()
            #     ax1 = plt.subplot(1, 2, 1)
            #     ax1.imshow(resize(mask[0], (256, 256)))
            #     ax1.set_title("gt")

            #     ax2 = plt.subplot(1, 2, 2)
            #     ax2.imshow(scores)
            #     ax2.set_title("pred")

            mask = mask.squeeze().numpy()
            name = name[0]
            # save results
            # print("save_seg_results")
            self.save_seg_results(normalize(scores), binary_scores, mask, name)
            # metrics of one batch
            if name.split("/")[-2] != "good":
                specificity, sensitivity, accuracy, coverage, auc = spec_sensi_acc_iou_auc(mask, binary_scores, scores)
                metrics.append([specificity, sensitivity, accuracy, coverage, auc])
                Mean_AUROC+=auc

            # print("Batch {},".format(i), "Cost total time {}s".format(time.time()-time_start))
        # metrics over all data
        metrics = np.array(metrics)
        metrics_mean = metrics.mean(axis=0)
        metrics_std = metrics.std(axis=0)
        print("metrics: specificity, sensitivity, accuracy, iou, auc")
        print("mean:", metrics_mean)
        print("std:", metrics_std)
        print("threshold:", self.threshold)
        print("threshold:",Mean_AUROC/ len(self.test_data_loader))
        # self.threshold = Mean_AUROC/ len(self.test_data_loader)
    def save_paths(self):
        # generating saving paths
        score_map_path = os.path.join(self.cfg.save_path, self.subpath+"/Results" + "/score_map")
        if not os.path.exists(score_map_path):
            os.makedirs(score_map_path)

        binary_score_map_path = os.path.join(self.cfg.save_path, self.subpath+"/Results" + "/binary_score_map")
        if not os.path.exists(binary_score_map_path):
            os.makedirs(binary_score_map_path)

        gt_pred_map_path = os.path.join(self.cfg.save_path, self.subpath+"/Results" + "/gt_pred_score_map")
        if not os.path.exists(gt_pred_map_path):
            os.makedirs(gt_pred_map_path)

        mask_path = os.path.join(self.cfg.save_path, self.subpath+"/Results" + "/mask")
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)

        gt_pred_seg_image_path = os.path.join(self.cfg.save_path, self.subpath+"/Results" + "/gt_pred_seg_image")
        if not os.path.exists(gt_pred_seg_image_path):
            os.makedirs(gt_pred_seg_image_path)

        return score_map_path, binary_score_map_path, gt_pred_map_path, mask_path, gt_pred_seg_image_path

    def save_seg_results(self, scores, binary_scores, mask, name):
        print("name: ",name)
        score_map_path, binary_score_map_path, gt_pred_score_map, mask_path, gt_pred_seg_image_path = self.save_paths()
        img_name = name.split("/")
        img_name = "-".join(img_name[-2:])
        # print(img_name)
        # score map
        # print("score_map_path: ", score_map_path)
        #imsave(os.path.join(score_map_path, "{}".format(img_name)), (scores*255.0).astype('uint8'))
        imsave(os.path.join(score_map_path, "{}".format(img_name)), scores)

        #imsave(os.path.join(score_map_path, "{}".format(img_name)), img_as_ubyte((scores))*255)

        # binary score map
        #imsave(os.path.join(binary_score_map_path, "{}".format(img_name)), (binary_scores*255.0).astype('uint8'))
        imsave(os.path.join(binary_score_map_path, "{}".format(img_name)), binary_scores)

        #imsave(os.path.join(binary_score_map_path, "{}".format(img_name)), img_as_ubyte((binary_scores))*255)

        # mask
        imsave(os.path.join(mask_path, "{}".format(img_name)), mask)

        # # pred vs gt map
        # imsave(os.path.join(gt_pred_score_map, "{}".format(img_name)), normalize(binary_scores + mask))
        visulization_score(img_file=name, mask_path=mask_path,
                     score_map_path=score_map_path, saving_path=gt_pred_score_map)
        # pred vs gt image
        visulization(img_file=name, mask_path=mask_path,
                     score_map_path=binary_score_map_path, saving_path=gt_pred_seg_image_path)

    def save_model(self, epoch=0):
        # save model weights
        torch.save({'autoencoder': self.autoencoder.state_dict()},
                   os.path.join(self.model_path, 'autoencoder.pth'))
        np.save(os.path.join(self.model_path, 'n_dim.npy'), self.n_dim)

    def load_model(self, path=None):
        print("Loading model...")
        if path is None:
            model_path = os.path.join(self.model_path, 'autoencoder.pth')
            print("model path:", model_path)
            if not os.path.exists(model_path):
                print("Model not exists.")
                return False

            if torch.cuda.is_available():
                data = torch.load(model_path)
            else:
                data = torch.load(model_path,
                                  map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU, using a function

            self.autoencoder.load_state_dict(data['autoencoder'])
            print("Model loaded:", model_path)
        return True

    # def save_dim(self):
    #     np.save(os.path.join(self.model_path, 'n_dim.npy'))

    def load_dim(self, model_path):
        dim_path = os.path.join(model_path, 'n_dim.npy')
        if not os.path.exists(dim_path):
            print("Dim not exists.")
            self.n_dim = None
        else:
            self.n_dim = np.load(os.path.join(model_path, 'n_dim.npy'))

    ########################################################
    #  Evaluation (testing)
    ########################################################
    def segmentation_results(self):
        def normalize(x):
            return x/x.max()

        for i, (img, mask, name) in enumerate(self.test_data_loader):    # batch size is 1.
            i += 1
            time_start = time.time()

            # segment
            img = img.to(self.device)
            scores, binary_scores = self.segment(img, threshold=self.threshold)

            mask = mask.squeeze().numpy()
            name = name[0]
            # save results
            if name[0].split("/")[-2] != "good":
                self.save_seg_results(normalize(scores), binary_scores, mask, name)
            # self.save_seg_results((scores-score_min)/score_range, binary_scores, mask, name)
            print("Batch {},".format(i), "Cost total time {}s".format(time.time()-time_start))

    ############################f##########################
    #  Evaluation of segmentation
    ######################################################
    def save_segment_paths(self, fpr):
        # generating saving paths
        binary_score_map_path = os.path.join(self.cfg.save_path, self.subpath+"/Results" + "/fpr_{}/binary_score_map".format(fpr))
        if not os.path.exists(binary_score_map_path):
            os.makedirs(binary_score_map_path)
        print("binary_score_map_path: ",binary_score_map_path)
        mask_path = os.path.join(self.cfg.save_path, self.subpath +"/Results"+ "/fpr_{}/mask".format(fpr))
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)

        gt_pred_seg_image_path = os.path.join(self.cfg.save_path, self.subpath+"/Results" + "/fpr_{}/gt_pred_seg_image".format(fpr))
        if not os.path.exists(gt_pred_seg_image_path):
            os.makedirs(gt_pred_seg_image_path)

        return binary_score_map_path, mask_path, gt_pred_seg_image_path

    def save_segment_results(self, binary_scores, mask, name, fpr):
        binary_score_map_path, mask_path, gt_pred_seg_image_path = self.save_segment_paths(fpr)
        img_name = name.split("/")
        img_name = "-".join(img_name[-2:])
        print(img_name)
        # binary score map
        imsave(os.path.join(binary_score_map_path, "{}".format(img_name)), binary_scores)

        # mask
        imsave(os.path.join(mask_path, "{}".format(img_name)), mask)

        # pred vs gt image
        visulization(img_file=name, mask_path=mask_path,
                     score_map_path=binary_score_map_path, saving_path=gt_pred_seg_image_path)

    def estimate_thred_with_fpr(self, expect_fpr=0.05):
        """
        Use training set to estimate the threshold.
        """
        threshold = 0
        scores_list = []
        for i, normal_img in enumerate(self.train_data_loader):
            normal_img = normal_img[0:1].to(self.device)
            scores_list.append(self.score(normal_img).data.cpu().numpy())
        scores = np.concatenate(scores_list, axis=0)

        # find the optimal threshold
        max_step = 100
        min_th = scores.min()
        max_th = scores.max()
        delta = (max_th - min_th) / max_step
        for step in range(max_step):
            threshold = max_th - step * delta
            # segmentation
            binary_score_maps = np.zeros_like(scores)
            binary_score_maps[scores <= threshold] = 0
            binary_score_maps[scores > threshold] = 1

            # estimate the optimal threshold base on user defined min_area
            fpr = binary_score_maps.sum() / binary_score_maps.size
            print(
                "threshold {}: find fpr {} / user defined fpr {}".format(threshold, fpr, expect_fpr))
            if fpr >= expect_fpr:  # find the optimal threshold
                print("find optimal threshold:", threshold)
                print("Done.\n")
                break
        return threshold

    def segment_evaluation_with_fpr(self, expect_fpr=0.05):
        # estimate threshold
        thred = self.estimate_thred_with_fpr(expect_fpr=expect_fpr)

        # segment
        i = 0
        metrics = []
        time_start = time.time()
        for i, (img, mask, name) in enumerate(self.test_data_loader):    # batch size is 1.
            i += 1

            # segment
            img = img.to(self.device)
            scores, binary_scores = self.segment(img, threshold=thred)

            mask = mask.squeeze().numpy()
            name = name[0]
            # save results
            self.save_segment_results(binary_scores, mask, name, expect_fpr)
            print("Batch {},".format(i), "Cost total time {}s".format(time.time()-time_start))
        print("threshold:", thred)

    def segment_evaluation_with_otsu_li(self, seg_method='otsu'):
        """
        ref: skimage.filters.threshold_otsu
        skimage.filters.threshold_li
        e.g.
        thresh = filters.threshold_otsu(image) #返回一个阈值
        dst =(image <= thresh)*1.0 #根据阈值进行分割
        """
        from skimage.filters import threshold_li
        from skimage.filters import threshold_otsu

        # segment
        thred = 0
        time_start = time.time()
        for i, (img, mask, name) in enumerate(self.test_data_loader):    # batch size is 1.
            i += 1

            # segment
            img = img.to(self.device)

            # estimate threshold and seg
            if seg_method == 'otsu':
                thred = threshold_otsu(img.detach().cpu().numpy())
            else:
                thred = threshold_li(img.detach().cpu().numpy())
            scores, binary_scores = self.segment(img, threshold=thred)

            mask = mask.squeeze().numpy()
            name = name[0]
            # save results
            self.save_segment_results(binary_scores, mask, name, seg_method)
            print("Batch {},".format(i), "Cost total time {}s".format(time.time()-time_start))
        print("threshold:", thred)

    def segmentation_evaluation(self):
        if self.load_model():
            print("Model Loaded.")
        else:
            print("None pretrained models.")
            return
        self.segment_evaluation_with_fpr(expect_fpr=self.cfg.except_fpr)

    def validation(self, epoch):
        i = 0
        masks = []
        scores = []
        for i, (img, mask, name) in enumerate(self.test_data_loader):  # batch size is 1.
            i += 1
            # data
            time_start = time.time()

            img = img.to(self.device)
            mask = mask.squeeze().numpy()
            # print("test mask: ",mask)
            # np.savetxt("./test_mask",mask.squeeze(),fmt="%.2f")
            # score
            score = self.score(img).data.cpu().numpy()
            # np.savetxt("./test_score",score.squeeze(),fmt="%.2f")
            # np.savetxt("./test_img",img.data.cpu().numpy().squeeze()[0],fmt="%.2f")

            # auc_roc(mask,score)
            masks.append(mask)
            scores.append(score)
            print("Batch {},".format(i), "Cost total time {}s".format(time.time() - time_start))

        # as array
        masks = np.array(masks)
        masks[masks <= 0.5] = 0
        masks[masks > 0.5] = 1
        masks = masks.astype(np.bool)
        scores = np.array(scores)

        # auc score
        print("mask: ",masks.shape)
        auc_score, roc = auc_roc(masks, scores)
        # p,r,t = sklearn.metrics.precision_recall_curve(masks,scores)
        # auc_pr = sklearn.metrics.auc(r,p)
        # metrics over all data
        print("auc_roc:", auc_score)
        # print("auc_pr:", roc[0],roc[1],roc[2])
        
        out_file = os.path.join(self.eval_path, '{}_epoch_auc.csv'.format(self.model_name))
        if not os.path.exists(out_file):
            with open(out_file, mode='w') as f:
                f.write("Epoch" + ",AUC" + "\n")
        with open(out_file, mode='a+') as f:
            f.write(str(epoch) + "," + str(auc_score) + "\n")

    def metrics_evaluation(self, expect_fpr=0.3, max_step=5000):
        from sklearn.metrics import auc
        from sklearn.metrics import roc_auc_score, average_precision_score
        from skimage import measure
        import pandas as pd

        if self.load_model():
            print("Model Loaded.")
        else:
            print("None pretrained models.")
            return

        print("Calculating AUC, IOU, PRO metrics on testing data...")
        time_start = time.time()
        masks = []
        scores = []
        for i, (img, mask, name) in enumerate(self.test_data_loader):  # batch size is 1.
            # data
            img = img.to(self.device)
            mask = mask.squeeze().numpy()

            # anomaly score
            # anomaly_map = self.score(img).data.cpu().numpy()
            anomaly_map = self.score(img).data.cpu().numpy()

            masks.append(mask)
            scores.append(anomaly_map)
            #print("Batch {},".format(i), "Cost total time {}s".format(time.time() - time_start))

        # as array
        masks = np.array(masks)
        scores = np.array(scores)
        
        # binary masks
        masks[masks <= 0.5] = 0
        masks[masks > 0.5] = 1
        masks = masks.astype(np.bool)
        
        # auc score (image level) for detection
        labels = masks.any(axis=1).any(axis=1)
#         preds = scores.mean(1).mean(1)
        preds = scores.max(1).max(1)    # for detection
        det_auc_score = roc_auc_score(labels, preds)
        det_pr_score = average_precision_score(labels, preds)
        
        # auc score (per pixel level) for segmentation
        seg_auc_score = roc_auc_score(masks.ravel(), scores.ravel())
        seg_pr_score = average_precision_score(masks.ravel(), scores.ravel())
        # metrics over all data
        print(f"Det AUC: {det_auc_score:.4f}, Seg AUC: {seg_auc_score:.4f}")
        print(f"Det PR: {det_pr_score:.4f}, Seg PR: {seg_pr_score:.4f}")
        
        # per region overlap and per image iou
        max_th = scores.max()
        min_th = scores.min()
        delta = (max_th - min_th) / max_step
        
        ious_mean = []
        ious_std = []
        pros_mean = []
        pros_std = []
        threds = []
        fprs = []
        binary_score_maps = np.zeros_like(scores, dtype=np.bool)
        for step in range(max_step):
            thred = max_th - step * delta
            # segmentation
            binary_score_maps[scores <= thred] = 0
            binary_score_maps[scores > thred] = 1

            pro = []    # per region overlap
            iou = []    # per image iou
            # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
            # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map 
            for i in range(len(binary_score_maps)):    # for i th image
                # pro (per region level)
                label_map = measure.label(masks[i], connectivity=2)
                props = measure.regionprops(label_map)
                for prop in props:
                    x_min, y_min, x_max, y_max = prop.bbox    # find the bounding box of an anomaly region 
                    cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                    # cropped_mask = masks[i][x_min:x_max, y_min:y_max]   # bug!
                    cropped_mask = prop.filled_image    # corrected!
                    intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                    pro.append(intersection / prop.area)
                # iou (per image level)
                intersection = np.logical_and(binary_score_maps[i], masks[i]).astype(np.float32).sum()
                union = np.logical_or(binary_score_maps[i], masks[i]).astype(np.float32).sum()
                if masks[i].any() > 0:    # when the gt have no anomaly pixels, skip it
                    iou.append(intersection / union)
            # against steps and average metrics on the testing data
            ious_mean.append(np.array(iou).mean())
#             print("per image mean iou:", np.array(iou).mean())
            ious_std.append(np.array(iou).std())
            pros_mean.append(np.array(pro).mean())
            pros_std.append(np.array(pro).std())
            # fpr for pro-auc
            masks_neg = ~masks
            fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
            fprs.append(fpr)
            threds.append(thred)
            
        # as array
        threds = np.array(threds)
        pros_mean = np.array(pros_mean)
        pros_std = np.array(pros_std)
        fprs = np.array(fprs)
        
        ious_mean = np.array(ious_mean)
        ious_std = np.array(ious_std)
        
        # save results
        data = np.vstack([threds, fprs, pros_mean, pros_std, ious_mean, ious_std])
        df_metrics = pd.DataFrame(data=data.T, columns=['thred', 'fpr',
                                                        'pros_mean', 'pros_std',
                                                        'ious_mean', 'ious_std'])
        # save results
        df_metrics.to_csv(os.path.join(self.eval_path, 'thred_fpr_pro_iou.csv'), sep=',', index=False)

        
        # best per image iou
        best_miou = ious_mean.max()
        print(f"Best IOU: {best_miou:.4f}")
        
        # default 30% fpr vs pro, pro_auc
        idx = fprs <= expect_fpr    # find the indexs of fprs that is less than expect_fpr (default 0.3)
        fprs_selected = fprs[idx]
        fprs_selected = rescale(fprs_selected)    # rescale fpr [0,0.3] -> [0, 1]
        pros_mean_selected = pros_mean[idx]    
        pro_auc_score = auc(fprs_selected, pros_mean_selected)
        print("pro auc ({}% FPR):".format(int(expect_fpr*100)), pro_auc_score)

        # save results
        data = np.vstack([threds[idx], fprs[idx], pros_mean[idx], pros_std[idx]])
        df_metrics = pd.DataFrame(data=data.T, columns=['thred', 'fpr',
                                                        'pros_mean', 'pros_std'])
        df_metrics.to_csv(os.path.join(self.eval_path, 'thred_fpr_pro_{}.csv'.format(expect_fpr)), sep=',', index=False)

        # save auc, pro as 30 fpr
        with open(os.path.join(self.eval_path, 'pr_auc_pro_iou_{}.csv'.format(expect_fpr)), mode='w') as f:
                f.write("det_pr, det_auc, seg_pr, seg_auc, seg_pro, seg_iou\n")
                f.write(f"{det_pr_score:.5f},{det_auc_score:.5f},{seg_pr_score:.5f},{seg_auc_score:.5f},{pro_auc_score:.5f},{best_miou:.5f}")    
            

    def metrics_detecion(self, expect_fpr=0.3, max_step=5000):
        from sklearn.metrics import auc
        from sklearn.metrics import roc_auc_score, average_precision_score
        from skimage import measure
        import pandas as pd

        if self.load_model():
            print("Model Loaded.")
        else:
            print("None pretrained models.")
            return

        print("Calculating AUC, IOU, PRO metrics on testing data...")
        time_start = time.time()
        masks = []
        scores = []
        for i, (img, mask, name) in enumerate(self.test_data_loader):  # batch size is 1.
            # data
            img = img.to(self.device)
            mask = mask.squeeze().numpy()

            # anomaly score
            # anomaly_map = self.score(img).data.cpu().numpy()
            anomaly_map = self.score(img).data.cpu().numpy()

            masks.append(mask)
            scores.append(anomaly_map)
            #print("Batch {},".format(i), "Cost total time {}s".format(time.time() - time_start))

        # as array
        masks = np.array(masks)
        scores = np.array(scores)
        
        # binary masks
        masks[masks <= 0.5] = 0
        masks[masks > 0.5] = 1
        masks = masks.astype(np.bool)
        
        # auc score (image level) for detection
        labels = masks.any(axis=1).any(axis=1)
#         preds = scores.mean(1).mean(1)
        preds = scores.max(1).max(1)    # for detection
        det_auc_score = roc_auc_score(labels, preds)
        det_pr_score = average_precision_score(labels, preds)
        
        # auc score (per pixel level) for segmentation
        seg_auc_score = roc_auc_score(masks.ravel(), scores.ravel())
        seg_pr_score = average_precision_score(masks.ravel(), scores.ravel())
        # metrics over all data
        print(f"Det AUC: {det_auc_score:.4f}, Seg AUC: {seg_auc_score:.4f}")
        print(f"Det PR: {det_pr_score:.4f}, Seg PR: {seg_pr_score:.4f}")
        
        # save detection metrics
        with open(os.path.join(self.eval_path, 'det_pr_auc.csv'), mode='w') as f:
                f.write("det_pr, det_auc,seg_auc,seg_pr\n")
                f.write(f"{det_pr_score:.5f},{det_auc_score:.5f},{seg_auc_score:.4f},{seg_pr_score:.4f}") 
            
