import torch
from modules.FSIPANet import FSIPANet, VGG16FeatureExtractor
from modules.Losses import AdversarialLoss
import torch.optim as optim
from utils.io import load_ckpt
from utils.io import save_ckpt
from torchvision.utils import make_grid
from torchvision.utils import save_image
import torch.nn.functional as F
from modules.Discriminator import Discriminator
from torch import autograd
import os
import time
import visdom
import numpy as np
import pandas as pd


class FSIPAModel():
    def __init__(self):
        self.G = None
        self.lossNet = None
        self.iter = None
        self.optm_G = None
        self.device = None
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.comp_B = None
        self.grey = None
        self.l1_loss_val = 0.0
        self.loss_D=0.0
        self.loss_G=0.0
        self.loss_tv=0.0  
        self.loss_style=0.0   
        self.loss_perceptual=0.0  
        self.loss_valid=0.0  
        self.loss_hole=0.0  
        self.loss_edge=0.0
    
    def initialize_model(self, path_g=None, path_d=None, train=True):
        self.G = FSIPANet()
        if train:
            self.lossNet = VGG16FeatureExtractor()
            self.D = Discriminator(4)
            self.optm_G = optim.Adam(self.G.parameters(), lr = 2e-4)
            self.optm_D = optim.Adam(self.D.parameters(), lr = 2e-5)
            self.adversarial_loss = AdversarialLoss()
        try:
            start_iter = load_ckpt(path_g, [('generator', self.G)], [('optimizer_G', self.optm_G)])
            if train:
                start_iter = load_ckpt(path_d, [('discriminator', self.D)], [('optimizer_D', self.optm_D)])
                self.optm_G = optim.Adam(self.G.parameters(), lr = 2e-4)
                self.optm_D = optim.Adam(self.D.parameters(), lr = 2e-5)
                self.iter = start_iter
                print('Model Initialized, iter: ', start_iter)
        except:
            self.iter = 0
            print('No trained model, train from beginning')
        
    def cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Model moved to cuda")
            self.G.cuda()
            if self.lossNet is not None:
                self.lossNet.cuda()
                self.D.cuda()
                self.adversarial_loss.cuda()
        else:
            self.device = torch.device("cpu")
        
    def train(self, train_loader, save_path, finetune = False):
        vis=visdom.Visdom(env=u'Places2')
        self.G.train(finetune = finetune)
        if finetune:
            print("Starting to finetune")
            self.optm_G = optim.Adam(filter(lambda p:p.requires_grad, self.G.parameters()), lr = 5e-5)
            self.optm_D = optim.Adam(self.D.parameters(), lr = 5e-6)
        keep_training = True
        print("Starting training from iteration:{:d}".format(self.iter))
        s_time = time.time()
        epoch=0

        while keep_training:
            epoch += 1
            for items in train_loader:
                gt_images, grey_image, gt_edges,gt_edgescanny, masks = self.__cuda__(*items)
                masked_images = gt_images * masks[:,0:1,:,:]
                gt_mixed_edges=torch.cat([gt_edges,gt_edgescanny],dim=1)
                masked_edges = gt_mixed_edges * masks[:,0:1,:,:]
                self.mask1=masks
                masks = torch.cat([masks]*3, dim = 1)
                masks = masks[:,:3,:,:]
                self.grey = grey_image
                self.forward(masked_images, masks, masked_edges, gt_images, gt_mixed_edges)
                self.update_parameters()
                self.iter += 1
                if self.iter % 50 == 0:
                    e_time = time.time()
                    int_time = e_time - s_time                    
                    print("epoch:%d,Iteration:%d, l1_loss:%.4f, time_taken:%.2f" %(epoch,self.iter, self.l1_loss_val/50, int_time))
                    vis.line(Y=np.column_stack((np.array(self.l1_loss_val.cpu().numpy()/50),np.array(self.loss_D.cpu().numpy()/50),np.array(self.loss_G.cpu().numpy()/50),np.array(self.loss_tv.cpu().numpy()/50),np.array(self.loss_style.cpu().numpy()/50),np.array(self.loss_perceptual.cpu().numpy()/50),np.array(self.loss_valid.cpu().numpy()/50),np.array(self.loss_hole.cpu().numpy()/50),np.array(self.loss_edge.cpu().numpy()/50))),X=np.column_stack((np.array(self.iter),np.array(self.iter),np.array(self.iter),np.array(self.iter),np.array(self.iter),np.array(self.iter),np.array(self.iter),np.array(self.iter),np.array(self.iter))),win='Places2',opts=dict(title='loss_curve',legend=['l1_loss','loss_D','loss_G','loss_tv','loss_style','loss_perceptual','loss_valid','loss_hole','loss_edge']),update='append')

                    s_time = time.time()
                    self.l1_loss_val = 0.0
                    self.loss_D=0.0
                    self.loss_G=0.0
                    self.loss_tv=0.0  
                    self.loss_style=0.0   
                    self.loss_perceptual=0.0  
                    self.loss_valid=0.0  
                    self.loss_hole=0.0  
                    self.loss_edge=0.0
                                
                    
                
                if self.iter % 20000 == 0:
                    if not os.path.exists('{:s}'.format(save_path)):
                        os.makedirs('{:s}'.format(save_path))
                    save_ckpt('{:s}/g_{:d}.pth'.format(save_path, self.iter ), [('generator', self.G)], [('optimizer_G', self.optm_G)], self.iter )
                    save_ckpt('{:s}/d_{:d}.pth'.format(save_path, self.iter ), [('discriminator', self.D)], [('optimizer_D', self.optm_D)], self.iter )
        
    def test(self, test_loader, result_save_path):
        print("Starting to test")
        self.G.eval()
        for para in self.G.parameters():
            para.requires_grad = False
        count = 0
        for items in test_loader:
            gt_images, grey_image, gt_edges,gt_edgescanny, masks = self.__cuda__(*items)
            masked_images = gt_images * masks[:,0:1,:,:]
            gt_mixed_edges=torch.cat([gt_edges,gt_edgescanny],dim=1)
            masked_edges = gt_mixed_edges * masks[:,0:1,:,:]
            masks = torch.cat([masks]*3, dim = 1)
            masks = masks[:,:3,:,:]
            fake_B, _, _, _ = self.G(masked_images, masks, masked_edges)
            comp_B = fake_B * (1 - masks) + gt_images * masks
            comp_img_dir = os.path.join(result_save_path, 'comp_img')
            masked_img_dir = os.path.join(result_save_path, 'masked_img')
            if not os.path.exists(comp_img_dir):
                os.makedirs(comp_img_dir)
            #if not os.path.exists(masked_img_dir):
                #os.makedirs(masked_img_dir)
 
            for k in range(comp_B.size(0)):
                count += 1
                s = '%04d' % count  # 
                grid = make_grid(comp_B[k:k+1])
                file_path = '{:s}/places2_{:s}.jpg'.format(comp_img_dir, s)
                save_image(grid, file_path)
                """
                grid = make_grid(masked_images[k:k+1] +1 - masks[k:k+1] )
                file_path = '{:s}/masked_img_{:d}.jpg'.format(masked_img_dir, count)
                save_image(grid, file_path)
                """
    
    def forward(self, masked_image, mask, masked_edge, gt_image, gt_edge):
        self.real_A = masked_image
        self.real_B = gt_image
        self.mask = mask
        self.edge_gt = gt_edge
        fake_B, _, edge_small, edge_big = self.G(masked_image, mask, masked_edge)
        self.fake_B = fake_B
        self.edge_fake = [edge_small, edge_big]
        self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask
    
    def update_parameters(self):
        self.update_G()
        self.update_D()
    
    def update_G(self):
        self.optm_G.zero_grad()
        loss_G = self.get_g_loss()
        loss_G.backward()
        self.optm_G.step()
    
    def update_D(self):
        self.optm_D.zero_grad()
        loss_D = self.get_d_loss()
        loss_D.backward()
        self.optm_D.step()
    
    def get_d_loss(self):
        real_edges = self.edge_gt
        fake_edges = self.edge_fake
        loss_D = 0
        for i in range(2):
            fake_edge = fake_edges[i]
            real_edge = F.interpolate(real_edges, size = fake_edge.size()[2:])
            
            real_edge = real_edge.detach()
            fake_edge = fake_edge.detach()


            pred_real, _ = self.D(real_edge)
            pred_fake, _ = self.D(fake_edge)
            gp = self.gradient_penalty(self.D,real_edge,fake_edge,True, 10.0)
            loss_D += (self.adversarial_loss(pred_real, True, True)  + self.adversarial_loss(pred_fake, False, True))+gp
        self.loss_D += loss_D.detach()
        return loss_D.sum()
  
    def get_g_loss(self):
        real_B = self.real_B
        fake_B = self.fake_B
        comp_B = self.comp_B
        real_B_feats = self.lossNet(real_B)
        fake_B_feats = self.lossNet(fake_B)
        comp_B_feats = self.lossNet(comp_B)
        real_edge = self.edge_gt
        fake_edge = self.edge_fake
        tv_loss = self.TV_loss(comp_B * (1 - self.mask))
        style_loss = self.style_loss(real_B_feats, fake_B_feats) + self.style_loss(real_B_feats, comp_B_feats)
        preceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats) + self.preceptual_loss(real_B_feats, comp_B_feats)
        valid_loss = self.l1_loss(real_B, fake_B, self.mask)
        hole_loss = self.l1_loss(real_B, fake_B, (1 - self.mask))
        adv_loss_0 = self.edge_loss(fake_edge[1], real_edge)
        adv_loss_1 = self.edge_loss(fake_edge[0],F.interpolate(real_edge, scale_factor = 0.5))        
        adv_loss = adv_loss_0 + adv_loss_1
        
        loss_G = (  tv_loss * 0.01
                  + style_loss * 120
                  + preceptual_loss * 0.05
                  + valid_loss * 50
                  + hole_loss * 50) + 0.1 * adv_loss
        self.l1_loss_val += valid_loss.detach() + hole_loss.detach()
        self.loss_tv += tv_loss.detach()
        self.loss_style += style_loss.detach()
        self.loss_perceptual += preceptual_loss.detach()
        self.loss_valid += valid_loss.detach()
        self.loss_hole += hole_loss.detach()
        self.loss_edge += adv_loss.detach()
        self.loss_G += loss_G.detach()
        return loss_G
    
    def l1_loss(self, f1, f2, mask = 1):
        return torch.mean(torch.abs(f1 - f2)*mask)
    
    def edge_loss(self, fake_edge, real_edge):
        pred_fake, features_edge1 = self.D(fake_edge)
        return self.adversarial_loss(pred_fake, True, False)
    
    def style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
        return loss_value
    
    def TV_loss(self, x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
        w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
        return h_tv + w_tv
    
    def preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value

    def gradient_penalty(self,netD, real_data, fake_data, cuda, Lambda):
        BATCH_SIZE = real_data.size()[0]
        DIM = real_data.size()[2]
        alpha = torch.rand(BATCH_SIZE, 1)
        c=int(real_data.nelement())
        alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous()
        alpha = alpha.view(BATCH_SIZE, 4, DIM, DIM)
        if cuda:
            alpha = alpha.cuda()
        fake_data = fake_data.view(BATCH_SIZE, 4, DIM, DIM)
        interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

        if cuda:
            interpolates = interpolates.cuda()
        interpolates.requires_grad_(True)

        disc_interpolates, _  = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if cuda else torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)                              
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda
        return gradient_penalty.sum().mean()
            
    def __cuda__(self, *args):
        return (item.to(self.device) for item in args)
            