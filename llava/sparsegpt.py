import math
import time

import torch
import torch.nn as nn
import transformers

# from quant import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseGPT:

    def __init__(self, layer, pretrain_layer):
        self.layer = layer
        self.pretrain_layer = pretrain_layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev) #[768,768]
        self.nsamples = 0

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01, tops_mask=None, use_tops=False
    ):
        W = self.layer.weight.data.clone() #[768,768]
        pretrain_weights = self.pretrain_layer.weight.data.clone()
        W_onlymask = self.layer.weight.data.clone() #[768,768]

        delta_fp = W - pretrain_weights

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
            W_onlymask = W_onlymask.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
            W_onlymask = W_onlymask.t()
        W = W.float()
        W_onlymask = W_onlymask.float()


        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1 #如果对角线某个元素为0则设置为1
        # W[:, dead] = 0 

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp #对角线元素 加上 对角线元素的平均值*0.01
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H #求H的逆矩阵

        mask = None

        param_changes =  torch.abs(W - pretrain_weights) 
        flattened_changes = param_changes.flatten()
        sorted_changes, _ = torch.sort(flattened_changes)
        top_s_threshold = sorted_changes[int(len(sorted_changes) * sparsity)] # len(sorted_changes) = 188837888，这个参数数量是可训练的参数量，也就是q-former的参数量
        mask = param_changes <= top_s_threshold

        hessian_mask = torch.rand_like(W)

        hybrid_mask = torch.rand_like(W)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone() #[768,128]
            Q1 = torch.zeros_like(W1) #[768,128]
            Err1 = torch.zeros_like(W1) #[768,128]
            Losses1 = torch.zeros_like(W1) #[768,128]
            Hinv1 = Hinv[i1:i2, i1:i2] #[128,128]

            if use_tops == False:
                if prunen == 0: 
                    if mask is not None:
                        # mask1 = mask[:, i1:i2]
                        
                        tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                        param_changes_mean = param_changes[:, i1:i2].mean(dim=1, keepdim=True)
                        param_changes_std = param_changes[:, i1:i2].std(dim=1, keepdim=True)
                        tmp_mean = tmp.mean(dim=1, keepdim=True)
                        tmp_std = tmp.std(dim=1, keepdim=True)
                        param_changes_normalized = (param_changes[:, i1:i2] - param_changes_mean) / param_changes_std
                        tmp_normalized = (tmp - tmp_mean) / tmp_std
                        importance_score = 0.5 * param_changes_normalized + 0.5 * tmp_normalized
                        thresh = torch.sort(importance_score.flatten())[0][int(importance_score.numel() * sparsity)]
                        mask1 = importance_score <= thresh 
                        hybrid_mask[:, i1:i2] = mask1
                        
                        # thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                        # hessian_mask[:, i1:i2] = tmp <= thresh

                    else:
                        tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                        thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                        mask1 = tmp <= thresh #[768,128]
                else:
                    mask1 = torch.zeros_like(W1) == 1

            for i in range(count): # count 128
                w = W1[:, i] #[768]
                d = Hinv1[i, i] #scalar

                if use_tops == False:
                    if prunen != 0 and i % prunem == 0:
                        tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                        mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)
                else:
                    mask1 = tops_mask[:, i1:i2]

                # q = w.clone()
                # q[mask1[:, i]] = 0
                q = w.clone()
                mask_indices = mask1[:, i]
                q[mask_indices] = pretrain_weights[:, i1 + i][mask_indices]

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                # W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                W1[:, i:] += err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
                

            W[:, i1:i2] = Q1
            W_onlymask[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            # W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) #减去的这个值 对于被mask掉的参数来讲是不是参数值本身
            W[:, i2:] += Err1.matmul(Hinv[i1:i2, i2:]) 
            # scale_mask = 1.2
            

        # print("scale_mask:",scale_mask)
        delta_sf = torch.zeros_like(W)
        #delta_sf = (W - self.layer.weight.data) * (1-tops_mask.int())
        delta_sp = W - pretrain_weights # 补偿值
        param_total_mask = ((W_onlymask - pretrain_weights)!= 0).int() # mask值

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        return delta_sf, delta_fp, delta_sp, param_total_mask

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()