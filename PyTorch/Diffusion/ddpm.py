import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=2e-2, img_size=64, device="cpu"):
        
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        # Small resolution here. You would rather seperate sampler 
        # for getting to higher reslutions.
        self.device = device
        
        self.beta = self.perpare_noise_schedule().to(device)
        
        # alpha_t = 1 = beta_t
        # alpha_hat = ∏(alpha_i) over all i till noise_steps
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) #cumulative product
    
    def perpare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        '''
        x_t = √(alpha_hat) * x_0 + √(1 - alpha_hat) * epsilon
        '''
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:,None,None,None]
        sqrt_one_minus_alpha_hat = torch.sqrt((1. - self.alpha_hat[t]))[:,None,None,None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat*x + sqrt_one_minus_alpha_hat*eps

    def sample_timesteps(self, n):
        '''
        Sample time steps
        '''
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        
        logging.info(f"Sampling {n} new images ••• ")
        model.eval()
        
        with torch.no_grad():
            # Sampling from normal distribution
            x = torch.randn((n,3,self.img_size, self.img_size)).to(self.device)
            
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                # n-length vector of time (i)
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x,t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                # Add noise only at timesteps after 1
                if i > 1:
                    noise = torch.randn_like(x)
                else: 
                    noise = torch.zeros_like(x)
                
                x = 1 / torch.sqrt(alpha) * (x - ((1-alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)\
                    + torch.sqrt(beta) * noise
        
        model.train()
        
        # clip output to valid range
        # + 1 and /2 to bring to 0-1
        # *255 to bring to pixel range
        x = ((x.clamp(-1,1) + 1 )/2) 
        x = x * 255
        x = x.type(torch.uint8)
        
        return x
                
                

