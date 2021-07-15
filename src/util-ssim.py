import ssim
import torch
from torch.autograd import Variable
import time

t_start = time.time()
img1 = Variable(torch.rand(1, 1, 256, 256))
img2 = Variable(torch.rand(1, 1, 256, 256))
t_end = time.time()
print(f"init random images took {t_end - t_start}")

t_start = time.time()
if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()
t_end = time.time()
print(f"moving to GPU took {t_end - t_start}")

t_start = time.time()
print(ssim.ssim(img1, img2))
t_end = time.time()
print(f"took {t_end - t_start}")


# this version seems much faster
t_start = time.time()
ssim_loss = ssim.SSIM(window_size = 11)
print(ssim_loss(img1, img2))
t_end = time.time()
print(f"took {t_end - t_start}")