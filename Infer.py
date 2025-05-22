import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import UNet
from utils import *

# python test_AWGN2.py --logdir logs/logs/R2R_25_3 --test_data test --test_noiseL 25 --alpha 1.5 --training R2R
# python test_AWGN2.py --logdir logs/logs/R2R_15 --test_data test --test_noiseL 15 --alpha 1.5 --training R2R
# python test_AWGN2.py --logdir logs/logs/N2C_25 --test_data test --test_noiseL 0 --alpha 1.5 --training N2C

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="UNet")
parser.add_argument("--logdir", type=str, default="logs/logs/R2R_50", help='path of log files')
parser.add_argument("--test_data", type=str, default='test', help='test dataset')
parser.add_argument("--test_noiseL", type=float, default=50, help='noise level used on test set')
parser.add_argument("--alpha", type=float, default=1.5, help='R2R recorruption parameter')
parser.add_argument("--training", type=str, default="R2R", help='R2R or N2C')
opt = parser.parse_args()


def generate_low_freq_noise_torch(size):
    # Generate low-frequency noise directly on GPU.
    freq_noise = torch.randn(size, dtype=torch.float32, device='cuda')

    # Low-pass filter to retain only low frequencies
    X, Y = torch.meshgrid(torch.arange(size[0], device='cuda'), torch.arange(size[1], device='cuda'))
    centerX, centerY = size[0] // 2, size[1] // 2
    radius = min(centerX, centerY) / 8
    mask = torch.sqrt((X - centerX) ** 2 + (Y - centerY) ** 2) <= radius

    # Apply low-pass filter
    freq_noise = torch.fft.fftshift(torch.fft.fft2(freq_noise))
    freq_noise[~mask] = 0
    freq_noise = torch.fft.ifft2(torch.fft.ifftshift(freq_noise))
    low_freq_noise = torch.real(freq_noise)

    return low_freq_noise

def high_pass_filter(img, cutoff_frequency=4):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2

    mask = np.ones((rows, cols, 2), np.uint8)
    r = cutoff_frequency
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back

def normalize(data):
    return data / 255.


def main():
    # Build model
    print('Loading model ...\n')
    net = UNet()
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()

    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    if len(files_source) == 0:
        print(f"No images found in directory: {os.path.join('data', opt.test_data)}")
        return

    # process data
    psnr_test = 0
    for f in files_source:
        # image
        Img = cv2.imread(f)
        if Img is None:
            print(f"Error loading image: {f}")
            continue
        Img = cv2.resize(Img, (512, 512))
        Img = normalize(np.float32(Img[:, :, 0]))

        Img = high_pass_filter(Img)

        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.tensor(Img, dtype=torch.float32).cuda()
        noise = opt.test_noiseL / 255. * generate_low_freq_noise_torch(ISource.size()).cuda()
        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())

        if opt.training == 'R2R':
            alpha = opt.alpha
            out_test = None
            aver_num = 50
            eps = opt.test_noiseL / 255.

            for test_j in range(aver_num):
                INoisy_pert = INoisy + alpha * eps * generate_low_freq_noise_torch(ISource.size()).cuda()
                with torch.no_grad():
                    out_test_single = model(INoisy_pert)
                if out_test is None:
                    out_test = out_test_single.detach()
                else:
                    out_test += out_test_single.detach()
                del out_test_single

            out_test = torch.clamp(out_test / aver_num, 0., 1.)
            psnr = batch_PSNR(out_test, ISource, 1.)
            psnr_test += psnr

            out_test_np = out_test.cpu().numpy().squeeze()

        else:
            with torch.no_grad():  # this can save much memory
                Out = torch.clamp(model(INoisy), 0., 1.)
            psnr = batch_PSNR(Out, ISource, 1.)
            psnr_test += psnr

            out_test_np = Out.cpu().numpy().squeeze()

        print("%s PSNR %f" % (f, psnr))

        # Save images
        denoised_image_path = os.path.join('output', 'denoised', os.path.basename(f))
        os.makedirs(os.path.dirname(denoised_image_path), exist_ok=True)
        cv2.imwrite(denoised_image_path, (out_test_np * 255).astype(np.uint8))

    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)


if __name__ == "__main__":
    main()
