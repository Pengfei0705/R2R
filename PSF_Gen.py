import numpy as np
import matplotlib.pyplot as plt

# 参数设置
FWHM = 10 # 半高宽
a = 1    # 高斯函数幅值
mu = 0   # 高斯函数的中心坐标----期望
sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))

# 生成数据
x = np.linspace(-256, 256, 512)
y = np.linspace(-256, 256, 512)
x, y = np.meshgrid(x, y)
z = a * np.exp(-((x - mu)**2 + (y - mu)**2) / (2 * sigma**2))

# 显示图像
plt.imshow(z, extent=(-256, 256, -256, 256), origin='lower')
plt.colorbar()
plt.title('2D Gaussian Distribution')
plt.show()
print(z.shape)