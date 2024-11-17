from scipy import linalg
import numpy as np
from scipy import signal
import time

def soft_thresh(x, l = 0.5):
    return np.maximum(x - l, 0.)

def fista(A, b, x0, l, maxit):
    x = np.zeros(A.shape[1])
    pobj = []
    cpu_excution = []
    t = 0
    z = x.copy()
    L2_error = []
    L = linalg.norm(A) ** 2
    time1 = 0
    factor = linalg.norm(b)
    for ite in range(maxit):
        time0 = time.time()
        xold = x.copy()
        z = z + A.T.dot(b - A.dot(z)) / L
        x = soft_thresh(z, l / L)
        t0 = t
        t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)
        time1 += time.time() - time0
        cpu_excution.append(time1)       
        # this_pobj = 0.5 * linalg.norm(A.dot(x) - b) ** 2 + l * linalg.norm(x, 1)
        # L2_error.append((0.5 * linalg.norm(A.dot(x) - b) ** 2 + l * linalg.norm(x, 1))/factor)
        L2_error.append(10 * np.log10(linalg.norm(x - x0)/linalg.norm(x0)))
        # L2_error.append(linalg.norm(A.dot(x) - b)/linalg.norm(b))
        # pobj.append(this_pobj)
    times, pobj = map(np.array, [cpu_excution,pobj])
    return x, L2_error, times

def calculate_nmse(y_true, y_pred):
    # 计算 MSE（Mean Squared Error）
    mse = np.mean((y_true - y_pred) ** 2)
    # 计算真实值的方差
    variance = np.var(y_true)
    # 计算 NMSE
    nmse = mse / variance
    return nmse


# 计算能量函数
def energy_function(y_true, A, b_np, l):
    return (0.5 * linalg.norm(A.dot(y_true) - b_np) ** 2 + l * linalg.norm(y_true, 1))/linalg.norm(b_np)


def calculate_ssim(image1, image2, k1=0.01, k2=0.03, sigma=1.5, L=255):
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    mu1 = gaussian_blur(image1, sigma)
    mu2 = gaussian_blur(image2, sigma)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = gaussian_blur(image1 ** 2, sigma) - mu1_sq
    sigma2_sq = gaussian_blur(image2 ** 2, sigma) - mu2_sq
    sigma12 = gaussian_blur(image1 * image2, sigma) - mu12

    numerator = (2 * mu12 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / denominator

    return np.mean(ssim_map)

def gaussian_blur(image, sigma):
    kernel = create_gaussian_kernel(sigma)
    blurred_image = np.zeros_like(image, dtype=np.float64)
    
    for c in range(image.shape[2]):
        blurred_image[:, :, c] = signal.convolve2d(image[:, :, c], kernel, mode='same', boundary='symm')
    return blurred_image

def create_gaussian_kernel(sigma):
    size = int(2 * np.ceil(3 * sigma) + 1)
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - (size // 2)) ** 2 + (y - (size // 2)) ** 2) / (2 * sigma ** 2)), (size, size))
    kernel = kernel / np.sum(kernel)
    return kernel

def calculate_nmse(a_hat, a):
    squared_error = np.linalg.norm(a_hat - a)**2
    norm_a_hat_squared = np.linalg.norm(a_hat)**2
    nmse = 10 * np.log10(squared_error / norm_a_hat_squared)
    return nmse

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)  # 计算均方误差（MSE）
    max_pixel = np.max(img1)  # 图像像素的最大值
    psnr = 10 * np.log10((max_pixel ** 2) / mse)  # 计算PSNR值
    return psnr

# 时间装饰器
def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行要测量运行时间的代码块
        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time  # 计算运行时间
        print("Execution Time: {:.6f} seconds".format(execution_time))
        return result
    return wrapper

@measure_execution_time
def measure_runner_time(runner, total_period):
    runner.run(total_period)