import matplotlib.pyplot as plt

def plot_metrics(psnr_values, ssim_values):
    plt.figure()

    plt.plot(psnr_values, label="PSNR")
    plt.plot(ssim_values, label="SSIM")

    plt.xlabel("Image Index")
    plt.ylabel("Value")
    plt.title("Image Enhancement Performance")

    plt.legend()
    plt.show()