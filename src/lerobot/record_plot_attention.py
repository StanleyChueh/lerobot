import torch
import time
import matplotlib.pyplot as plt
import math

plt.ion()  # interactive mode

fig, ax = plt.subplots()
img = None

while True:
    try:
        attn = torch.load("/tmp/smolvla_attn.pt")
        attn = attn.mean(0)  # avg heads

        # ===== adjust these =====
        TXT_TOKEN_IDX = 5
        IMG_START = 0
        IMG_LEN = attn.shape[-1]
        # =======================

        heat = attn[TXT_TOKEN_IDX, IMG_START:IMG_START + IMG_LEN]

        # auto infer grid
        n = heat.numel()
        h = int(math.sqrt(n))
        while n % h != 0:
            h -= 1
        w = n // h
        heatmap = heat.reshape(h, w)

        if img is None:
            img = ax.imshow(heatmap, cmap="jet")
            plt.colorbar(img)
        else:
            img.set_data(heatmap)
            img.set_clim(vmin=heatmap.min(), vmax=heatmap.max())

        ax.set_title("Text → Image Attention (Live)")
        plt.pause(0.05)

    except FileNotFoundError:
        time.sleep(0.1)
