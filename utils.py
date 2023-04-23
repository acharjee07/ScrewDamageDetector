import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math


def visual(df,n):
    sample_df=df.sample(n)
    image_paths = sample_df['image_path'].values
    labels = sample_df['label'].values

    n = int(np.sqrt(len(sample_df)) ) # Set the number of images per row and column in the grid
    fig, ax = plt.subplots(nrows=n, ncols=n, figsize=(12, 12))  # Create the plot

    # Iterate through the image file paths and their labels, and plot each image with its label
    for i, (image_path, label) in enumerate(zip(image_paths, labels)):
        row, col = divmod(i, n)  # Calculate the row and column index for this image
        img = cv2.imread(image_path)  # Read the image file as an RGB array
        # Plot the image in the appropriate position in the grid, with its label as the title
        ax[row, col].imshow(img)
        if image_path.split('/')[-1][0:2]=='ok' or image_path.split('/')[-2]=='test':
            ax[row, col].set_title(label)
        else :
            ax[row, col].set_title(image_path.split('/')[-1])
        ax[row, col].axis('off')  # Hide the axis

    plt.show()  # Display the plot



def visualize_dataset(dataset, n):
    fig, axs = plt.subplots(n, n, figsize=(10,10))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            image, label = dataset[idx]
            axs[i,j].imshow(image.permute(1,2,0))
            axs[i,j].set_title(label.argmax().item())
            axs[i,j].axis('off')
    plt.tight_layout()
    plt.show()