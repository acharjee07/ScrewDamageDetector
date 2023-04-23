import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import torch
import torch.nn.functional as F
from typing import List
from tqdm import tqdm


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


def ensemble_models(models: List[torch.nn.Module], weights_path: List[str], test_loader: torch.utils.data.DataLoader, threshold: float) -> List[int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load the models' weights
    for i, model in enumerate(models):
        model=load_model(model, weights_path[i])
        model.eval()
        model.to(device)

    # make predictions on the test dataset using each model
    predictions = []
    img_paths=[]
    ground_truth=[]
    with torch.no_grad():
        for data,gt,path in tqdm(test_loader):
            data = data.to(device)
            outputs = []
            for model in models:
                output = model(data)
                outputs.append(output)
            ensemble_output = sum(outputs) / len(outputs)
            ensemble_output = torch.softmax(ensemble_output, dim=1)
            img_paths.extend(path)
            predictions.extend(ensemble_output.cpu().numpy().tolist())
            ground_truth.extend(gt)

    # apply the threshold to the predictions to get the final binary predictions
    binary_predictions = [1 if p[1] >= threshold else 0 for p in predictions]
    ground_truth=[x.argmax().item() if isinstance(x, torch.Tensor) else x for x in ground_truth]
    
    return [[pred,path,gt] for pred,path,gt in zip(binary_predictions,img_paths,ground_truth)]



def load_model(model,path):
    state_dict = torch.load(path)['state_dict']

    # create a new state dict with modified keys
    new_state_dict = {}
    for key in state_dict.keys():
        if key.startswith('backbone.backbone.'):
            new_key = key.replace('backbone.backbone.', 'backbone.')
        else:
            new_key = key
        new_state_dict[new_key] = state_dict[key]

    # use the new state dict for your model
#     print(new_state_dict)
    model.load_state_dict(new_state_dict)
    return model