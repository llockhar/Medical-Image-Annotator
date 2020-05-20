import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import os
from os.path import exists, join
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.transform import resize
import pandas as pd 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--xml_path", type=str, help="Path to XML annotation files")
parser.add_argument("--mask_path", type=str, help="Path to where output masks should be saved")
parser.add_argument("--resize_dim", type=int, default=None, help="Image size height, width for resizing masks")
parser.add_argument("--sigma", type=float, default=0.0, help="Sigma of Gaussian to apply to each landmark")
parser.add_argument("--overlay", type=bool, default=False, help="Whether to save a copy of images overlaid with masks")
parser.add_argument("--img_path", type=str, default="", help="Path to image files")
parser.add_argument("--overlay_path", type=str, default="", help="Path to where overlaid images should be saved")
args = parser.parse_args()

XML_PATH = args.xml_path
MASK_PATH = args.mask_path
RESIZE_DIM = args.resize_dim
SIGMA = args.sigma
OVERLAY = args.overlay
IMG_PATH = args.img_path
OVERLAY_PATH = args.overlay_path

if not exists(MASK_PATH):
    os.makedirs(MASK_PATH, exist_ok=True)
if len(OVERLAY_PATH) > 0 and not exists(OVERLAY_PATH):
    os.makedirs(OVERLAY_PATH, exist_ok=True)


def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    height = int(root.find('size')[1].text)
    width = int(root.find('size')[0].text)

    centroid_list = []
    for centroids in root.iter('object'):
        cx, cy = None, None
        for centroid in centroids.findall("centroid"):
            cx = int(centroid.find("cx").text)
            cy = int(centroid.find("cy").text)
        centroid_list.append([cx, cy])

    return height, width, centroid_list


def xmltomask(xmlname, outname):
    height, width, centroids = read_content(xmlname)
    mask = np.zeros((height, width, len(centroids)))
    for j,centroid in enumerate(centroids):
        mask[centroid[1], centroid[0], j] = 1

    mask_size = [RESIZE_DIM, RESIZE_DIM] if RESIZE_DIM is not None else [height, width]
    new_mask = np.zeros((mask_size[0], mask_size[1], len(centroids)))
    for j in range(len(centroids)):
        new_mask[:,:,j] = resize(mask[:,:,j], mask_size)
        if SIGMA > 0:
            new_mask[:,:,j] = gaussian((new_mask[:,:,j]), sigma=SIGMA, mode='constant')
        new_mask[:,:,j] /= new_mask[:,:,j].max()

    mask = new_mask.max(axis=2)
    mask = (mask*255).astype('uint8')
    imsave(outname, mask, check_contrast=False)
    
    return mask, len(centroids)


def overlayLabels(img, mask, outname):
    masked = np.ma.masked_where(mask<1, mask)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(1,1)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, aspect='auto', cmap='gray')
    ax.imshow(masked, vmin=1, vmax=255, aspect='auto', cmap='jet', alpha=0.7)
    fig.savefig(outname, dpi=img.shape[0])
    plt.close()


folder_tracker = []
filename_tracker = []
cell_count = []
for i,folder in enumerate(os.listdir(XML_PATH)):
    print('Processing folder {0} of {1}: '.format(i+1, len(os.listdir(XML_PATH))), folder)
    if not exists(join(MASK_PATH,folder)):
        os.mkdir(join(MASK_PATH,folder))
    if len(OVERLAY_PATH) > 0 and not exists(join(OVERLAY_PATH,folder)):
        os.mkdir(join(OVERLAY_PATH,folder))
    
    for filename in os.listdir(join(XML_PATH,folder)):
        if filename[-3:] == "xml":
            xml_name = join(XML_PATH,folder,filename)
            base_name = filename[:-4]

            out_name = join(MASK_PATH,folder,base_name + ".png")
            mask, centroid_count = xmltomask(xml_name, out_name)

            if OVERLAY:
                img = imread(join(IMG_PATH,folder,base_name + ".png"))
                img = resize(img, (RESIZE_DIM, RESIZE_DIM)) if RESIZE_DIM is not None else img
                out_name_overlay = join(OVERLAY_PATH,folder,base_name + ".png")
                overlayLabels(img, mask, out_name_overlay)

            folder_tracker.append(folder)
            filename_tracker.append(base_name + ".png")
            cell_count.append(centroid_count)

count_df = pd.DataFrame.from_dict(
    {'Folder':folder_tracker, 'Filename':filename_tracker, 'NumCentroids':cell_count})
count_df.to_excel(join(MASK_PATH, 'NumCentoids.xlsx'))