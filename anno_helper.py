import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.morphology import dilation, disk
from skimage.transform import resize

PATCH = 256

def overlayLabels(Image1, Image2_mask, outname):

    plt.figure()
    plt.imshow(Image1, cmap='gray')
    plt.imshow(Image2_mask, cmap='jet', alpha=0.5) 
    axes = plt.gca()
    axes.axis('off')
    plt.savefig(outname, bbox_inches='tight')
    plt.close()

def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    height = int(root.find('size')[1].text)
    width = int(root.find('size')[0].text)

    list_with_all_centroids = []

    for centroids in root.iter('object'):

        filename = root.find('filename').text

        cx, cy = None, None

        for centroid in centroids.findall("centroid"):
            cx = int(centroid.find("cx").text)
            cy = int(centroid.find("cy").text)

        list_with_single_centroids = [cx, cy]
        list_with_all_centroids.append(list_with_single_centroids)

    return filename, height, width, list_with_all_centroids

def xmltomask(xmlname, outname):

    # base_file = filename[:-4]
    filename, height, width, centroids = read_content(xmlname)

    mask = np.zeros((height, width, len(centroids)))
    for j,centroid in enumerate(centroids):
        mask[centroid[1], centroid[0], j] = 1

    # mask = (dilation((mask*1), disk(5))*1)
    # mask = gaussian((mask*1), sigma=2)
    new_mask = np.zeros((PATCH, PATCH, len(centroids)))
    for j in range(len(centroids)):
        new_mask[:,:,j] = resize(mask[:,:,j],(PATCH,PATCH), preserve_range=True)
        # mask[:,:,j] /= mask[:,:,j].max()
        if mask[:,:,j].sum() != 1 or mask[:,:,j].max() != 1:
            print(filename)
        new_mask[:,:,j] = gaussian((new_mask[:,:,j]), sigma=1.5, mode='constant')
        new_mask[:,:,j] /= new_mask[:,:,j].max()
    mask = new_mask.max(axis=2)
    mask *= 255
    mask = mask.astype(np.uint8)
    # print(np.unique(mask))
    # plt.imshow(mask)
    # plt.show()
    # plt.imshow(mask)
    # plt.show()

    imsave(outname, mask, check_contrast=False)

def crop_flow(coords_file,in_dir,folder,file_name,out_dir):

    flow_img = imread(os.path.join(in_dir,folder,file_name))
    
    crop_top = coords_file['Top'].values[0]-1
    crop_left = coords_file['Left'].values[0]-1
    crop_height = coords_file['Height'].values[0] + 1
    crop_width = coords_file['Width'].values[0] + 1

    flow_crop = flow_img[crop_top:crop_top+crop_height,crop_left:crop_left+crop_width]

    imsave(os.path.join(out_dir,folder,file_name),flow_crop,check_contrast=False)

    return 

def shift_xml(coords_file,folder,file_name,out_dir):
    crop_top = coords_file['Top'].values[0] - 1
    crop_left = coords_file['Left'].values[0] - 1
    crop_height = coords_file['Height'].values[0] + 1
    crop_width = coords_file['Width'].values[0] + 1
    filename, height, width, centroids = \
        read_content(os.path.join(in_dir,folder,file_name[:-3] + 'xml'))

    imageShape = [crop_height, crop_width, 1]
    writer = PascalVocWriter(folder, filename, imageShape)

    for i,centroid in enumerate(centroids):
        centroid[1] -= crop_top
        centroid[0] -= crop_left
        # centroid[1] += 71
        writer.addCentroid(centroid[0], centroid[1], str(i+1))

    outname = os.path.join(out_dir,folder,file_name)
    writer.save(targetFile=outname)

if __name__ == '__main__':

    img_path = "C:/Users/llockhar/Desktop/Images4cellCropped2019"
    xml_path = 'C:/Users/llockhar/Desktop/CroppedXML'
    mask_path = 'C:/Users/llockhar/Desktop/Gauss256Sigma1_5'


    if not os.path.exists(mask_path):
        os.mkdir(mask_path)
        
    for folder in os.listdir(img_path):
        print(folder)
        if not os.path.exists(os.path.join(mask_path,folder)):
            os.mkdir(os.path.join(mask_path,folder))
        
        for filename in os.listdir(os.path.join(img_path,folder)):
            if filename[-3:] == "png":
                file_name = os.path.join(img_path,folder,filename)
                # mask_name = os.path.join(mask_path,folder,filename[-10:-4] + ".png")
                # img = imread(file_name)
                # mask = imread(mask_name)

                out_name = os.path.join(mask_path,folder,filename[:-4] + ".png")
                # overlayLabels(img, mask, outname)
                xml_name = os.path.join(xml_path,folder,filename[:-4] + ".xml")
                # if i == 9:
                #     print(filename,xml_name)
                xmltomask(xml_name,out_name)
