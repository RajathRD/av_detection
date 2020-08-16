#!/usr/bin/env python
# coding: utf-8

# #Waymo Open Dataset Tutorial
# 
# - Website: https://waymo.com/open
# - GitHub: https://github.com/waymo-research/waymo-open-dataset
# 
# This tutorial demonstrates how to use the Waymo Open Dataset with two frames of data. Visit the [Waymo Open Dataset Website](https://waymo.com/open) to download the full dataset.
# 
# To use, open this notebook in [Colab](https://colab.research.google.com).
# 
# Uncheck the box "Reset all runtimes before running" if you run this colab directly from the remote kernel. Alternatively, you can make a copy before trying to run it by following "File > Save copy in Drive ...".
# 
# 

# In[8]:


import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import utils
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from glob import glob
tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from engine import train_one_epoch, evaluate


# In[9]:


import torch
import torch.utils.data as data


# In[10]:


DATA_FOLDER = "../data/training_0000"

print (torch.cuda.is_available())
print (torch.cuda.device_count())
print (torch.cuda.get_device_name())
device = torch.device("cuda")


# ## Setting up Data

# In[11]:


import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# In[12]:


class WaymoDataset(data.Dataset):
    def __init__(self, root, transforms=None):
        self.label_map = {0:0, 1: 1, 2:2, 4:3}
        self.root = root
        self.transforms = transforms
        self.data_files = list(sorted(os.listdir(root)))
        dataset = [tf.data.TFRecordDataset(os.path.join(root, FILENAME), compression_type='') for FILENAME in self.data_files[:1]]
        
        self.frames = []
        for data_file in dataset:
            for idx, data in enumerate(data_file):
                if idx % 5 != 0:
                    continue
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                self.frames.append(frame)
        
#         self.images = [frame.images[0].image for frame in self.frames]
        self.targets = []
        for i, frame in enumerate(self.frames):
            
            target = {}
            target_bbox = []
            target_labels = []
            target_areas = []
        
            for camera_labels in frame.camera_labels:
                
                if camera_labels.name != 1:
                    continue
                    
                for label in camera_labels.labels:                    
                    xmin= label.box.center_x - 0.5 * label.box.length
                    ymin = label.box.center_y - 0.5 * label.box.width
                    xmax = xmin + label.box.length
                    ymax = ymin + label.box.width
                    area = label.box.length * label.box.width
                    target_bbox.append([xmin, ymin, xmax, ymax])
                    target_labels.append(self.label_map[label.type])
                    target_areas.append(area)
                    
            target['boxes'] = torch.as_tensor(target_bbox, dtype=torch.float32)
#             target["labels"] = torch.as_tensor(np.eye(3)[np.array(target_labels)], dtype=torch.int64)
            target['labels'] = torch.as_tensor(np.array(target_labels), dtype=torch.int64)
#             target['image_id'] = frame.context.name + "_" + str(frame.timestamp_micros)
            target['image_id'] = torch.tensor([int(frame.context.name.split("_")[-2] + str(i))])
            
            target["area"] = torch.as_tensor(target_areas, dtype=torch.float32)
            target["iscrowd"] = torch.zeros((len(target['boxes'])), dtype=torch.int64)
            self.targets.append(target)
#             print (i, set([i for i in target_labels]))
        
    def __getitem__(self, index):

        img = Image.fromarray(tf.image.decode_jpeg(self.frames[index].images[0].image).numpy())
        target = self.targets[index]
#         image_path = os.path.join(IMAGE_FOLDER, target['image_id'])
#         img = Image.open(image_path).convert('RGB')
#         img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.frames)


# In[3]:


# dataset = WaymoDataset('../data/training_0000/', transforms=get_transform(train=True))

# data_loader = torch.utils.data.DataLoader(dataset,\
#                                           batch_size=2,\
#                                           shuffle=True,\
#                                           num_workers=4,\
#                                           collate_fn=utils.collate_fn)


# In[4]:


# dataset.targets[0]['image_id']


# In[13]:


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


# In[ ]:


# images, targets = next(iter(data_loader))
# images = list([image.to(device) for image in images])
# # for k, v in t.items():
# #     print (k)
# #     if k != "image_id":
# #         d = {k: v.to(device)}
# targets = [{k: v.to(device) for k, v in t.items() if k != "image_id"} for t in targets]


# In[ ]:


# model.to(device)
# output = model(images, targets)


# In[ ]:


# output


# In[ ]:


# dataset.frames[0].context.name.split("_")[-2]


# In[14]:


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
# our dataset has two classes only - background and person
num_classes = 4
# use our dataset and defined transformations
dataset = WaymoDataset('../data/training_0000', get_transform(train=True))
dataset_test = WaymoDataset('../data/training_0000', get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-5])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-5:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)


# In[15]:



# get the model using our helper function
model = get_model(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10


# In[16]:


for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

print("That's it!")


# # Inference

# In[ ]:


# INFERENCE_FILE = "segment-10455472356147194054_1560_000_1580_000_with_camera_labels.tfrecord"


# # In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')

# def show_image(image, target_bbox, pred_bbox):
#     """Show a camera image and the given camera labels."""
        
#     fig, ax = plt.subplots(1, 2, figsize=(20, 15))
#     for patch in target_bbox:
#         ax[0].add_patch(Rectangle(
#         xy=(patch[0], patch[1]),
#         width=patch[2] - patch[0],
#         height=patch[3] - patch[1],
#         linewidth=1,
#         edgecolor="green",
#         facecolor='none'))
#     ax[0].imshow(image)
# #     if title:
#     ax[0].title.set_text("Ground Truth")
#     ax[0].grid(False)
#     ax[0].axis('off')
 
#     for patch in pred_bbox:
#         ax[1].add_patch(Rectangle(
#         xy=(patch[0], patch[1]),
#         width=patch[2] - patch[0],
#         height=patch[3] - patch[1],
#         linewidth=1,
#         edgecolor="red",
#         facecolor='none'))
#     ax[1].imshow(image)
    
#     ax[1].title.set_text("Prediction")
#     ax[1].grid(False)
#     ax[1].axis('off')
#     plt.show()


# # In[ ]:


# label_map = {0:0, 1: 1, 2:2, 4:3}
# score_threshold = 0.4

# data_file = tf.data.TFRecordDataset(os.path.join(DATA_FOLDER, INFERENCE_FILE), compression_type='')

# frames = []

# for idx, data in enumerate(data_file):
#     frame = open_dataset.Frame()
#     frame.ParseFromString(bytearray(data.numpy()))
#     frames.append(frame)


# # In[ ]:


# from matplotlib.pyplot import figure


# # In[ ]:



# for i, frame in enumerate(frames):
#     if i == 81:
#         break
#     image = tf.image.decode_jpeg(frame.images[0].image).numpy()
#     if i%40 != 0:
#         continue
#     target = {}
#     target_bbox = []
#     target_labels = []
#     target_areas = []
    
#     for camera_labels in frame.camera_labels:

#         if camera_labels.name != 1:
#             continue

#         for label in camera_labels.labels:                    
#             xmin= label.box.center_x - 0.5 * label.box.length
#             ymin = label.box.center_y - 0.5 * label.box.width
#             xmax = xmin + label.box.length
#             ymax = ymin + label.box.width
#             area = label.box.length * label.box.width
#             target_bbox.append([xmin, ymin, xmax, ymax])
#             target_labels.append(label_map[label.type])
#             target_areas.append(area)       
    
# #     show_image(image, target_bbox, [1, 2, 1], title="Ground Truth")
#     img = Image.fromarray(image)
             
#     target['boxes'] = torch.as_tensor(target_bbox, dtype=torch.float32)
#     target['labels'] = torch.as_tensor(np.array(target_labels), dtype=torch.int64)
#     target['image_id'] = torch.tensor([int(frame.context.name.split("_")[-2] + str(i))])
#     target["area"] = torch.as_tensor(target_areas, dtype=torch.float32)
#     target["iscrowd"] = torch.zeros((len(target['boxes'])), dtype=torch.int64)

#     img, target = get_transform(train=False)(img, target)
    
#     output = model([img], [target])
    
#     pred_bbox = [x.data.numpy() for idx, x in enumerate(output[0]['boxes']) if output[0]["scores"][idx] > score_threshold]
#     show_image(image, target_bbox, pred_bbox)


# #     break


# # In[ ]:




