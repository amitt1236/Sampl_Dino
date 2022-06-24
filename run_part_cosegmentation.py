import matplotlib.pyplot as plt
import torch
from part_cosegmentation import find_part_cosegmentation, draw_part_cosegmentation

# Enter pretrained model
######################################################################
# model_type: type of model to extract descriptors from. Choose from [dino_vits8 | dino_vits16 | dino_vitb8 |
# dino_vitb16 | vit_small_patch8_224 | vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
model_type = 'dino_vits8'

# pretrained model
model_dir = "/home/projects/yonina/arielkes/amitaf/Sampl_Dino/dino_deitsmall8_pretrain_full_checkpoint.pth"

# head: chose between teacher and student networks
head = "student"
######################################################################

# Choose image paths:
images_paths = ['/home/projects/yonina/arielkes/amitaf/Sampl_Dino/images/cat.jpg',
                '/home/projects/yonina/arielkes/amitaf/Sampl_Dino/images/ibex.jpg']  # @param

# load_size: size of the smaller edge of loaded images. If None, does not resize.
load_size = 360

#  layer: layer to extract descriptors from.
layer = 11

#  facet: facet to extract descriptors from. options: ['key' | 'query' | 'value' | 'token']
facet = 'key'

#  bin: if True use a log-binning descriptor.
bin = False

#  threshold of saliency maps to distinguish fg and bg.
thresh = 0.065

#  stride of first convolution layer. small stride -> higher resolution.
stride = 4

#  Elbow coefficient for setting number of clusters.
elbow = 0.975

#  percentage of votes needed for a cluster to be considered salient
votes_percentage = 75

#  sample every ith descriptor for training clustering.
sample_interval = 100

#  Use low resolution saliency maps -- reduces RAM usage.
low_res_saliency_maps = True

#  number of final object parts.
num_parts = 4

#  number of crop augmentations to apply on each input image. relevant for small sets.
num_crop_augmentations = 20

#  If true, use three clustering stages instead of two. relevant for small sets.
three_stages = True

#  elbow method for finding amount of clusters when using three clustering stages.
elbow_second_stage = 0.94

with torch.no_grad():
    # computing part cosegmentation
    parts_imgs, pil_images = find_part_cosegmentation(images_paths, elbow, load_size, layer, facet, bin, thresh,
                                                      model_type, stride, votes_percentage, sample_interval,
                                                      low_res_saliency_maps, num_parts, num_crop_augmentations,
                                                      three_stages, elbow_second_stage, head=head, model_dir=model_dir)

    figs, axes = [], []
    for pil_image in pil_images:
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(pil_image)
        figs.append(fig)
        axes.append(ax)

    # displaying part segmentations
    part_figs = draw_part_cosegmentation(num_parts, parts_imgs, pil_images)

plt.show()
