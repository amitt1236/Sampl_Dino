from  extractor import ViTExtractor
from sklearn.manifold import TSNE
import plotly.express as px
import skimage.measure
import numpy as np
import torch

def t_sne(image_path, seg_path, image_size, model_type, model_dir, stride ,layer, facet, bin: bool = False, include_cls: bool = False):
    # extractor init
    extractor = ViTExtractor(model_type, stride, model_dir=model_dir, device=device)
    # image bach as tensors resized 
    image_batch, image_pil = extractor.preprocess(image_path, image_size)
    # descriptors shape Bx1xtxd' where d' is the dimension of the descriptors
    descriptor = extractor.extract_descriptors(image_batch.to(device), layer, facet, bin, include_cls=False).cpu().numpy()
    # to shape txd (single image only)
    descriptor = np.squeeze(descriptor,axis=(1,0))
    
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    tsne = TSNE(2, verbose=1, random_state=7)
    tsne_proj = tsne.fit_transform(descriptor)

    _,im = extractor.preprocess(seg_path, image_size)
    im = np.asarray(im)
    # to 8x8 patches
    im = skimage.measure.block_reduce(im, (8,8,1) , np.max)
    # flatten the patches
    im = np.reshape(im, (784,3))
    #pool rgb to one channel
    im = skimage.measure.block_reduce(im, (1,3) , np.mean)
    # add mask vlaues as 3rd column
    tsne_proj = np.hstack((tsne_proj, im))
    
    fig = px.scatter(
    tsne_proj, x=0, y=1,
    color=2
)
    fig.show()


if __name__ == '__main__':

    # Params
    #######################################################################
    #######################################################################

    image_path = "/Users/amitaflalo/Desktop/Sampl_Dino/img.png"
    seg_path = "/Users/amitaflalo/Desktop/Sampl_Dino/img2.png"

    # Resize image.
    image_size = (224,224)

    # Path of the image to load
    image_path = "./img.png"

    # Path where to save visualizations
    output_dir = '.'

    # pretrained model, ckpt dir
    pretrained_weights = "/Users/amitaflalo/Desktop/Sampl_Dino/dino_deitsmall8_pretrain_full_checkpoint.pth"

    # Vit architecture.  choices=['vit_tiny', 'vit_small', 'vit_base']
    arch = 'dino_vits8'

    # Key to use in the checkpoint (example, for dino: "teacher"/"student")
    checkpoint_key = "student"

    stride = 8

    layer = 11

    facet = 'key'

    #######################################################################
    #######################################################################

    device = 'cpu' if torch.backends.mps.is_available() else 'cpu'

    t_sne(image_path, seg_path, image_size, arch ,pretrained_weights, stride, 11, facet)
