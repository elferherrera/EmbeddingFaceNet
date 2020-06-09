"""
Face detection function
It loads the networks used for face detections and returns
the bounding boxes for the detected faces
"""

from typing import List

import numpy as np
#from typing import Tuple, List

import torch

# Importing Neural networks model
from embedding_facenet.models.inception_resnet_v1 import InceptionResnetV1

# Selecting the available device for calculations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading partial MTCNN networks
# Seting the networks to eval mode and not training
# This avoids auto grad from being calculated
RESNET = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


def create_embedding(imgs) -> List[float]:
    """
    Face embeddings
    Given an image or an array of images with the same size
    the bounding boxes for the detected faces are returned

    Important: All images have to have their color channer as RGB

    inputs:
        imgs: image or array of images

    output:
        embedding: Array with embeddings

    Note: The output for the boxes will allways be a list that corresponds to
    the number of images used as input. Meaning that if just one image is used,
    the size of the output will be (1, 512).

    """

    if isinstance(imgs, (np.ndarray, torch.Tensor)):
        imgs = torch.as_tensor(imgs, device=DEVICE)

        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)

    else:
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]

        if any(img.size != imgs[0].size for img in imgs):
            raise Exception("MTCNN batch processing only compatible with equal-dimension images.")

        imgs = np.stack([np.uint8(img) for img in imgs])

    # Moving all images to a torch tensor and to the available device
    imgs = torch.as_tensor(imgs, device=DEVICE)

    # Image standarization
    imgs = image_standarization(imgs)

    # Changing dimensions of images if they are as a batch
    # In pytorch the number of images has to be the first
    # dimension
    model_dtype = next(RESNET.parameters()).dtype
    imgs = imgs.permute(0, 3, 1, 2).type(model_dtype)

    with torch.no_grad():
        embeddings = RESNET(imgs)

    return embeddings.tolist()


def image_standarization(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Standarizing values in tensor to calculate embeddings
    """
    return (image_tensor - 127.5) / 128.0
