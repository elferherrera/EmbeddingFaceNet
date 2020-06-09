"""
Test suite for embedding calculations
"""

import torch
import embedding_facenet


def test_shape_one_image():
    """
    Test to check if dimensions when using one tensor are
    correct.
    """

    fake_face = torch.ones((160, 160, 3))

    embeddings = embedding_facenet.create_embedding(fake_face)

    embeddings_shape = torch.Tensor(embeddings).shape

    # There should be 1 faces with 512 points marking the
    # bounding box for the face
    assert embeddings_shape == (1, 512)


def test_shape_image_list():
    """
    Test to check if dimensions when using a list of tensors are
    correct.
    """

    fake_face = torch.ones((160, 160, 3))
    list_faces = [fake_face, fake_face, fake_face]

    embeddings = embedding_facenet.create_embedding(list_faces)

    embeddings_shape = torch.Tensor(embeddings).shape

    # There should be 1 faces with 512 points marking the
    # bounding box for the face
    assert embeddings_shape == (3, 512)


def test_shape_image_tensor():
    """
    Test to check if dimensions when using a tensor of tensors are
    correct.
    """

    fake_face = torch.ones((160, 160, 3))

    tensor_faces = torch.zeros((3, 160, 160, 3))
    tensor_faces[0, :, :, :] = fake_face
    tensor_faces[1, :, :, :] = fake_face
    tensor_faces[2, :, :, :] = fake_face

    embeddings = embedding_facenet.create_embedding(tensor_faces)

    embeddings_shape = torch.Tensor(embeddings).shape

    # There should be 1 faces with 512 points marking the
    # bounding box for the face
    assert embeddings_shape == (3, 512)
