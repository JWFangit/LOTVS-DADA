from __future__ import division
import numpy as np
import scipy.io
import scipy.ndimage
from scipy.misc import imread, imresize
import tensorflow as tf


def euler2mat(z, y, x):
    """Converts euler angles to rotation matrix
     TODO: remove the dimension for 'N' (deprecated for converting all source
           poses altogether)
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        z: rotation angle along z axis (in radians) -- size = [B, N]
        y: rotation angle along y axis (in radians) -- size = [B, N]
        x: rotation angle along x axis (in radians) -- size = [B, N]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
    """
    B = tf.shape(z)[0]
    N = 1
    z = tf.clip_by_value(z, -np.pi, np.pi)
    y = tf.clip_by_value(y, -np.pi, np.pi)
    x = tf.clip_by_value(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z = tf.expand_dims(tf.expand_dims(z, -1), -1)
    y = tf.expand_dims(tf.expand_dims(y, -1), -1)
    x = tf.expand_dims(tf.expand_dims(x, -1), -1)

    zeros = tf.zeros([B, N, 1, 1])
    ones = tf.ones([B, N, 1, 1])

    cosz = tf.cos(z)
    sinz = tf.sin(z)
    rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
    rotz_2 = tf.concat([sinz, cosz, zeros], axis=3)
    rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
    zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = tf.cos(y)
    siny = tf.sin(y)
    roty_1 = tf.concat([cosy, zeros, siny], axis=3)
    roty_2 = tf.concat([zeros, ones, zeros], axis=3)
    roty_3 = tf.concat([-siny, zeros, cosy], axis=3)
    ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

    cosx = tf.cos(x)
    sinx = tf.sin(x)
    rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
    rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
    rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
    xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

    rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
    return rotMat


def pose_vec2mat(vec):
    """Converts 6DoF parameters to transformation matrix
    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 4, 4]
    """
    batch_size, _ = vec.get_shape().as_list()
    translation = tf.slice(vec, [0, 0], [-1, 3])
    translation = tf.expand_dims(translation, -1)
    rx = tf.slice(vec, [0, 3], [-1, 1])
    ry = tf.slice(vec, [0, 4], [-1, 1])
    rz = tf.slice(vec, [0, 5], [-1, 1])
    rot_mat = euler2mat(rz, ry, rx)
    rot_mat = tf.squeeze(rot_mat, axis=[1])
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch_size, 1, 1])
    transform_mat = tf.concat([rot_mat, translation], axis=2)
    transform_mat = tf.concat([transform_mat, filler], axis=1)
    return transform_mat


def padding(img, shape_r=240, shape_c=320, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = imresize(img, (shape_r, new_cols))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:,
        ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols), ] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = imresize(img, (new_rows, shape_c))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def resize_fixation(img, rows=480, cols=640):
    out = np.zeros((rows, cols))
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0] * factor_scale_r))
        c = int(np.round(coord[1] * factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 1

    return out


def padding_fixation(img, shape_r=480, shape_c=640):
    img_padded = np.zeros((shape_r, shape_c))

    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = resize_fixation(img, rows=shape_r, cols=new_cols)
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:,
        ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols), ] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = resize_fixation(img, rows=new_rows, cols=shape_c)
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def preprocess_images(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 3))

    for i, path in enumerate(paths):

        original_image = imread(path)
        if original_image.ndim == 2:
            copy = np.zeros((original_image.shape[0], original_image.shape[1], 3))
            copy[:, :, 0] = original_image
            copy[:, :, 1] = original_image
            copy[:, :, 2] = original_image
            original_image = copy
        padded_image = padding(original_image, shape_r, shape_c, 3)
        ims[i] = padded_image

    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68
    ims = ims[:, :, :, ::-1]

    return ims


def preprocess_maps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1))

    for i, path in enumerate(paths):
        # original_map = cv2.imread(path, 0)
        # original_map = mpimg.imread(path)
        original_map = imread(path)
        padded_map = padding(original_map, shape_r, shape_c, 1)
        ims[i, :, :, 0] = padded_map.astype(np.float32)
        ims[i, :, :, 0] /= 255.0

    return ims


def preprocess_fixmaps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1))

    for i, path in enumerate(paths):
        fix_map = scipy.io.loadmat(path)["I"]
        ims[i, :, :, 0] = padding_fixation(fix_map, shape_r=shape_r, shape_c=shape_c)

    return ims


def postprocess_predictions(pred, shape_r, shape_c):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    pred = pred / np.max(pred) * 255

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        # pred = cv2.resize(pred, (new_cols, shape_r))
        pred = imresize(pred, (shape_r, new_cols))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        # pred = cv2.resize(pred, (shape_c, new_rows))
        pred = imresize(pred, (new_rows, shape_c))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    img = scipy.ndimage.filters.gaussian_filter(img, sigma=7)
    img = img / np.max(img) * 255

    return img
