import numpy as np
import keras.backend as K


def kl_loss(y_true, y_pred, eps=K.epsilon()):
    """
    Kullback-Leiber divergence (sec 4.2.3 of [1]). Assumes shape (b, 1, h, w) for all tensors.

    :param y_true: groundtruth.
    :param y_pred: prediction.
    :param eps: regularization epsilon.
    :return: loss value (one symbolic value per batch element).
    """
    P = y_pred
    P = P / (K.epsilon() + K.sum(P, axis=[1, 2, 3], keepdims=True))
    Q = y_true
    Q = Q / (K.epsilon() + K.sum(Q, axis=[1, 2, 3], keepdims=True))

    kld = K.sum(Q * K.log(eps + Q / (eps + P)), axis=[1, 2, 3])

    return kld


def information_gain(y_true, y_pred, y_base, eps=K.epsilon()):
    """
    Information gain (sec 4.1.3 of [1]). Assumes shape (b, 1, h, w) for all tensors.

    :param y_true: groundtruth.
    :param y_pred: prediction.
    :param y_base: baseline.
    :param eps: regularization epsilon.
    :return: loss value (one symbolic value per batch element).
    """
    P = y_pred
    P = P / (K.epsilon() + K.max(P, axis=[1, 2, 3], keepdims=True))
    Q = y_true
    B = y_base

    Qb = K.round(Q)  # discretize at 0.5
    N = K.sum(Qb, axis=[1, 2, 3], keepdims=True)

    ig = K.sum(Qb * (K.log(eps + P) / K.log(2) - K.log(eps + B) / K.log(2)), axis=[1, 2, 3]) / (K.epsilon() + N)

    return ig


def nss_loss(y_true, y_pred):
    """
    Normalized Scanpath Saliency (sec 4.1.2 of [1]). Assumes shape (b, 1, h, w) for all tensors.

    :param y_true: groundtruth.
    :param y_pred: prediction.
    :return: loss value (one symbolic value per batch element).
    """
    P = y_pred
    P = P / (K.epsilon() + K.max(P, axis=[1, 2, 3], keepdims=True))
    Q = y_true

    Qb = K.round(Q)  # discretize at 0.5
    N = K.sum(Qb, axis=[1, 2, 3], keepdims=True)

    mu_P = K.mean(P, axis=[1, 2, 3], keepdims=True)
    std_P = K.std(P, axis=[1, 2, 3], keepdims=True)
    P_sign = (P - mu_P) / (K.epsilon() + std_P)

    nss = (P_sign * Qb) / (K.epsilon() + N)
    nss = K.sum(nss, axis=[1, 2, 3])

    return -nss  # maximize nss


def cc_loss(y_true, y_pred):
    eps = K.epsilon()
    P = y_pred
    P = P / (eps + K.sum(P, axis=[1, 2, 3], keepdims=True))
    Q = y_true
    Q = Q / (eps + K.sum(Q, axis=[1, 2, 3], keepdims=True))

    N = y_pred._shape_as_list()[1] * y_pred._shape_as_list()[2]

    E_pq = K.sum(Q * P, axis=[1, 2, 3], keepdims=True)
    E_q = K.sum(Q, axis=[1, 2, 3], keepdims=True)
    E_p = K.sum(P, axis=[1, 2, 3], keepdims=True)
    E_q2 = K.sum(Q ** 2, axis=[1, 2, 3], keepdims=True) + eps
    E_p2 = K.sum(P ** 2, axis=[1, 2, 3], keepdims=True) + eps

    num = E_pq - ((E_p * E_q) / N)
    den = K.sqrt((E_q2 - E_q ** 2 / N) * (E_p2 - E_p ** 2 / N))

    return K.sum(- (num + eps) / (den + eps), axis=[1, 2, 3])  # 相关系数。|cc|<=1, =0 则不相关 1 则正相关， -1 则表示负相关

# # ACL_full loss
#
# # KL-Divergence Loss
# def kl_divergence(y_true, y_pred):
#     # max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=1), axis=1)), shape_r_out, axis=1)), shape_c_out, axis=2)
#     max_y_pred = K.expand_dims(K.repeat_elements(
#         K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
#         y_pred.shape[3], axis=3))
#     y_pred /= max_y_pred
#
#     max_y_true = K.expand_dims(K.repeat_elements(
#         K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_true, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
#         y_pred.shape[3], axis=3))
#     y_bool = K.cast(K.greater(max_y_true, 0.1), 'float32')
#
#     sum_y_true = K.expand_dims(K.repeat_elements(
#         K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(y_true, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
#         y_pred.shape[3], axis=3))
#     sum_y_pred = K.expand_dims(K.repeat_elements(
#         K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
#         y_pred.shape[3], axis=3))
#     y_true /= (sum_y_true + K.epsilon())
#     y_pred /= (sum_y_pred + K.epsilon())
#     return 10 * K.sum(y_bool * y_true * K.log((y_true / (y_pred + K.epsilon())) + K.epsilon()))
#
#
# # Correlation Coefficient Loss
# def correlation_coefficient(y_true, y_pred):
#     max_y_pred = K.expand_dims(K.repeat_elements(
#         K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
#         y_pred.shape[3], axis=3))
#     y_pred /= max_y_pred
#
#     # max_y_true = K.expand_dims(K.repeat_elements(
#     #     K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_true, axis=[2, 3, 4])), shape_r_out, axis=2)),
#     #     shape_c_out, axis=3))
#     max_y_true = K.max(y_true, axis=[2, 3, 4])
#     y_bool = K.cast(K.greater(max_y_true, 0.1), 'float32')
#
#     sum_y_true = K.expand_dims(K.repeat_elements(
#         K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(y_true, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
#         y_pred.shape[3], axis=3))
#     sum_y_pred = K.expand_dims(K.repeat_elements(
#         K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
#         y_pred.shape[3], axis=3))
#
#     y_true /= (sum_y_true + K.epsilon())
#     y_pred /= (sum_y_pred + K.epsilon())
#
#     N = y_pred._shape_as_list()[2] * y_pred._shape_as_list()[3]
#     sum_prod = K.sum(y_true * y_pred, axis=[2, 3, 4])
#     sum_x = K.sum(y_true, axis=[2, 3, 4])
#     sum_y = K.sum(y_pred, axis=[2, 3, 4])
#     sum_x_square = K.sum(K.square(y_true), axis=[2, 3, 4])+ K.epsilon()
#     sum_y_square = K.sum(K.square(y_pred), axis=[2, 3, 4])+ K.epsilon()
#
#     num = sum_prod - ((sum_x * sum_y) / N)
#     den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))
#
#     return K.sum(y_bool*(-2 * num/den))#
#
#
# # Normalized Scanpath Saliency Loss
# def nss_loss(y_true, y_pred):
#     max_y_pred = K.expand_dims(K.repeat_elements(
#         K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
#         y_pred.shape[3], axis=3))
#     y_pred /= max_y_pred
#     # y_pred_flatten = K.batch_flatten(y_pred)
#
#     # max_y_true = K.expand_dims(K.repeat_elements(
#     #     K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_true, axis=[2, 3, 4])), shape_r_out, axis=2)),
#     #     shape_c_out, axis=3))
#     max_y_true = K.max(y_true, axis=[2, 3, 4])
#     y_bool = K.cast(K.greater(max_y_true, 0.1), 'float32')
#
#     y_mean = K.mean(y_pred, axis=[2, 3, 4])
#     y_mean = K.expand_dims(K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(y_mean),
#                            y_pred.shape[2], axis=2)), y_pred.shape[3], axis=3))
#
#     y_std = K.std(y_pred, axis=[2, 3, 4])
#     y_std = K.expand_dims(K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(y_std),
#                            y_pred.shape[2], axis=2)), y_pred.shape[3], axis=3))
#
#     y_pred = (y_pred - y_mean) / (y_std + K.epsilon())
#
#     return -K.sum(y_bool*((K.sum(y_true * y_pred, axis=[2, 3, 4])) / (K.sum(y_true, axis=[2, 3, 4]))))