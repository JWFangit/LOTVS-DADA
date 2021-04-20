batch_size = 2
nb_train = 8000
nb_epoch = 1
nb_videos_val = 100

# -----
smooth_weight = 0.5

node = 16
input_t = 5  # 表示一次输入多少张图片
input_shape = (256, 192)
small_output_shape = (int(input_shape[0] / 8), int(input_shape[1] / 8))
out_dims = (None, input_t) + small_output_shape + (512,)
output_dim = (input_t,) + small_output_shape + (1,)

shape_r, shape_c = input_shape[0], input_shape[1]
# shape_r_out, shape_c_out = int(input_shape[0] / 2), int(input_shape[1] / 2)
shape_r_out, shape_c_out = input_shape[0], input_shape[1]


mode = 'test'  # 'train' or 'test'
train_data_pth = 'E:/DADA_dataset/train'
val_data_pth = 'E:/DADA_dataset/val'


pre_train = False
pre_train_path = './models/SCAFNet.h5'

test_path = 'E:/DADA_dataset/test/'
test_save_path = './predicts/'

# test_path = 'E:/UCF/val/'
# test_save_path = 'E:/predicts/UCF/MyNet/'


