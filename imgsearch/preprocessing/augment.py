import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def augment_img(img_path, n):
    '''
    Create n randomly augmented copies of a given image and save them in the same folder
    '''
    data_aug = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    img = load_img(img_path)
    arr_img = img_to_array(img)
    exp_img = arr_img.reshape((1,) + arr_img.shape)

    cur_dir = os.path.dirname(img_path)  # current img directory

    prev_len = len(os.listdir(cur_dir))  # number of element current img directory

    cur_len = prev_len

    while cur_len - prev_len != n:  # until n new imgs are successfully created
        next(
            data_aug.flow(exp_img,
                          batch_size=1,
                          save_to_dir=cur_dir,
                          save_prefix='aug',
                          save_format='jpg')
        )

        cur_len = len(os.listdir(cur_dir))
