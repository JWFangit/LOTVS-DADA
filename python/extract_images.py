import glob
import os
import tqdm
import cv2


def video2image(video_pth, save_pth, mode='images'):
    vc = cv2.VideoCapture(video_pth)  #
    c = 0
    rval = vc.isOpened()
    while rval:  #
        c = c + 1
        rval, frame = vc.read()
        if rval:
            name = os.path.join(save_pth, str(c).zfill(4) + '.jpg')
            if mode == 'maps':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(name, frame)
            cv2.waitKey(1)
        else:
            break
    vc.release()


if __name__ == '__main__':
    videos_pth = './videos/*'  # Change to your own path
    maps_pth = './maps/*'  # Change to your own path

    DADA_dataset_pth = './DADA_dataset'  # Change to your own path of DADA_dataset

    all_videos = glob.glob(videos_pth)
    all_maps = glob.glob(maps_pth)

    for video in tqdm.tqdm(all_videos):
        print(video)
        video_name = os.path.basename(video)
        temp = video_name.split('_')
        cc, category, folder = temp[0], temp[1], temp[2].split('.')[0]

        save_images_pth = os.path.join(DADA_dataset_pth, category, folder, cc)

        if not os.path.exists(save_images_pth):
            os.makedirs(save_images_pth)

        video2image(video, save_images_pth)

    # for maps in tqdm.tqdm(all_maps):  # If you  want to use maps after video conversion, Please cancel these comments..
    #     print(maps)
    #     maps_name = os.path.basename(maps)
    #     temp = maps_name.split('_')
    #     cc, category, folder = temp[0], temp[1], temp[2].split('.')[0]
    #     save_maps_pth = os.path.join(DADA_dataset_pth, category, folder, cc)
    #
    #     if not os.path.exists(save_maps_pth):
    #         os.makedirs(save_maps_pth)
    #
    #     video2image(maps, save_maps_pth, 'maps')
