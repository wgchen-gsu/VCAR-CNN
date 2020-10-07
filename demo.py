import argparse
import numpy as np
from pathlib import Path
import matlab.engine
import os
import time
import cv2
from model_car import get_model

def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video_path", type=str, required=True, help="test compressed video")
    parser.add_argument("--ori_video", type=str, required=True, help="original video")
    parser.add_argument("--model", type=str, default="srresnet", help="model architecture, 'srresnet' only")
    parser.add_argument("--weight_file", type=str, default='models/weights.393-6.991-30.79444.hdf5', help="trained weight file")
    parser.add_argument("--output_dir", type=str, default='output_A', help="if set, save resulting sequence")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--step", type=int, default=2)
    args = parser.parse_args()
    return args

def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)

def main():
    args = get_args()
    rows = args.height
    cols = args.width
    iter_step = args.step
    video_path = args.video_path
    ori_path   = args.ori_video
    weight_file = args.weight_file
    model = get_model(args.model)
    model.load_weights(weight_file)
    num_input = 3

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if os.path.exists(video_path) == False:
        os._exit(0)

    print(ori_path)

    len = os.path.getsize(video_path)
    num_F = len//(rows*cols*3//2)
    out_image = np.zeros((rows, cols * 3, 1), dtype=np.uint8)
    
    eng = matlab.engine.start_matlab()

    for n in range(0, num_F,  iter_step):
        comp_im, skip_F = eng.frame_compensation(video_path, n+1, num_F, rows, cols, nargout=2)  
        ori_im = eng.read_y(ori_path, n+1, rows, cols, nargout=1)

        im_data = np.array(comp_im)
        ori_data= np.array(ori_im)
        
        if int(skip_F) == 1:
            out_image[:, :cols, 0] = ori_data[:rows, :cols]
            out_image[:, cols:cols*2, 0]   = im_data[:rows, :cols]
            out_image[:, cols*2:cols*3, 0] = im_data[:rows, :cols]

            str_name = '%(name)05d.png'%{'name':n}
            cv2.imwrite(str(output_dir.joinpath(str_name)), out_image)
            continue

        xim = np.zeros((rows, cols, num_input), dtype=np.uint8)
        for k in range(0, num_input):
            xim[:,:,k] = im_data[rows*k:rows*(k+1), :cols]

        pred = model.predict(np.expand_dims(xim.astype(np.float), 0)) 
        rim  = get_image(pred[0])
        
        out_image[:, :cols, 0] = ori_data[:rows, :cols]
        out_image[:, cols:cols*2, 0] = im_data[:rows, :cols]
        out_image[:, cols*2:cols*3, 0] = rim[:rows, :cols, 0]

        str_name = '%(name)05d.png'%{'name':n}
        cv2.imwrite(str(output_dir.joinpath(str_name)), out_image)            

if __name__ == '__main__':
    main()