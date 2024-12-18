import argparse
import sys
import cv2 as cv
import numpy as np
import torch
from model import FFDNet
from senxor.utils import connect_senxor, data_to_frame, remap


def prep_input(frame: np.ndarray) -> torch.Tensor:
    """function to prepare mi48 frame as input for model inference
    Args:
        frame (np.ndarray): frame given by mi48.read()
    Returns:
        torch.Tensor: input tensor for model inference
    """
    img = np.expand_dims(frame, 0)
    img = np.float32(img / 255)
    return torch.FloatTensor([img])


def test_gray(args):
    mi48 = connect_senxor()
    mi48.start(stream=True, with_header=True)
    data, _ = mi48.read()
    if data is None:
        mi48.stop()
        sys.exit(1)
    
    ncols, nrows = mi48.fpa_shape

    # load model
    model = FFDNet(is_gray=True)
    if args.cuda:
        image = image.cuda()
        noise_sigma = noise_sigma.cuda()
        model = model.cuda()

    noise_sigma = torch.FloatTensor([0.])

    model_path = args.model_path.replace('"', '') + ('net_gray.pth')
    print(f"> Loading model param in {model_path}...")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    cv.namedWindow("raw")
    cv.namedWindow("denoise")
    SCALE = 3
    
    with torch.no_grad():
            while True:
                data, _ = mi48.read()
                if data is None:
                    mi48.stop()
                    sys.exit(1)
                
                frame = remap(data_to_frame(data, (ncols, nrows)))
                image = prep_input(frame)    
                image_pred = torch.squeeze(model(image, noise_sigma).cpu()).numpy()
                image_pred = (image_pred*255.).clip(0, 255).astype(np.uint8)
    
                cv.imshow("raw", cv.resize(frame, dsize=None, fx=SCALE, fy=SCALE, interpolation=cv.INTER_CUBIC))
                cv.imshow("new", cv.resize(image_pred, dsize=None, fx=SCALE, fy=SCALE, interpolation=cv.INTER_CUBIC))
                key = cv.waitKey(1)  # & 0xFF
                if key == ord("q"):
                    break
            mi48.stop()
            cv.destroyAllWindows()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default='./test_data/gray.png',           help='Test image path.')
    parser.add_argument("--model_path", type=str, default='./models/',                      help='Model loading and saving path.')
    parser.add_argument("--use_gpu", action='store_true',                                   help='Train and test using GPU.')
    args = parser.parse_args()
    args.cuda = args.use_gpu and torch.cuda.is_available()
    test_gray(args)

