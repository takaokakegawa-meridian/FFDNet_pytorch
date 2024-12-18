import argparse
import cv2 as cv
import numpy as np
import torch
from model import FFDNet

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
    # load image
    frame = cv.imread(args.test_path, cv.IMREAD_GRAYSCALE)
    image = prep_input(frame)

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
    print('\n')

    with torch.no_grad():
        image_pred = torch.squeeze(model(image, noise_sigma).cpu()).numpy()
        image_pred = (image_pred*255.).clip(0, 255).astype(np.uint8)
        print(image_pred.shape)
    
    SCALE = 3
    cv.imshow("raw", cv.resize(frame, dsize=None, fx=SCALE, fy=SCALE, interpolation=cv.INTER_CUBIC))
    cv.imshow("new", cv.resize(image_pred, dsize=None, fx=SCALE, fy=SCALE, interpolation=cv.INTER_CUBIC))
    cv.waitKey(0)
    cv.destroyAllWindows()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default='./test_data/gray.png',           help='Test image path.')
    parser.add_argument("--model_path", type=str, default='./models/',                      help='Model loading and saving path.')
    parser.add_argument("--use_gpu", action='store_true',                                   help='Train and test using GPU.')
    args = parser.parse_args()
    args.cuda = args.use_gpu and torch.cuda.is_available()
    test_gray(args)

