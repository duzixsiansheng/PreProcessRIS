import glob
import tqdm
import torch
import torchvision

import hopenet
from deepface import DeepFace
import DPR.model.defineHourglass_512_gray_skip 
from dpr import get_shading
import dlib
import numpy as np


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def shape_to_np(shape):
    coords = np.zeros((68, 2))
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def norm_face_shape(faceshape):
    h = faceshape[:, 1].max() - faceshape[:, 1].min()
    w = faceshape[:, 0].max() - faceshape[:, 0].min()

    # norm
    scale = 1.0 / max(h, w)
    faceshape = faceshape * scale

    # move to center
    faceshape[:, 1] = faceshape[:, 1] - (faceshape[:, 1].min() + faceshape[:, 1].max()) * 0.5 + 0.5
    faceshape[:, 0] = faceshape[:, 0] - (faceshape[:, 0].min() + faceshape[:, 0].max()) * 0.5 + 0.5
    return faceshape


def crop_face(img, det):
    x_min, y_min, x_max, y_max = det.left(), det.top(), det.right(), det.bottom()
    bbox_width = abs(x_max - x_min)
    bbox_height = abs(y_max - y_min)
    x_min -= 2 * bbox_width / 4
    x_max += 2 * bbox_width / 4
    y_min -= 3 * bbox_height / 4
    y_max += bbox_height / 4
    x_min = max(x_min, 0); y_min = max(y_min, 0)
    x_max = min(img.shape[1], x_max); y_max = min(img.shape[0], y_max)
    # Crop image
    img = img[y_min:y_max,x_min:x_max]

    return img


def filter_faceshape(f, thresh):
    mid8 = f[8, 0]
    mid79 = 0.5 * (f[7, 0] + f[9, 0])
    # image = plot_face_shape(f)
    # imageio.imwrite('./tmp.png', image)
    return abs(mid8 - 0.5) < thresh and abs(mid79 - 0.5) < thresh


def demo():
    ######################################
    # Model initialization
    ######################################
    
    print('=> Load pretrained models')
    # face detection and face landmark detection
    dlib_face_detector = dlib.get_frontal_face_detector()
    dlib_face_shape_predictor = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')
    
    # age, gender, emotion, and race models
    deepface_models = {}
    deepface_models['age'] = DeepFace.build_model('Age')
    deepface_models['gender'] = DeepFace.build_model('Gender')
    deepface_models['emotion'] = DeepFace.build_model('Emotion')
    deepface_models['race'] = DeepFace.build_model('Race')
    
    # headpose model
    headpose_model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    headpose_model.load_state_dict(torch.load('weights/hopenet_robust_alpha1.pkl'))
    headpose_model.eval().cuda()
        
    # shading model
    shading_model = DPR.model.defineHourglass_512_gray_skip.HourglassNet()
    shading_model.load_state_dict(torch.load('DPR/trained_model/trained_model_03.t7'))
    shading_model.cuda()
    shading_model.train(False)
    

    ######################################
    # Model initialization
    ######################################
    print('=> Start to predict')
    impath = './04910.png'
    
    rst = {}
    # face detection and check face size
    img = dlib.load_rgb_image(impath)
    dets = dlib_face_detector(img, 1)
    
    if len(dets) < 1:
        print('No face detected')
        return

    if len(dets) > 1:
        print('More than 1 face detected')
        return
    
    det = dets[0]
    short_side = min(det.right() - det.left(), det.bottom() - det.top())
    if short_side < 200:
        print('Face is too small')
        return
    
    rst['dlib_face_loc'] = rect_to_bb(det)

    # detect face shape and check the degree of the face shape
    shape = dlib_face_shape_predictor(img, det)
    shape = shape_to_np(shape)
    face_shape = shape[:17, :] # only face
    norm_shape = norm_face_shape(face_shape.copy())
    # if not filter_faceshape(norm_shape, 0.1):
    #     print('Please face to the camera')
    #     return

    rst['dlib_face_shape'] = shape
    
    # crop face
    face_crop = crop_face(img.copy(), det)


    ######################################
    # Following parts should be calculated in servers
    ######################################
    with torch.no_grad():
        # head pose
        yaw, pitch, roll = hopenet.get_head_pose(face_crop, headpose_model) 
        rst['headpose'] = [yaw, pitch, roll]
    
        # shading
        shading = get_shading(shading_model, face_crop)
        rst['shading'] = shading.cpu().numpy()[0, :, 0, 0]
    
        # age, gender, race, emotion
        deepface_rst = DeepFace.analyze(img_path = impath, models=deepface_models, actions = ['age', 'gender', 'race', 'emotion'], enforce_detection=False, detector_backend='mtcnn', prog_bar=False)
        for k in deepface_rst:
            rst[k] = deepface_rst[k]

        print(rst)


if __name__ == '__main__':
    demo()
