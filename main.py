import glob
import tqdm
import torch
import torchvision

import hopenet
from deepface import DeepFace
import DPR.model.defineHourglass_512_gray_skip 
from dpr import get_shading


print('*' * 50)
print('Load pretrained models')
print('*' * 50)

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

print('*' * 50)
print('Start to predict')
print('*' * 50)
imlist = glob.glob('./test_dataset/**/*.png')
for impath in imlist:
    with torch.no_grad():
        # age, gender, race, emotion
        rst = DeepFace.analyze(img_path = impath, models=deepface_models, actions = ['age', 'gender', 'race', 'emotion'], enforce_detection=False)

        # head pose
        yaw, pitch, roll = hopenet.get_head_pose(impath, headpose_model)
        rst['headpose'] = [yaw, pitch, roll]

        # shading
        shading = get_shading(shading_model, impath)
        rst['shading'] = shading.cpu().numpy()[0, :, 0, 0]

        print(rst)
