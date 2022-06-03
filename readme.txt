
1.build with build.sh

2.download the following models into models folder：
catalog.pkl
dlibshape_predictor_68_face_landmarks.dat  
e4e_ffhq_encode.pt  
stylegan2-ffhq-config-f.pt  
vgg16-397923af.pth

3.restful api:
  /api/ris
  source:source path in contianer;
  ref: ref path in container;
  style: hair;
  path: result path in container;

Please refine api with path out of container

4. run docker
ARCHIVE_ROOT = "/var/www/archive/"

docker run --gpus all -it --rm --network host -p 8088:8088 -v "target folder":/var/www/archive ris:1.0 bash

