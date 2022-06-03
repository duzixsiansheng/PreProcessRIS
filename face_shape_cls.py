import json
import numpy as np
import cv2
import imageio
from sklearn.cluster import KMeans

def load_face_shape(datapath='./ffhq-dataset-v2.json'):
    with open(datapath, 'r') as f:
        data = json.load(f)
    
    faceshapes = []
    for id in data:
        faceshape = np.array(data[id]['image']['face_landmarks'])
        h = faceshape[:, 1].max() - faceshape[:, 1].min()
        w = faceshape[:, 0].max() - faceshape[:, 0].min()

        # norm
        scale = 1.0 / max(h, w)
        faceshape = faceshape * scale

        # move to center
        faceshape[:, 1] = faceshape[:, 1] - (faceshape[:, 1].min() + faceshape[:, 1].max()) * 0.5 + 0.5
        faceshape[:, 0] = faceshape[:, 0] - (faceshape[:, 0].min() + faceshape[:, 0].max()) * 0.5 + 0.5

        faceshapes.append(faceshape)
    
    faceshapes = np.stack(faceshapes)
    
    return faceshapes


def plot_face_shape(faceshape, imsize=300, text_vis=True):
    image = np.zeros((imsize, imsize, 3)) + 1

    for i, (x, y) in enumerate(faceshape):
        x = int(x * imsize)
        y = int(y * imsize)
        image = cv2.circle(image, (x, y), 5, (1, 0, 0), -1)
        if text_vis:
            image = cv2.putText(image,"%d"%i, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 0, 1))
    return image

   



def k_means():
    faceshapes = load_face_shape()
    faceshapes = faceshapes[:, :17]

    # mean_face = np.mean(faceshapes, axis=0)
    # image = plot_face_shape(mean_face)
    # imageio.imwrite('./mean_face_shape.png', image)

    thresh = 0.03
    filtered_faceshapes = []
    for f in faceshapes:
        mid8 = f[8, 0]
        mid79 = 0.5 * (f[7, 0] + f[9, 0])
        # image = plot_face_shape(f)
        # imageio.imwrite('./tmp.png', image)
        if abs(mid8 - 0.5) < thresh and abs(mid79 - 0.5) < thresh:
            filtered_faceshapes.append(f)
    faceshapes = np.stack(filtered_faceshapes, axis=0)

    faceshapes = faceshapes.reshape(faceshapes.shape[0], -1)
    cluster_nums = [3, 7, 11]
    for cluster_num in cluster_nums:
        # cluster_num = 3
        kmeans = KMeans(n_clusters = cluster_num, random_state=0).fit(faceshapes)
        center_faces = kmeans.cluster_centers_
        for i, center_face in enumerate(center_faces):
            center_face = center_face.reshape(-1, 2)
            image = plot_face_shape(center_face, text_vis=False)
            imageio.imwrite('./kmeans_cluster%d_%d.png'%(cluster_num, i), image)


def draw_mean_face():
    faceshapes = load_face_shape()
    mean_face = np.mean(faceshapes, axis=0)
    image = plot_face_shape(mean_face)
    imageio.imwrite('./mean_face.png', image)


if __name__ == '__main__':
    # faceshapes = load_face_shape()
    # draw_mean_face()
    k_means()