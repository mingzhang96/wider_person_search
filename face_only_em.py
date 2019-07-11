import cv2
import json
import argparse
import os.path as osp
import numpy as np
import mxnet as mx

from utils.timer import Timer
from utils.pkl import my_pickle, my_unpickle
from mtcnn.mtcnn_detector import MtcnnDetector
from arcface.face_em import FaceModel

trainval_root = '/local/home/share/safe_data_dir_3/zhangming/wider/person_search_trainval'
test_root = '/local/home/share/safe_data_dir_3/zhangming/wider/person_search_test'

def load_json(name):
    with open(name) as f:
        data = json.load(f)
        return data

def select(w, h, boundingboxes):
    if len(boundingboxes) == 1:
        return 0
    else:
        midx, midy = w/2, h/3
        dist, area = [], []
        for bbox in boundingboxes:
            x, y = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
            dist.append(np.sqrt((x-midx)*(x-midx)+(y-midy)*(y-midy)))
            area.append((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
        dist /= np.linalg.norm(dist)
        area /= np.linalg.norm(area)
        p = area/dist
        return np.argsort(p)[-1]

def face_det_cast(img, detector):
    height, width, _ = img.shape
    ratio = height / 200.0
    w, h = int(width/ratio), 200
    resize_img = cv2.resize(img, (w,h))
    # det
    results = detector.detect_face(resize_img)
    # selection
    if results is None:
        return None, None
    bboxes, landmarks = results
    if len(bboxes) != 1:
        ind = select(w, h, bboxes)
        bbox = bboxes[ind]
        landmark = landmarks[ind]
    else:
        bbox = bboxes[0]
        landmark = landmarks[0]
    bbox[:4] *= ratio
    landmark *= ratio
    return bbox, landmark

def face_det_candi(img, rect, detector):
    crop = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]].copy()
    # det
    results = detector.detect_face(crop)
    if results is None:
        return crop, None, None
    bboxes, landmarks = results
    if len(bboxes) != 1:
        ind = select(rect[2], rect[3], bboxes)
        bbox = bboxes[ind]
        landmark = landmarks[ind]
    else:
        bbox = bboxes[0]
        landmark = landmarks[0]
    return crop, bbox, landmark

def face_exfeat(img, fbbox, landmark, face_model):
    # _, aligned = face_model.get_input(img, fbbox, landmark)
    feat = face_model.get_feature(img)
    return feat

def main(args):
    is_test = True if args.is_test == '1' else False
    _t = Timer()
    embedding = FaceModel(model='./arcface/model/'+args.arch,
                          ctx=mx.gpu(args.gpu))
    if is_test:
        this_dir, json_path, save_name = osp.join(test_root, 'test'), osp.join(test_root, 'test.json'), 'face_em_test_'+args.arch+'.pkl'
    else:
        this_dir, json_path, save_name = osp.join(trainval_root, 'val'), osp.join(trainval_root, 'val.json'), 'face_em_val_'+args.arch+'.pkl'
    data_raw = load_json(json_path)
    movie_num, movie_cnt = len(data_raw.keys()), 0

    face_data = my_unpickle('./features/face_det_test.pkl')

    face_dict = {}
    # det/extract val face feat
    for movie, info in face_data.items():
        face_dict.update({movie:{'cast':[], 'candidates':[]}})
        movie_cnt += 1
        casts, casts_num = info['cast'], len(info['cast'])
        candidates, candidates_num = info['candidates'], len(info['candidates'])
        for i, cast in enumerate(casts):
            # img_path = osp.join(this_dir, cast['img'])
            # img = cv2.imread(img_path)
            cast_id = cast['id']
            _t.tic()
            fbbox, landmark, img = cast['fbbox'], cast['landmark'], cast['aligned']
            assert fbbox is not None, 'Cast: No face detected !'
            ffeat = face_exfeat(img, fbbox, landmark, embedding)
            _t.toc()
            print('%s %d/%d ... %s %d/%d ... time: %.3f s average: %.3f s'%(movie, movie_cnt, movie_num, 
                                                cast_id, i+1, casts_num, _t.diff, _t.average_time))
            face_dict[movie]['cast'].append({
                'id': cast_id,
                'ffeat': ffeat
            })
        for i, candidate in enumerate(candidates):
            # img_path = osp.join(this_dir, candidate['img'])
            # img = cv2.imread(img_path)
            candidate_id = candidate['id']
            _t.tic()
            fbbox, landmark, img = candidate['fbbox'], candidate['landmark'], candidate['aligned']
            if fbbox is None:
                _t.toc()
                print('%s %d/%d ... %s %d/%d ... time: %.3f s average: %.3f s'%(movie, movie_cnt, movie_num, 
                                                    candidate_id, i+1, candidates_num, _t.diff, _t.average_time))
                face_dict[movie]['candidates'].append({
                    'id': candidate_id,
                    'ffeat': None
                })
                continue
            ffeat = face_exfeat(img, fbbox, landmark, embedding)
            _t.toc()
            print('%s %d/%d ... %s %d/%d ... time: %.3f s average: %.3f s'%(movie, movie_cnt, movie_num, 
                                                candidate_id, i+1, candidates_num, _t.diff, _t.average_time))
            face_dict[movie]['candidates'].append({
                'id': candidate_id,
                'ffeat': ffeat
            })
    my_pickle(face_dict, osp.join('./features', save_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is-test', type=str, default='0', choices=['0', '1'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--arch', type=str, default="model-r50-am-lfw")
    args = parser.parse_args()
    main(args)
