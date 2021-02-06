import torch 
import json 
import os 
import h5py 
from torch.utils.data import Dataset 


class UDADataset(Dataset):

    # output: image region, caption
    def __init__(self, data_path):
        self.img_features = h5py.File(os.path.join(data_path, 'coco_detections.hdf5'), 'r')
        train_data_path = os.path.join(data_path, 'annotations')
        with open(os.path.join(train_data_path, 'captions_train2014.json')) as f:
            self.train_data = json.load(f)['annotations'] 
        

    def __getitem__(self, i):
        cap_dict = self.train_data[i]
        img_id = str(cap_dict['image_id']) + '_features' 
        img = torch.FloatTensor(self.img_features[img_id])
        txt = cap_dict['caption']

        return img, txt

    #def __len__(self):
    #    return len(self.txt) 


if __name__ == "__main__":
    path = 'data' 
    data = UDADataset(path) 
    img_feature, txt = data[0]
    print(txt)



