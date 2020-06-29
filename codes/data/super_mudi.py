import os, random, cv2
import numpy as np
import torch
import torch.utils.data as data
# import data.util as util
import util
from utils_cw import get_items_from_file, load_h5


class SuperMUDIDataset(data.Dataset):
    """
    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, etc) and GT image pairs.
    If only GT images are provided, generate LQ images on-the-fly.
    """

    def __init__(self, opt):
        super(SuperMUDIDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.list_fname = self.opt['data_list']
        self.slice_axis = self.opt['slice_axis']
        assert os.path.isfile(self.list_fname), f'Error: file path is empty. {self.list_fname}'

        data_fnames = get_items_from_file(self.list_fname, format='json')
        self.data_fnames = list(filter(lambda x: os.path.isfile(x), data_fnames))
        print(f"{len(self.data_fnames)} data found!")

        self.random_scale_list = [1]

    def __getitem__(self, index):
        scale = self.opt['scale']

        # load h5 data
        hr_data, lr_data = load_h5(self.data_fnames[index], ['highres_data', 'anisotropic_data'])

        # if self.opt['phase'] != 'train':  # modcrop in the validation / test phase
        #     img_GT = util.modcrop(img_GT, scale) #! 尺寸整除?

        if self.opt['phase'] == 'train':
            slice_idx = random.randint(0, hr_data.shape[self.slice_axis]-1)
            hr_slice = hr_data.take(slice_idx, axis=self.slice_axis)
            lr_slice = lr_data.take(slice_idx, axis=self.slice_axis)

            # augmentation - flip, rotate
            img_lr, img_hr = util.augment([lr_slice, hr_slice], self.opt['use_flip'], self.opt['use_rot'])
            img_lr, img_hr = img_lr[np.newaxis, ...], img_hr[np.newaxis, ...]

        img_hr = torch.from_numpy(np.ascontiguousarray(img_hr)).float()
        img_lr = torch.from_numpy(np.ascontiguousarray(img_lr)).float()

        return {'LQ': img_lr, 'GT': img_hr}

    def __len__(self):
        return len(self.data_fnames)


if __name__ == '__main__':
    options = {'data_type': 'LQGT', 'data_list':r"\\mega\homes\clwang\super_mudi\traindata_list.json", 
               'slice_axis':0, 'scale':2, 'phase':'train', 'use_flip':False, 'use_rot':False}
    
    ds = SuperMUDIDataset(options)
    for elem in ds:
        print(elem['LQ'].shape, elem['GT'].shape)
