#%%
import os
import nibabel as nib
import numpy as np
import h5py
from glob import glob

# %%
nii_dir = r"\\mega\homes\clwang\super_mudi\cdmri0012"
assert os.path.isdir(nii_dir), 'Dir not exist!'

hr_fname = glob(os.path.join(nii_dir, '*_applytopup.nii.gz'))[0]
ani_fname = glob(os.path.join(nii_dir, '*anisotropic*'))[0]
iso_fname = glob(os.path.join(nii_dir, '*isotropic*'))[0]

assert os.path.isfile(hr_fname), f'{hr_fname} not exist!'
assert os.path.isfile(ani_fname), f'{ani_fname} not exist!'
assert os.path.isfile(iso_fname), f'{iso_fname} not exist!'


hr_data = nib.load(hr_fname).get_fdata()
ani_data = nib.load(ani_fname).get_fdata()
iso_data = nib.load(iso_fname).get_fdata()

print('Data shape:\n', hr_data.shape, ani_data.shape, iso_data.shape)

#%%
out_dir = os.path.join(nii_dir, 'Prepared')
os.makedirs(out_dir)

n_vol = ani_data.shape[-1]
for idx in range(n_vol):
    out_fname = os.path.join(out_dir, os.path.basename(nii_dir)+f'_{idx:04}.h5')
    print('save h5 as {}...'.format(out_fname))
    with h5py.File(out_fname, 'w') as h5f:
        h5f.create_dataset('highres_data', data=hr_data[0:-1,:,:,idx], dtype=np.float64, compression='gzip', compression_opts=5) # int16
        h5f.create_dataset('anisotropic_data', data=ani_data[...,idx], dtype=np.float64, compression='gzip', compression_opts=5)
        h5f.create_dataset('isotropic_data',  data=iso_data[...,idx], dtype=np.float64, compression='gzip', compression_opts=5)
        
        print('saved highres_data:', np.shape(h5f.get('highres_data')))
        print('saved anisotropic_data:', np.shape(h5f.get('anisotropic_data')))
        print('saved isotropic_data:', np.shape(h5f.get('isotropic_data')))
 

# %%
