
from pynpoint import *
from scripts.mcmc_module import MCMCsamplingModule_rdi
import numpy as np

fwhm = 4.*0.01225  # (arcsec)
cent_size = fwhm  # (arcsec)
edge_size = 3.  # (arcsec)
#extra_rot = -134.24  # (deg)
extra_rot = 0.0  # (deg)
aperture = 5.*0.01225  # (arcsec)
#pca_number = 5  # k band
pca_number = 14  # rdi H


position = (101.4, 90.8)    # 2c
#position = (122, 86)    # 2b

band = "H"
psf_scaling = 7.94187*76.42

#band = "Ks"
#psf_scaling = 6.9*76.42

output_place = f'./output/{band}/'

working_place_in = f'./input/{band}/'
pipeline = Pypeline(working_place_in, './', output_place)


#module = Hdf5ReadingModule(name_in='read',
#                           input_filename='tyc_irdis_bks.hdf5',
#                           input_dir=None,
#                           tag_dictionary={'science_crop': 'science',
#                                           'flux_crop': 'flux',
#                                           'fluxpos': 'fluxpos'})
#pipeline.add_module(module)
## pipeline.run_module('read')

simplex = pipeline.get_data(f'fluxpos{pca_number:03.0f}')
sep = simplex[-1, 2]
ang = simplex[-1, 3]
mag = simplex[-1, 4]
print(sep, ang, mag)
module = MCMCsamplingModule_rdi(
                            name_in=f'mcmc_pca_{pca_number:03.0f}',
                            image_in_tag='science_crop_tc_masked',
                            reference_in_tag='ref_crop_tc_masked',
                            psf_in_tag='flux_crop_mean',
                            chain_out_tag=f'mcmc_pca_{pca_number:03.0f}',
                            param=(sep, ang, mag),
                            bounds=((sep-0.01, sep+0.01), (ang-5., ang+5.), (mag-0.5, mag+0.5)),
                            nwalkers=10,
                            nsteps=10,
                            psf_scaling=-psf_scaling,
                            pca_number=pca_number,
                            aperture=(int(np.round(position[0],0)), int(np.round(position[1],0)), aperture),
                            mask=(cent_size, edge_size),
                            extra_rot=extra_rot,
                            merit='poisson',
                            residuals='mean',
                            resume=False)
pipeline.add_module(module)
#pipeline.run_module('mcmc')
module = SystematicErrorModule(name_in=f'error_pca_{pca_number:03.0f}',
                               image_in_tag='science_crop_tc_masked',
                               psf_in_tag='flux_crop_mean',
                               offset_out_tag='offset',
                               position=(sep, ang),
                               magnitude=mag,
                               angles=(0., 359., 360),
                               psf_scaling=psf_scaling,
                               merit='gaussian',
                               aperture=aperture,
                               pca_number=pca_number,
                               mask=(cent_size, edge_size),
                               extra_rot=extra_rot,
                               residuals='mean',
                               offset=3.)

#pipeline.add_module(module)
# pipeline.run_module('error')

pipeline.run()
# note: set offset to 1 or 2 for 1c