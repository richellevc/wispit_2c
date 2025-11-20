
from pynpoint import *

fwhm = 4.*0.01225  # (arcsec)
cent_size = fwhm  # (arcsec)
edge_size = 3.  # (arcsec)
#extra_rot = -134.24  # (deg)
extra_rot = 0.0  # (deg)
aperture = 5.*0.01225  # (arcsec)
psf_scaling = 6.9*76.42
pca_number = 5
position = (122, 86)

band = "Ks"

output_place = f'./output/{band}/'

pipeline = Pypeline('./', './', output_place)

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
module = MCMCsamplingModule(name_in='mcmc',
                            image_in_tag='science_crop',
                            psf_in_tag='flux_crop_mean',
                            chain_out_tag='mcmc',
                            param=(sep, ang, mag),
                            bounds=((sep-0.01, sep+0.01), (ang-5., ang+5.), (mag-0.5, mag+0.5)),
                            nwalkers=200,
                            nsteps=500,
                            psf_scaling=-psf_scaling,
                            pca_number=pca_number,
                            aperture=(position[0], position[1], aperture),
                            mask=(cent_size, edge_size),
                            extra_rot=extra_rot,
                            merit='gaussian',
                            residuals='mean',
                            resume=False)
pipeline.add_module(module)
#pipeline.run_module('mcmc')
module = SystematicErrorModule(name_in='error',
                               image_in_tag='science',
                               psf_in_tag='flux',
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
pipeline.add_module(module)
# pipeline.run_module('error')

pipeline.run()
# note: set offset to 1 or 2 for 1c