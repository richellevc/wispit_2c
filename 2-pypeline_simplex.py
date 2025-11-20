from pynpoint import *

fwhm = 4.*0.01225  # (arcsec)
cent_size = fwhm  # (arcsec)
edge_size = 3.  # (arcsec)
#extra_rot = -134.24  # (deg)
extra_rot = 0.0  # (deg)
aperture = 5.*0.01225  # (arcsec)
#psf_scaling = 6.9*76.42
psf_scaling = 7.94187*76.42 # test assuming latter is exp, first is nd
#pca_number = 5  # k band
pca_number = 5  # rdi H
position = (102, 90)

band = "H"

output_place = f'./output/{band}/'

working_place_in = f'./input/{band}/'
pipeline = Pypeline(working_place_in, './', output_place)

module_simplex = SimplexMinimizationModule(position=position,
                                   magnitude=7.5,
                                   psf_scaling=-psf_scaling,
                                   name_in='simplex',
                                   image_in_tag='science_crop_tc',
                                   reference_in_tag='ref_crop_tc',
                                   psf_in_tag='flux_crop_mean',
                                   res_out_tag='simplex',
                                   flux_position_tag='fluxpos',
                                   #merit='gaussian',
                                   merit='poisson',
                                   aperture=fwhm,
                                   sigma=0.,
                                   tolerance=0.01,
                                   pca_number=range(1, pca_number+2),
                                   cent_size=cent_size,
                                   edge_size=edge_size,
                                   extra_rot=extra_rot,
                                   residuals='mean',
                                   offset=3.)

module_simplex_write = FitsWritingModule(name_in='write_simplex',
                            data_tag=f'simplex{pca_number:03.0f}',
                            file_name=f'simplex{pca_number:03.0f}.fits',
                            output_dir=None,
                            data_range=None,
                            overwrite=True)
# note: set offset to 1 or 2 for 1c

module_hdf5 = Hdf5WritingModule(name_in='write_hdf5',
                           file_name='tyc_irdis_bks.hdf5',
                           output_dir=None,
                           tag_dictionary={'science_crop': 'science_crop',
                                           'flux_crop': 'flux_crop',
                                           'fluxpos{pca_number:03.0f}': 'fluxpos'},
                           keep_attributes=True,
                           overwrite=True)

pipeline.add_module(module_simplex)
pipeline.add_module(module_simplex_write)
pipeline.add_module(module_hdf5)

pipeline.run()