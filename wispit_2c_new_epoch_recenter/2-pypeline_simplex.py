from pynpoint import *

fwhm = 4.*0.01225  # (arcsec)

cent_size = 5*0.01225
edge_size = 13*0.01225

extra_rot = 0.0  # (deg)
aperture = 5.*0.01225  # (arcsec)
pca_number = 14  # rdi H


position = (52.5, 41)    # 2c

band = "H"
psf_scaling = 7.94187*76.42


output_place = f'./output/{band}/'

working_place_in = f'./input/{band}/'
pipeline = Pypeline(working_place_in, './', output_place)

module_simplex = SimplexMinimizationModule(position=position,
                                   magnitude=7.5,       # 2c
                                   psf_scaling=-psf_scaling,
                                   name_in='simplex',
                                   image_in_tag='science_crop_tc_masked',
                                   reference_in_tag='ref_crop_tc_masked',
                                   psf_in_tag='flux_crop_mean',
                                   res_out_tag='simplex',
                                   flux_position_tag='fluxpos',
                                   merit='poisson',
                                   aperture=fwhm,
                                   sigma=0.,
                                   tolerance=0.01,
                                   pca_number=range(7, 16),
                                   cent_size=cent_size,
                                   edge_size=edge_size,
                                   extra_rot=extra_rot,
                                   residuals='mean',
                                   offset=2.)

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