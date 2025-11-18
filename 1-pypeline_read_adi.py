from pynpoint import *

fwhm = 4.*0.01225  # (arcsec)
cent_size = fwhm  # (arcsec)
edge_size = 3.  # (arcsec)
#extra_rot = -134.24  # (deg)   # original with christians header
extra_rot = 0.0  # (deg) - extra rot already taken into account in parang.dat
aperture = 5.*0.01225  # (arcsec)
psf_scaling = 6.9*76.42
pca_number = 5
position = (65, 29)

band = "Ks"

science_file = f'./input/{band}/sci.fits'
flux_file = f'./input/{band}/flux_cube_summed.fits'
parang_file = f'./input/{band}/parang.dat'

output_place = f'./output/{band}/'

pipeline = Pypeline('./', './', output_place)

module = FitsReadingModule(name_in=f'read1',
                           input_dir=None,
                           image_tag='science',
                           overwrite=True,
                           check=False,
                           filenames=[science_file, ])

pipeline.add_module(module)

module = FitsReadingModule(name_in=f'read2',
                           input_dir=None,
                           image_tag='flux',
                           overwrite=True,
                           check=False,
                           filenames=[flux_file, ])

pipeline.add_module(module)

module = ParangReadingModule(name_in=f'parang',
                             data_tag='science',
                             file_name=parang_file,
                             input_dir=None,
                             overwrite=True)

pipeline.add_module(module)

# module = ShiftImagesModule(name_in=f'shift',
#                            image_in_tag='science',
#                            image_out_tag='science_shift',
#                            shift_xy=(0.5, 0.5),
#                            interpolation='spline')
#
# pipeline.add_module(module)

# The number of lines that are removed in left, right, bottom, and top direction.
module = RemoveLinesModule(name_in=f'cut1',
                           image_in_tag='science',
                           image_out_tag='science_crop',
                           lines=(470, 470, 470, 470))

pipeline.add_module(module)

module = RemoveLinesModule(name_in=f'cut2',
                           image_in_tag='flux',
                           image_out_tag='flux_crop',
                           lines=(470, 470, 470, 470))

pipeline.add_module(module)

module = PSFpreparationModule(name_in=f'prep',
                              image_in_tag='science_crop',
                              image_out_tag='prep',
                              mask_out_tag=None,
                              norm=False,
                              cent_size=cent_size,
                              edge_size=edge_size)

pipeline.add_module(module)

module = PcaPsfSubtractionModule(name_in=f'pca',
                                 images_in_tag='prep',
                                 reference_in_tag='prep',
                                 res_mean_tag='pca',
                                 pca_numbers=range(1, 21),
                                 extra_rot=extra_rot,
                                 subtract_mean=True)

pipeline.add_module(module)

module = FitsWritingModule(name_in=f'write1',
                           data_tag=f'pca',
                           file_name=f'pca.fits',
                           output_dir=None,
                           data_range=None,
                           overwrite=True)

pipeline.add_module(module)

pipeline.run()

#module = SimplexMinimizationModule(position=position,
#                                   magnitude=10.,
#                                   psf_scaling=-psf_scaling,
#                                   name_in=f'simplex',
#                                   image_in_tag='science_crop',
#                                   psf_in_tag='flux_crop',
#                                   res_out_tag=f'simplex',
#                                   flux_position_tag=f'fluxpos',
#                                   merit='gaussian',
#                                   aperture=fwhm,
#                                   sigma=0.,
#                                   tolerance=0.01,
#                                   pca_number=range(1, 11),
#                                   cent_size=cent_size,
#                                   edge_size=edge_size,
#                                   extra_rot=extra_rot,
#                                   residuals='mean',
#                                   offset=3.)
#
#pipeline.add_module(module)
#
#module = FitsWritingModule(name_in=f'write2',
#                           data_tag=f'simplex{pca_number:03.0f}',
#                           file_name=f'simplex{pca_number:03.0f}.fits',
#                           output_dir=None,
#                           data_range=None,
#                           overwrite=True)
#
#pipeline.add_module(module)

# pipeline.run_module(f'read1')
# pipeline.run_module(f'read2')
# pipeline.run_module(f'parang')
# pipeline.run_module(f'cut1')
# pipeline.run_module(f'cut2')
# pipeline.run_module(f'prep')
# pipeline.run_module(f'pca')
# pipeline.run_module(f'write1')
# pipeline.run_module(f'simplex')
# pipeline.run_module(f'write2')

#module = Hdf5WritingModule(name_in='write_hdf5',
#                           file_name='tyc_irdis_bks.hdf5',
#                           output_dir=None,
#                           tag_dictionary={'science_crop': 'science_crop',
#                                           'flux_crop': 'flux_crop',
#                                           f'fluxpos{pca_number:03.0f}': 'fluxpos'},
#                           keep_attributes=True,
#                           overwrite=True)
#
#pipeline.add_module(module)
# pipeline.run_module('write_hdf5')
