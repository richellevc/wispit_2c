from pynpoint import *

fwhm = 4.*0.01225  # (arcsec)
cent_size = fwhm  # (arcsec)
edge_size = 3.  # (arcsec)
extra_rot = -134.24  # (deg)
aperture = 5.*0.01225  # (arcsec)
psf_scaling = 6.9*76.42
pca_number = 5
position = (65, 29)
module = SimplexMinimizationModule(position=position,
                                   magnitude=10.,
                                   psf_scaling=-psf_scaling,
                                   name_in='simplex',
                                   image_in_tag='science_crop',
                                   psf_in_tag='flux_crop',
                                   res_out_tag='simplex',
                                   flux_position_tag='fluxpos',
                                   merit='gaussian',
                                   aperture=fwhm,
                                   sigma=0.,
                                   tolerance=0.01,
                                   pca_number=range(1, 11),
                                   cent_size=cent_size,
                                   edge_size=edge_size,
                                   extra_rot=extra_rot,
                                   residuals='mean',
                                   offset=3.)

# note: set offset to 1 or 2 for 1c