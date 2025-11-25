import warnings

import math
import sys

from typing import Union, Tuple

import numpy as np
from photutils import aperture_photometry, CircularAperture
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import minimize
from skimage.feature import hessian_matrix
from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint import FakePlanetModule, DerotateAndStackModule, DarkCalibrationModule, \
    PSFpreparationModule, PcaPsfSubtractionModule
from pynpoint.util.analysis import fake_planet, merit_function
from pynpoint.util.image import crop_image, center_subpixel, cartesian_to_polar, create_mask, \
    rotate_images, rotate_coordinates
from astropy.modeling import models, fitting
from scipy.ndimage import rotate

# from pynpoint.util.module import rotate_coordinates
from pynpoint.util.residuals import combine_residuals

from matplotlib import pyplot as plt

class SimplexMinimizationModule_background(ProcessingModule):
    """
    Pipeline module to measure the flux and position of a planet by injecting negative fake planets
    and minimizing a figure of merit.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 psf_in_tag: str,
                 res_out_tag: str,
                 flux_position_tag: str,
                 position: Tuple[int, int],
                 magnitude: float,
                 psf_scaling: float = -1.,
                 merit: str = 'hessian',
                 aperture: float = 0.1,
                 sigma: float = 0.0,
                 tolerance: float = 0.1,
                 extra_rot: float = 0.,
                 residuals: str = 'median') -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the science images that are read as input.
        psf_in_tag : str
            Tag of the database entry with the reference PSF that is used as fake planet. Can be
            either a single image or a stack of images equal in size to ``image_in_tag``.
        res_out_tag : str
            Tag of the database entry with the image residuals that are written as output. The
            residuals are stored for each step of the minimization. The last image contains the
            best-fit residuals.
        flux_position_tag : str
            Tag of the database entry with the flux and position results that are written as output.
            Each step of the minimization stores the x position (pix), y position (pix), separation
            (arcsec), angle (deg), contrast (mag), and the chi-square value. The last row contains
            the best-fit results.
        position : tuple(int, int)
            Approximate position (x, y) of the planet (pix). This is also the location where the
            figure of merit is calculated within an aperture of radius ``aperture``.
        magnitude : float
            Approximate magnitude of the planet relative to the star.
        psf_scaling : float
            Additional scaling factor of the planet flux (e.g., to correct for a neutral density
            filter). Should be negative in order to inject negative fake planets.
        merit : str
            Figure of merit for the minimization. Can be 'hessian', to minimize the sum of the
            absolute values of the determinant of the Hessian matrix, or 'poisson', to minimize the
            sum of the absolute pixel values, assuming a Poisson distribution for the noise
            (Wertz et al. 2017), or 'gaussian', to minimize the ratio of the squared pixel values
            and the variance of the pixels within an annulus but excluding the aperture area.
        aperture : float
            Aperture radius (arcsec) at the position specified at *position*.
        sigma : float
            Standard deviation (arcsec) of the Gaussian kernel which is used to smooth the images
            before the figure of merit is calculated (in order to reduce small pixel-to-pixel
            variations).
        tolerance : float
            Absolute error on the input parameters, position (pix) and contrast (mag), that is used
            as acceptance level for convergence. Note that only a single value can be specified
            which is used for both the position and flux so tolerance=0.1 will give a precision of
            0.1 mag and 0.1 pix. The tolerance on the output (i.e., the chi-square value) is set to
            np.inf so the condition is always met.
        extra_rot : float
            Additional rotation angle of the images in clockwise direction (deg).
        residuals : str
            Method for combining the residuals ('mean', 'median', 'weighted', or 'clipped').

        Returns
        -------
        NoneType
            None
        """

        super(SimplexMinimizationModule_background, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)

        self.m_res_out_port = self.add_output_port(res_out_tag)
        self.m_flux_position_port = self.add_output_port(flux_position_tag)

        self.m_position = position
        self.m_magnitude = magnitude
        self.m_psf_scaling = psf_scaling
        self.m_merit = merit
        self.m_aperture = aperture
        self.m_sigma = sigma
        self.m_tolerance = tolerance
        self.m_extra_rot = extra_rot
        self.m_residuals = residuals

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. The position and contrast of a planet is measured by injecting
        negative copies of the PSF template and applying a simplex method (Nelder-Mead) for
        minimization of a figure of merit at the planet location.

        Returns
        -------
        NoneType
            None
        """

        self.m_res_out_port.del_all_data()
        self.m_res_out_port.del_all_attributes()

        self.m_flux_position_port.del_all_data()
        self.m_flux_position_port.del_all_attributes()

        parang = self.m_image_in_port.get_attribute('PARANG')
        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')

        aperture = (self.m_position[1], self.m_position[0], self.m_aperture/pixscale)

        self.m_sigma /= pixscale

        psf = self.m_psf_in_port.get_all()
        images = self.m_image_in_port.get_all()

        if psf.shape[0] != 1 and psf.shape[0] != images.shape[0]:
            raise ValueError('The number of frames in psf_in_tag does not match with the number '
                             'of frames in image_in_tag. The DerotateAndStackModule can be '
                             'used to average the PSF frames (without derotating) before applying '
                             'the SimplexMinimizationModule.')

        center = center_subpixel(psf)

        def _objective(arg):
            sys.stdout.write('.')
            sys.stdout.flush()

            pos_y = arg[0]
            pos_x = arg[1]
            mag = arg[2]

            sep_ang = cartesian_to_polar(center, pos_y, pos_x)

            fake = fake_planet(images=images,
                               psf=psf,
                               parang=parang,
                               position=(sep_ang[0], sep_ang[1]),
                               magnitude=mag,
                               psf_scaling=self.m_psf_scaling)

            im_res_derot = rotate_images(images=fake,
                                         angles=-1.*parang+self.m_extra_rot)

            res_stack = combine_residuals(method=self.m_residuals,
                                          res_rot=im_res_derot,
                                          residuals=fake,
                                          angles=parang)

            self.m_res_out_port.append(res_stack, data_dim=3)

            chi_square = merit_function(residuals=res_stack[0, ],
                                        merit=self.m_merit,
                                        aperture=aperture,
                                        sigma=self.m_sigma)

            position = rotate_coordinates(center, (pos_y, pos_x), -self.m_extra_rot)

            res = np.asarray([position[1],
                              position[0],
                              sep_ang[0]*pixscale,
                              (sep_ang[1]-self.m_extra_rot) % 360.,
                              mag,
                              chi_square])

            self.m_flux_position_port.append(res, data_dim=2)

            return chi_square

        sys.stdout.write('Running SimplexMinimizationModule_background')
        sys.stdout.flush()

        pos_init = rotate_coordinates(center,
                                      (self.m_position[1], self.m_position[0]),  # (y, x)
                                      self.m_extra_rot)

        minimize(fun=_objective,
                 x0=[pos_init[0], pos_init[1], self.m_magnitude],
                 method='Nelder-Mead',
                 tol=None,
                 options={'xatol': self.m_tolerance, 'fatol': float('inf')})

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

        history = f'merit = {self.m_merit}'
        self.m_flux_position_port.copy_attributes(self.m_image_in_port)
        self.m_flux_position_port.add_history('SimplexMinimizationModule_background', history)

        self.m_res_out_port.copy_attributes(self.m_image_in_port)
        self.m_res_out_port.add_history('SimplexMinimizationModule_background', history)
        self.m_res_out_port.close_port()


class SimplexMinimizationModule_rdi(ProcessingModule):
    """
    Module to measure the flux and position of a planet by injecting negative fake planets and
    minimizing a function of merit.
    """

    def __init__(self,
                 position,
                 magnitude,
                 psf_scaling=-1.,
                 name_in="simplex",
                 image_in_tag="im_arr",
                 reference_in_tag="ref_arr",
                 psf_in_tag="im_psf",
                 res_out_tag="simplex_res",
                 flux_position_tag="flux_position",
                 merit="hessian",
                 aperture=0.1,
                 sigma=0.027,
                 tolerance=0.1,
                 pca_number=20,
                 norm=False,
                 cent_size=None,
                 edge_size=None,
                 extra_rot=0.,
                 subtract_mean=False):
        """
        Constructor of SimplexMinimizationModule.

        :param position: Approximate position (x, y) of the planet (pix). This is also the location
                         where the function of merit is calculated with an aperture of radius
                         *aperture*.
        :type position: (int, int)
        :param magnitude: Approximate magnitude of the planet relative to the star.
        :type magnitude: float
        :param psf_scaling: Additional scaling factor of the planet flux (e.g., to correct for a
                            neutral density filter). Should be negative in order to inject
                            negative fake planets.
        :type psf_scaling: float
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry with images that are read as input.
        :type image_in_tag: str
        :param psf_in_tag: Tag of the database entry with the reference PSF that is used as fake
                           planet. Can be either a single image (2D) or a cube (3D) with the
                           dimensions equal to *image_in_tag*.
        :type psf_in_tag: str
        :param res_out_tag: Tag of the database entry with the image residuals that are written
                            as output. Contains the results from the PSF subtraction during the
                            minimization of the function of merit. The last image is the image
                            with the best-fit residuals.
        :type res_out_tag: str
        :param flux_position_tag: Tag of the database entry with flux and position results that are
                                  written as output. Each step of the minimization saves the
                                  x position (pix), y position (pix), separation (arcsec),
                                  angle (deg), contrast (mag), and the function of merit. The last
                                  row of values contain the best-fit results.
        :type flux_position_tag: str
        :param merit: Function of merit for the minimization. Can be either *hessian*, to minimize
                      the sum of the absolute values of the determinant of the Hessian matrix,
                      or *sum*, to minimize the sum of the absolute pixel values
                      (Wertz et al. 2017).
        :type merit: str
        :param aperture: Aperture radius (arcsec) used for the minimization at *position*.
        :type aperture: float
        :param sigma: Standard deviation (arcsec) of the Gaussian kernel which is used to smooth
                      the images before the function of merit is calculated (in order to reduce
                      small pixel-to-pixel variations). Highest astrometric and photometric
                      precision is achieved when sigma is optimized.
        :type sigma: float
        :param tolerance: Absolute error on the input parameters, position (pix) and
                          contrast (mag), that is used as acceptance level for convergence. Note
                          that only a single value can be specified which is used for both the
                          position and flux so tolerance=0.1 will give a precision of 0.1 mag
                          and 0.1 pix. The tolerance on the output (i.e., function of merit)
                          is set to np.inf so the condition is always met.
        :type tolerance: float
        :param pca_number: Number of principal components used for the PSF subtraction.
        :type pca_number: int
        :param cent_size: Radius of the central mask (arcsec). No mask is used when set to None.
        :type cent_size: float
        :param edge_size: Outer radius (arcsec) beyond which pixels are masked. No outer mask is
                          used when set to None. The radius will be set to half the image size if
                          the *edge_size* value is larger than half the image size.
        :type edge_size: float
        :param extra_rot: Additional rotation angle of the images in clockwise direction (deg).
        :type extra_rot: float

        :return: None
        """

        super(SimplexMinimizationModule_rdi, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        if reference_in_tag == image_in_tag:
            self.m_reference_in_port = self.m_image_in_port
        else:
            self.m_reference_in_port = self.add_input_port(reference_in_tag)
        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)
        self.m_res_out_port = self.add_output_port(res_out_tag)
        self.m_flux_position_port = self.add_output_port(flux_position_tag)

        self.m_position = (int(position[1]), int(position[0]))
        self.m_magnitude = magnitude
        self.m_psf_scaling = psf_scaling
        self.m_merit = merit
        self.m_aperture = aperture
        self.m_sigma = sigma
        self.m_tolerance = tolerance
        self.m_pca_number = pca_number
        self.m_norm = norm
        self.m_cent_size = cent_size
        self.m_edge_size = edge_size
        self.m_extra_rot = extra_rot
        self.m_subtract_mean = subtract_mean

        self.m_image_in_tag = image_in_tag
        self.m_reference_in_tag = reference_in_tag
        self.m_psf_in_tag = psf_in_tag

    def run(self):
        """
        Run method of the module. The position and flux of a planet are measured by injecting
        negative fake companions and applying a simplex method (Nelder-Mead) for minimization
        of a function of merit at the planet location. The default function of merit is the
        image curvature which is calculated as the sum of the absolute values of the
        determinant of the Hessian matrix.

        :return: None
        """

        def _objective(arg):
            sys.stdout.write('.')
            sys.stdout.flush()

            pos_y = arg[0]
            pos_x = arg[1]
            mag = arg[2]

            sep, ang = cartesian_to_polar(center=center,
                                          y_pos=pos_y,
                                          x_pos=pos_x)

            fake_planet = FakePlanetModule(position=(sep*pixscale, ang),
                                           magnitude=mag,
                                           psf_scaling=self.m_psf_scaling,
                                           interpolation="spline",
                                           name_in="fake_planet",
                                           image_in_tag=self.m_image_in_tag,
                                           psf_in_tag=self.m_psf_in_tag,
                                           image_out_tag="simplex_fake")

            fake_planet.connect_database(self._m_data_base)
            fake_planet.run()

            prep = PSFpreparationModule(name_in="prep",
                                        image_in_tag="simplex_fake",
                                        image_out_tag="simplex_rdi_prep",
                                        mask_out_tag=None,
                                        norm=self.m_norm,
                                        resize=None,
                                        cent_size=self.m_cent_size,
                                        edge_size=self.m_edge_size)

            prep.connect_database(self._m_data_base)
            prep.run()

            if self.m_reference_in_tag == self.m_image_in_tag:

                psf_sub = PcaPsfSubtractionModule(pca_numbers=self.m_pca_number,
                                                  name_in="pca_simplex",
                                                  images_in_tag="simplex_rdi_prep",
                                                  reference_in_tag="simplex_rdi_prep",
                                                  res_mean_tag="simplex_res_mean",
                                                  res_median_tag=None,
                                                  res_arr_out_tag="simplex_res_arr_",
                                                  res_rot_mean_clip_tag=None,
                                                  extra_rot=self.m_extra_rot,
                                                  subtract_mean=self.m_subtract_mean)

                psf_sub.connect_database(self._m_data_base)
                psf_sub.run()

            else:

                ref_prep = PSFpreparationModule(name_in="ref_prep",
                                                image_in_tag=self.m_reference_in_tag,
                                                image_out_tag="ref_simplex_rdi_prep",
                                                mask_out_tag=None,
                                                norm=self.m_norm,
                                                resize=None,
                                                cent_size=self.m_cent_size,
                                                edge_size=self.m_edge_size)

                ref_prep.connect_database(self._m_data_base)
                ref_prep.run()

                psf_sub = PcaPsfSubtractionModule(pca_numbers=[self.m_pca_number],
                                                  name_in="rdipca_simplex",
                                                  images_in_tag="simplex_rdi_prep",
                                                  reference_in_tag="ref_simplex_rdi_prep",
                                                  res_mean_tag="simplex_rdi_res_mean",
                                                  res_median_tag=None,
                                                  res_arr_out_tag="simplex_res_arr_",
                                                  res_rot_mean_clip_tag=None,
                                                  extra_rot=self.m_extra_rot,
                                                  subtract_mean=self.m_subtract_mean)

                psf_sub.connect_database(self._m_data_base)
                psf_sub.run()

            res_input_port = self.add_input_port("simplex_res_mean")
            im_res = res_input_port.get_all()

            if len(im_res.shape) == 3:
                if im_res.shape[0] == 1:
                    im_res = np.squeeze(im_res, axis=0)
                else:
                    raise ValueError("Multiple residual images found, expecting only one.")

            self.m_res_out_port.append(im_res, data_dim=3)

            im_crop = crop_image(image=im_res,
                                 center=self.m_position,
                                 size=2*int(math.ceil(self.m_aperture)))

            npix = im_crop.shape[0]

            if self.m_merit == "hessian":

                x_grid = y_grid = np.linspace(-(npix-1)/2, (npix-1)/2, npix)
                xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
                rr_grid = np.sqrt(xx_grid*xx_grid+yy_grid*yy_grid)

                hessian_rr, hessian_rc, hessian_cc = hessian_matrix(im_crop,
                                                                    sigma=self.m_sigma,
                                                                    mode='constant',
                                                                    cval=0.,
                                                                    order='rc')

                hes_det = (hessian_rr*hessian_cc) - (hessian_rc*hessian_rc)
                hes_det[rr_grid > self.m_aperture] = 0.
                merit = np.sum(np.abs(hes_det))

            elif self.m_merit == "sum":

                if self.m_sigma > 0.:
                    im_crop = gaussian_filter(input=im_crop, sigma=self.m_sigma)

                aperture = CircularAperture((npix/2., npix/2.), self.m_aperture)
                phot_table = aperture_photometry(np.abs(im_crop), aperture, method='exact')
                merit = phot_table['aperture_sum']

            else:
                raise ValueError("Function of merit not recognized.")

            position = rotate_coordinates(center, (pos_y, pos_x), -self.m_extra_rot)

            res = np.asarray((position[1],
                              position[0],
                              sep,
                              (ang-self.m_extra_rot)%360.,
                              mag,
                              merit))

            self.m_flux_position_port.append(res, data_dim=2)

            return merit

        self.m_res_out_port.del_all_data()
        self.m_res_out_port.del_all_attributes()

        self.m_flux_position_port.del_all_data()
        self.m_flux_position_port.del_all_attributes()

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        self.m_aperture /= pixscale
        self.m_sigma /= pixscale

        psf_size = self.m_psf_in_port.get_shape()[1:]
        center = (psf_size[0]/2., psf_size[1]/2.)

        sys.stdout.write("Running SimplexMinimizationModule_rdi")
        sys.stdout.flush()

        pos_init = rotate_coordinates(center, self.m_position, self.m_extra_rot)
        pos_init = (int(pos_init[0]), int(pos_init[1])) # (y, x)

        minimize(fun=_objective,
                 x0=[pos_init[0], pos_init[1], self.m_magnitude],
                 method="Nelder-Mead",
                 tol=None,
                 options={'xatol': self.m_tolerance, 'fatol': float("inf")})

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()

        self.m_res_out_port.add_history("SimplexMinimizationModule_rdi",
                                                    "Merit function = "+str(self.m_merit))

        self.m_flux_position_port.add_history("SimplexMinimizationModule_rdi",
                                                          "Merit function = "+str(self.m_merit))

        self.m_res_out_port.copy_attributes(self.m_image_in_port)
        self.m_flux_position_port.copy_attributes(self.m_image_in_port)

        self.m_res_out_port.close_port()


class SimplexMinimizationModule_cADI(ProcessingModule):
    """
    Module to measure the flux and position of a planet by injecting negative fake planets and
    minimizing a function of merit.
    """

    def __init__(self,
                 position,
                 magnitude,
                 psf_scaling=-1.,
                 name_in="simplex",
                 image_in_tag="im_arr",
                 psf_in_tag="im_psf",
                 res_out_tag="simplex_res",
                 flux_position_tag="flux_position",
                 merit="hessian",
                 aperture=0.1,
                 sigma=0.027,
                 tolerance=0.1,
                 pca_number=20,
                 cent_size=None,
                 edge_size=None,
                 extra_rot=0.):
        """
        Constructor of SimplexMinimizationModule_cADI.

        :param position: Approximate position (x, y) of the planet (pix). This is also the location
                         where the function of merit is calculated with an aperture of radius
                         *aperture*.
        :type position: (int, int)
        :param magnitude: Approximate magnitude of the planet relative to the star.
        :type magnitude: float
        :param psf_scaling: Additional scaling factor of the planet flux (e.g., to correct for a
                            neutral density filter). Should be negative in order to inject
                            negative fake planets.
        :type psf_scaling: float
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry with images that are read as input.
        :type image_in_tag: str
        :param psf_in_tag: Tag of the database entry with the reference PSF that is used as fake
                           planet. Can be either a single image (2D) or a cube (3D) with the
                           dimensions equal to *image_in_tag*.
        :type psf_in_tag: str
        :param res_out_tag: Tag of the database entry with the image residuals that are written
                            as output. Contains the results from the PSF subtraction during the
                            minimization of the function of merit. The last image is the image
                            with the best-fit residuals.
        :type res_out_tag: str
        :param flux_position_tag: Tag of the database entry with flux and position results that are
                                  written as output. Each step of the minimization saves the
                                  x position (pix), y position (pix), separation (arcsec),
                                  angle (deg), contrast (mag), and the function of merit. The last
                                  row of values contain the best-fit results.
        :type flux_position_tag: str
        :param merit: Function of merit for the minimization. Can be either *hessian*, to minimize
                      the sum of the absolute values of the determinant of the Hessian matrix,
                      or *sum*, to minimize the sum of the absolute pixel values
                      (Wertz et al. 2017).
        :type merit: str
        :param aperture: Aperture radius (arcsec) used for the minimization at *position*.
        :type aperture: float
        :param sigma: Standard deviation (arcsec) of the Gaussian kernel which is used to smooth
                      the images before the function of merit is calculated (in order to reduce
                      small pixel-to-pixel variations). Highest astrometric and photometric
                      precision is achieved when sigma is optimized.
        :type sigma: float
        :param tolerance: Absolute error on the input parameters, position (pix) and
                          contrast (mag), that is used as acceptance level for convergence. Note
                          that only a single value can be specified which is used for both the
                          position and flux so tolerance=0.1 will give a precision of 0.1 mag
                          and 0.1 pix. The tolerance on the output (i.e., function of merit)
                          is set to np.inf so the condition is always met.
        :type tolerance: float
        :param pca_number: Number of principal components used for the PSF subtraction.
        :type pca_number: int
        :param cent_size: Radius of the central mask (arcsec). No mask is used when set to None.
        :type cent_size: float
        :param edge_size: Outer radius (arcsec) beyond which pixels are masked. No outer mask is
                          used when set to None. The radius will be set to half the image size if
                          the *edge_size* value is larger than half the image size.
        :type edge_size: float
        :param extra_rot: Additional rotation angle of the images in clockwise direction (deg).
        :type extra_rot: float

        :return: None
        """

        super(SimplexMinimizationModule_cADI, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)
        self.m_res_out_port = self.add_output_port(res_out_tag)
        self.m_flux_position_port = self.add_output_port(flux_position_tag)

        self.m_position = (int(position[1]), int(position[0]))
        self.m_magnitude = magnitude
        self.m_psf_scaling = psf_scaling
        self.m_merit = merit
        self.m_aperture = aperture
        self.m_sigma = sigma
        self.m_tolerance = tolerance
        self.m_pca_number = pca_number
        self.m_cent_size = cent_size
        self.m_edge_size = edge_size
        self.m_extra_rot = extra_rot

        self.m_image_in_tag = image_in_tag
        self.m_psf_in_tag = psf_in_tag

    def run(self):
        """
        Run method of the module. The position and flux of a planet are measured by injecting
        negative fake companions and applying a simplex method (Nelder-Mead) for minimization
        of a function of merit at the planet location. The default function of merit is the
        image curvature which is calculated as the sum of the absolute values of the
        determinant of the Hessian matrix.

        :return: None
        """

        def _objective(arg):
            sys.stdout.write('.')
            sys.stdout.flush()

            pos_y = arg[0]
            pos_x = arg[1]
            mag = arg[2]

            sep = math.sqrt((pos_y-center[0])**2+(pos_x-center[1])**2)*pixscale
            ang = math.atan2(pos_y-center[0], pos_x-center[1])*180./math.pi - 90.

            fake_planet = FakePlanetModule(position=(sep, ang),
                                           magnitude=mag,
                                           psf_scaling=self.m_psf_scaling,
                                           interpolation="spline",
                                           name_in="fake_planet",
                                           image_in_tag=self.m_image_in_tag,
                                           psf_in_tag=self.m_psf_in_tag,
                                           image_out_tag="simplex_fake",
                                           verbose=False)

            fake_planet.connect_database(self._m_data_base)
            fake_planet.run()

            prep = PSFpreparationModule(name_in="prep",
                                        image_in_tag="simplex_fake",
                                        image_out_tag="simplex_prep",
                                        image_mask_out_tag=None,
                                        mask_out_tag=None,
                                        norm=False,
                                        cent_size=self.m_cent_size,
                                        edge_size=self.m_edge_size,
                                        verbose=False)

            prep.connect_database(self._m_data_base)
            prep.run()

            # --- cADI of full frame

            create_median_cADI = DerotateAndStackModule(name_in="create_median_cADI",
                                                        image_in_tag="simplex_prep",
                                                        image_out_tag="simplex_im_median_cADI",
                                                        derotate=False,
                                                        stack="mean")

            create_median_cADI.connect_database(self._m_data_base)
            create_median_cADI.run()

            subtract_im_median_cADI = DarkCalibrationModule(name_in="subtract_im_median_cADI",
                                                            image_in_tag="simplex_prep",
                                                            dark_in_tag="simplex_im_median_cADI",
                                                            image_out_tag="simplex_im_arr_sub_cADI")

            subtract_im_median_cADI.connect_database(self._m_data_base)
            subtract_im_median_cADI.run()

            derotate_and_median_cADI = DerotateAndStackModule(name_in="derotate_and_median_cADI",
                                                              image_in_tag="simplex_im_arr_sub_cADI",
                                                              image_out_tag="simplex_res_median_cADI",
                                                              derotate=True,
                                                              stack="median",
                                                              extra_rot=self.m_extra_rot)

            derotate_and_median_cADI.connect_database(self._m_data_base)
            derotate_and_median_cADI.run()

            res_input_port = self.add_input_port("simplex_res_median_cADI")
            im_res = res_input_port.get_all()

            if len(im_res.shape) == 3:
                if im_res.shape[0] == 1:
                    im_res = np.squeeze(im_res, axis=0)
                else:
                    raise ValueError("Multiple residual images found, expecting only one.")

            self.m_res_out_port.append(im_res, data_dim=3)

            im_crop = crop_image(image=im_res,
                                 center=self.m_position,
                                 size=2*int(math.ceil(self.m_aperture)))

            npix = im_crop.shape[0]

            if self.m_merit == "hessian":

                x_grid = y_grid = np.linspace(-(npix-1)/2, (npix-1)/2, npix)
                xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
                rr_grid = np.sqrt(xx_grid*xx_grid+yy_grid*yy_grid)

                hessian_rr, hessian_rc, hessian_cc = hessian_matrix(im_crop,
                                                                    sigma=self.m_sigma,
                                                                    mode='constant',
                                                                    cval=0.,
                                                                    order='rc')

                hes_det = (hessian_rr*hessian_cc) - (hessian_rc*hessian_rc)
                hes_det[rr_grid > self.m_aperture] = 0.
                merit = np.sum(np.abs(hes_det))

            elif self.m_merit == "sum":

                if self.m_sigma > 0.:
                    im_crop = gaussian_filter(input=im_crop, sigma=self.m_sigma)

                aperture = CircularAperture((npix/2., npix/2.), self.m_aperture)
                phot_table = aperture_photometry(np.abs(im_crop), aperture, method='exact')
                merit = phot_table['aperture_sum']

            else:
                raise ValueError("Function of merit not recognized.")

            position = rotate_coordinates(center, (pos_y, pos_x), -self.m_extra_rot)

            res = np.asarray((position[1],
                              position[0],
                              sep,
                              (ang-self.m_extra_rot)%360.,
                              mag,
                              merit))

            self.m_flux_position_port.append(res, data_dim=2)

            return merit

        self.m_res_out_port.del_all_data()
        self.m_res_out_port.del_all_attributes()

        self.m_flux_position_port.del_all_data()
        self.m_flux_position_port.del_all_attributes()

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        self.m_aperture /= pixscale
        self.m_sigma /= pixscale

        psf_size = self.m_psf_in_port.get_shape()[1:]
        center = (psf_size[0]/2., psf_size[1]/2.)

        sys.stdout.write("Running SimplexMinimizationModule_cADI")
        sys.stdout.flush()

        pos_init = rotate_coordinates(center, self.m_position, self.m_extra_rot)
        pos_init = (int(pos_init[0]), int(pos_init[1])) # (y, x)

        minimize(fun=_objective,
                 x0=[pos_init[0], pos_init[1], self.m_magnitude],
                 method="Nelder-Mead",
                 tol=None,
                 options={'xatol': self.m_tolerance, 'fatol': float("inf")})

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()

        self.m_res_out_port.add_history_information("SimplexMinimizationModule_cADI",
                                                    "Merit function = "+str(self.m_merit))

        self.m_flux_position_port.add_history_information("SimplexMinimizationModule_cADI",
                                                          "Merit function = "+str(self.m_merit))

        self.m_res_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_flux_position_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_res_out_port.close_port()

