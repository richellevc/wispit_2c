"""
Adjusted Pypline Modules used for reduction.
"""

import sys

import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


import skimage.draw as draw

import os
import time
import warnings

from typing import Any, List, Optional, Tuple, Union
from multiprocessing import Pool

import numpy as np
import emcee

from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.analysis import (
    pixel_variance,
)
from pynpoint.util.image import (
    create_mask,
    polar_to_cartesian,
)
from pynpoint.util.mcmc import lnprob
from scripts.mcmc import lnprob_rdi # edited to do rdi
from pynpoint.util.module import progress, memory_frames



class UnsharpMaskModule(ProcessingModule):
    """
    Pipeline module to apply unsharp masking to the input images. Negative pixels are clipped at -50, and the center
    is optionally masked.
    """

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: Optional[str],
                 image_out_tag: Optional[str],
                 kernel_size: float = 5.,
                 cent_size: Optional[float] = None) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str, None
            Tag of the database entry with the input images that are read as input. Not read if set
            to None.
        image_out_tag : str, None
            Tag of the database entry with the output images that are written as output. Not written
            if set to None.
        kernel_size : int
            Size of the Gaussian kernel (in pixels) for the unsharp masking.
        cent_size : float, None, optional
            Radius of the central mask (in arcsec). No mask is used when set to None.
        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        # Variables
        self.m_kernel_size = kernel_size
        self.m_cent_size = cent_size

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    @typechecked
    def run(self) -> None:
        """Run method of the module. Applies a Gaussian filter and a central mask to the spatial dimensions of the
        images. Negative pixels are clipped at -50.

        Returns
        -------
        NoneType
            None
        """

        # Get the PIXSCALE and MEMORY attributes
        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        memory = self._m_config_port.get_attribute('MEMORY')

        # Get the numnber of dimensions and shape
        ndim = self.m_image_in_port.get_ndim()
        im_shape = self.m_image_in_port.get_shape()

        # Convert m_cent_size from arcseconds to pixels
        if self.m_cent_size is not None:
            self.m_cent_size /= pixscale

        nimages = self.m_image_in_port.get_shape()[0]
        frames = memory_frames(memory, nimages)

        sigma = self.m_kernel_size

        # Create 2D disk mask which will be applied to the image before USM to mask bright central region (iwa)
        mask = create_mask((int(im_shape[-2]), int(im_shape[-1])),
                           (self.m_cent_size, None)).astype(bool)

        start_time = time.time()

        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), 'Applying Gaussian filter...', start_time)

            images = self.m_image_in_port[frames[i]:frames[i+1], ]
            im_filter = gaussian_filter(images, (0, sigma, sigma))
            images_usm = images - im_filter
            images_usm[:, ~mask] = 0.
            images_usm[images_usm < -50.] = -50.
            self.m_image_out_port.append(images_usm, data_dim=3)

        history = f'Gaussian kernel size (pixels): {self.m_kernel_size}'
        self.m_image_out_port.add_history("Unsharp Masking", history)
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.close_port()


class EllipseMaskModule(ProcessingModule):
    """
    Pipeline module for applying a (coronagraph) transmission correction to the input images. For a given
    input transmission curve, the module scales the flux of the input images by 1.0 / transmission.
    """

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 epoch: str,
                 gap: Optional[bool] = True,
                 both_gaps: Optional[bool] = False,
                 mask_out_tag: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the input images that are read as input.
        image_out_tag : str
            Tag of the database entry with the output images that are written as output.
        mask_out_tag : str, None, optional
            Tag of the database entry with the mask that is written as output. If set to None, no
            mask array is saved.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        # Variables
        self.m_gap = gap
        self.m_epoch = epoch
        self.m_both_gaps = both_gaps

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        if mask_out_tag is None:
            self.m_mask_out_port = None
        else:
            self.m_mask_out_port = self.add_output_port(mask_out_tag)


    @typechecked
    def run(self) -> None:
        """
        Run method of module, scales image flux by 1/transmission.
        """

        # Get the PIXSCALE and MEMORY attributes
        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        memory = self._m_config_port.get_attribute('MEMORY')

        # Get the numnber of dimensions and shape
        ndim = self.m_image_in_port.get_ndim()
        im_shape = self.m_image_in_port.get_shape()

        if ndim == 3:
            # Number of images
            nimages = im_shape[-3]

            # Split into batches to comply with memory constraints
            frames = memory_frames(memory, nimages)

        elif ndim == 4:
            # Process all wavelengths per exposure at once
            frames = np.linspace(0, im_shape[-3], im_shape[-3] + 1)

        # Create 2D disk mask which will be applied to every frame
        mask = np.ones((int(im_shape[-2]), int(im_shape[-1])), dtype=bool)
        if self.m_epoch == '2023-10-19':
            rot_shift = -105.58
        elif self.m_epoch == '2024-10-04':
            rot_shift = -108.13
        elif self.m_epoch == '2025-03-21':
            rot_shift = 17.3
        else:
            print(f'epoch not recognized: {self.m_epoch}')
            sys.exit()
        # outer gap
        ellipse_outer = draw.ellipse(100, 100, 48, 33, rotation=np.radians(3 + rot_shift))
        mask[ellipse_outer] = 0
        ellipse_inner = draw.ellipse(100, 100, 27, 19, rotation=np.radians(2 + rot_shift))
        mask[ellipse_inner] = 1
        if self.m_both_gaps:
            # inner
            ellipse_outer_inner = draw.ellipse(100, 100, 15, 10, rotation=np.radians(1.5 + rot_shift))
            mask[ellipse_outer_inner] = 0
            ellipse_inner_inner = draw.ellipse(100, 100, 5, 5, rotation=np.radians(2 + rot_shift))
            mask[ellipse_inner_inner] = 1

        start_time = time.time()
        # Run the TransmissionModule for each subset of frames
        for i in range(frames[:-1].size):
            # Print progress to command line
            progress(i, len(frames[:-1]), 'Applying ellipse mask to frame...', start_time)

            if ndim == 3:
                # Get the images and ensure they have the correct 3D shape with the following
                # three dimensions: (batch_size, height, width)
                images = self.m_image_in_port[frames[i]:frames[i + 1], ]

                if images.ndim == 2:
                    warnings.warn('The input data has 2 dimensions whereas 3 dimensions are '
                                  'required. An extra dimension has been added.')
                    images = images[np.newaxis, ...]

            elif ndim == 4:
                # Process all wavelengths per exposure at once
                images = self.m_image_in_port[:, i, ]

            # Apply the mask
            if self.m_gap:
                images[:, mask] = 0.
            else:
                images[:, ~mask] = 0.

            # Write processed images to output port
            if ndim == 3:
                self.m_image_out_port.append(images, data_dim=3)
            elif ndim == 4:
                self.m_image_out_port.append(images, data_dim=4)

        # Copy attributes of input images
        self.m_image_out_port.copy_attributes(self.m_image_in_port)

        # Store information about mask
        if self.m_mask_out_port is not None:
            self.m_mask_out_port.set_all(mask)
            self.m_mask_out_port.copy_attributes(self.m_image_in_port)


class CircleMaskModule(ProcessingModule):
    """
    Pipeline module for applying a (coronagraph) transmission correction to the input images. For a given
    input transmission curve, the module scales the flux of the input images by 1.0 / transmission.
    """

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 epoch: str,
                 gap: Optional[bool] = True,
                 both_gaps: Optional[bool] = False,
                 mask_out_tag: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the input images that are read as input.
        image_out_tag : str
            Tag of the database entry with the output images that are written as output.
        mask_out_tag : str, None, optional
            Tag of the database entry with the mask that is written as output. If set to None, no
            mask array is saved.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        # Variables
        self.m_gap = gap
        self.m_epoch = epoch
        self.m_both_gaps = both_gaps

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        if mask_out_tag is None:
            self.m_mask_out_port = None
        else:
            self.m_mask_out_port = self.add_output_port(mask_out_tag)


    @typechecked
    def run(self) -> None:
        """
        Run method of module, scales image flux by 1/transmission.
        """

        # Get the PIXSCALE and MEMORY attributes
        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        memory = self._m_config_port.get_attribute('MEMORY')

        # Get the numnber of dimensions and shape
        ndim = self.m_image_in_port.get_ndim()
        im_shape = self.m_image_in_port.get_shape()

        if ndim == 3:
            # Number of images
            nimages = im_shape[-3]

            # Split into batches to comply with memory constraints
            frames = memory_frames(memory, nimages)

        elif ndim == 4:
            # Process all wavelengths per exposure at once
            frames = np.linspace(0, im_shape[-3], im_shape[-3] + 1)

        # Create 2D disk mask which will be applied to every frame
        mask = np.ones((int(im_shape[-2]), int(im_shape[-1])), dtype=bool)
        if self.m_epoch == '2023-10-19':
            rot_shift = -105.58
        elif self.m_epoch == '2024-10-04':
            rot_shift = -108.13
        elif self.m_epoch == '2025-03-21':
            rot_shift = 17.3
        else:
            print(f'epoch not recognized: {self.m_epoch}')
            sys.exit()
        # outer gap
        #ellipse_outer = draw.ellipse(100, 100, 48, 33, rotation=np.radians(3 + rot_shift))
        #mask[ellipse_outer] = 0
        #ellipse_inner = draw.ellipse(100, 100, 27, 19, rotation=np.radians(2 + rot_shift))
        #mask[ellipse_inner] = 1
        if self.m_both_gaps:
            # inner
            ellipse_outer_inner = draw.ellipse(99, 99, 13, 13, rotation=np.radians(1.5 + rot_shift))
            mask[ellipse_outer_inner] = 0
            ellipse_inner_inner = draw.ellipse(99, 99, 5, 5, rotation=np.radians(2 + rot_shift))
            mask[ellipse_inner_inner] = 1

        start_time = time.time()
        # Run the TransmissionModule for each subset of frames
        for i in range(frames[:-1].size):
            # Print progress to command line
            progress(i, len(frames[:-1]), 'Applying ellipse mask to frame...', start_time)

            if ndim == 3:
                # Get the images and ensure they have the correct 3D shape with the following
                # three dimensions: (batch_size, height, width)
                images = self.m_image_in_port[frames[i]:frames[i + 1], ]

                if images.ndim == 2:
                    warnings.warn('The input data has 2 dimensions whereas 3 dimensions are '
                                  'required. An extra dimension has been added.')
                    images = images[np.newaxis, ...]

            elif ndim == 4:
                # Process all wavelengths per exposure at once
                images = self.m_image_in_port[:, i, ]

            # Apply the mask
            if self.m_gap:
                images[:, mask] = 0.
            else:
                images[:, ~mask] = 0.

            # Write processed images to output port
            if ndim == 3:
                self.m_image_out_port.append(images, data_dim=3)
            elif ndim == 4:
                self.m_image_out_port.append(images, data_dim=4)

        # Copy attributes of input images
        self.m_image_out_port.copy_attributes(self.m_image_in_port)

        # Store information about mask
        if self.m_mask_out_port is not None:
            self.m_mask_out_port.set_all(mask)
            self.m_mask_out_port.copy_attributes(self.m_image_in_port)


class MSEModule(ProcessingModule):
    """
    Pipeline module to save most correlated frames based on MSE.
    """

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 reference_in_tag: str,
                 reference_out_tag: str,
                 library_size: int = 200,
                 selection_out_tag: Optional[str] = "None",) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the input images that are read as input. Not read if set
            to None.
        reference_in_tag : str
            Tag of the database entry with the reference images that are read as input.
        reference_out_tag : str
            Tag of the database entry with the lowest MSE output reference images.
        library_size: int
            Number of frames to be saved in the reference library. Default is 200.
        selection_out_tag : str
            Tag of the database entry that stores the boolean selection mask of the frames that are selected
        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        # Variables
        self.m_library_size = library_size

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_reference_in_port = self.add_input_port(reference_in_tag)
        self.m_reference_out_port = self.add_output_port(reference_out_tag)

        self.m_selection_out_port = self.add_output_port(selection_out_tag)

    @typechecked
    def run(self) -> None:
        """Run method of the module. Calculate MSE between the input images and the reference images.
        If the images have dim (n, m), the MSE is calculated over the n and m dimensions, if the images
        have dim (k, n, m), the median of the k images is calculated and the MSE is calculated over the n and m dimensions.

        Returns
        -------
        NoneType
            None
        """

        # Get the MEMORY attributes
        memory = self._m_config_port.get_attribute('MEMORY')

        # Get the numnber of dimensions and shape
        ndim = self.m_image_in_port.get_ndim()
        if ndim == 3:
            image = np.median(self.m_image_in_port[:,], axis=0)
        elif ndim == 2:
            image = self.m_image_in_port[:,]

        cube = self.m_reference_in_port[:,].copy()

        # set image and cube to be np.nan where pixel value is 0, so masked pixels are not included in the MSE calculation
        image = np.where(image == 0, np.nan, image)
        cube = np.where(self.m_reference_in_port[:,] == 0, np.nan, cube)

        # calculate MSE
        diff = cube - image  # will broadcast image to all k frames
        squared_diff = diff ** 2
        mse = np.nanmean(squared_diff, axis=(1, 2))  # mean over n and m for each frame

        # set MSE constraints
        mselim_size = np.sort(mse)[self.m_library_size]
        mselim = mse < mselim_size


        plt.hist(mse[mselim], bins=100)
        plt.show()
        ref_cube_select = self.m_reference_in_port[mselim,]
        index_cube = np.arange(self.m_reference_in_port[:,].shape[0])
        result = np.column_stack((index_cube, mselim))

        history = f'Selected the top {self.m_library_size} most similar ref library frames based on MSE'

        if self.m_selection_out_port != "None":
            self.m_selection_out_port.append(result)

        self.m_reference_out_port.append(ref_cube_select)
        self.m_reference_out_port.copy_attributes(self.m_image_in_port)
        self.m_reference_out_port.add_history("MSE module", history)
        self.m_reference_out_port.close_port()


class MCMCsamplingModule_rdi(ProcessingModule):
    """
    Pipeline module to measure the separation, position angle, and
    contrast of a planet with injection of negative artificial planets
    and sampling of the posterior distribution with ``emcee``, an
    affine invariant Markov chain Monte Carlo (MCMC) ensemble sampler.
    """

    __author__ = "Tomas Stolker"

    @typechecked
    def __init__(
        self,
        name_in: str,
        image_in_tag: str,
        psf_in_tag: str,
        chain_out_tag: str,
        param: Tuple[float, float, float],
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        nwalkers: int = 100,
        nsteps: int = 200,
        psf_scaling: float = -1.0,
        pca_number: int = 20,
        aperture: Union[float, Tuple[int, int, float]] = 0.1,
        mask: Optional[
            Union[
                Tuple[float, float],
                Tuple[None, float],
                Tuple[float, None],
                Tuple[None, None],
            ]
        ] = None,
        extra_rot: float = 0.0,
        merit: str = "gaussian",
        residuals: str = "median",
        reference_in_tag: Optional[str] = None,
        resume: bool = False,
        **kwargs: Union[float, Tuple[float, float, float]],
    ) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Database tag with the science images.
        psf_in_tag : str
            Database tag with the reference PSF that is used as
            artificial planet. The dataset can be either a single
            image, or a stack of images with the dimensions equal
            to ``image_in_tag``.
        chain_out_tag : str
            Database tag were the posterior samples will be stored.
            The shape of the array is ``(nsteps, nwalkers, 3)``. The
            mean acceptance fraction and the integrated autocorrelation
            time are stored as attributes.
        param : tuple(float, float, float)
            The approximate separation (arcsec), angle (deg), and
            contrast (mag), for example obtained with the
            :class:`~pynpoint.processing.fluxposition.SimplexMinimizationModule`.
            The angle is measured in counterclockwise direction with
            respect to the upward direction (i.e., East of North). The
            separation and angle are also used as (fixed) position for the
            aperture if ``aperture`` contains a float (i.e. the radius).
        bounds : tuple(tuple(float, float), tuple(float, float), tuple(float, float))
            The prior boundaries for the separation (arcsec), angle
            (deg), and contrast (mag). Each set of boundaries is
            specified as a tuple.
        nwalkers : int
            Number of walkers.
        nsteps : int
            Number of steps per walker.
        psf_scaling : float
            Additional scaling factor of the planet flux (e.g. to
            correct for a neutral density filter or difference in
            exposure time). The value should be negative in order
            to inject negative fake planets.
        pca_number : int
            Number of principal components used for the PSF
            subtraction.
        aperture : float, tuple(int, int, float)
            Either the aperture radius (arcsec) at the position of
            ``param`` or tuple with the position and aperture radius
            (arcsec) as ``(pos_x, pos_y, radius)``.
        mask : tuple(float, float), None
            Inner and outer mask radius (arcsec) for the PSF
            subtraction. Both elements of the tuple can be set to
            ``None``. Masked pixels are excluded from the PCA
            computation, resulting in a smaller runtime. Masking is
            done after the artificial planet is injected.
        extra_rot : float
            Additional rotation angle of the images (deg).
        merit : str
            Figure of merit for the minimization ('hessian',
            'gaussian', or 'poisson'). Either the determinant of the
            Hessian matrix is minimized ('hessian') or the flux of each
            pixel ('gaussian' or 'poisson'). For the latter case, the
            estimate noise is assumed to follow a Poisson (see Wertz et
            al. 2017) or Gaussian distribution (see Wertz et al. 2017
            and Stolker et al. 2020).
        residuals : str
            Method used for combining the residuals ('mean', 'median',
            'weighted', or 'clipped').
        reference_in_tag : str, None
            Tag of the database entry with the reference images that
            are read as input. The data of the ``image_in_tag`` itself
            is used as reference data for the PSF subtraction if set to
            ``None``. Note that the mean is not subtracted from the
            data of ``image_in_tag`` and ``reference_in_tag`` in case
            the ``reference_in_tag`` is used, to allow for flux and
            position measurements in the context of RDI.
        resume : bool
            Resume from the last state of the chain that was stored by
            the backend of ``emcee``. Set to ``True`` for continuing
            with the samples from a previous run, for example when it
            was interrupted or to create more steps for the walkers.
            The backend data of ``emcee`` is stored with the tag
            ``[chain_out_tag]_backend`` in the HDF5 database.

        Keyword arguments
        -----------------
        sigma : tuple(float, float, float)
            The standard deviations that randomly initializes the
            start positions of the walkers in a small ball around
            the a priori preferred position. The tuple should contain
            a value for the separation (arcsec), position angle (deg),
            and contrast (mag). The default is set to
            ``(1e-5, 1e-3, 1e-3)``.

        Returns
        -------
        NoneType
            None
        """

        if "sigma" in kwargs:
            self.m_sigma = kwargs["sigma"]
        else:
            self.m_sigma = (1e-5, 1e-3, 1e-3)

        super().__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)

        if reference_in_tag is None:
            self.m_reference_in_port = None
        else:
            self.m_reference_in_port = self.add_input_port(reference_in_tag)

        self.m_chain_out_port = self.add_output_port(chain_out_tag)

        self.m_param = param
        self.m_bounds = bounds
        self.m_nwalkers = nwalkers
        self.m_nsteps = nsteps
        self.m_psf_scaling = psf_scaling
        self.m_pca_number = pca_number
        self.m_aperture = aperture
        self.m_extra_rot = extra_rot
        self.m_merit = merit
        self.m_residuals = residuals
        self.m_resume = resume

        if mask is None:
            self.m_mask = (None, None)
        else:
            self.m_mask = mask

        if self.m_reference_in_port is not None and self.m_merit != "poisson":
            raise NotImplementedError(
                "The reference_in_tag can only be used in combination with "
                "the 'poisson' figure of merit."
            )

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. The posterior distributions of the
        separation, position angle, and flux contrast are sampled with
        the affine invariant Markov chain Monte Carlo (MCMC) ensemble
        sampler ``emcee``. At each step, a negative copy of the PSF
        template is injected and the likelihood function is evaluated
        at the approximate position of the planet.

        Returns
        -------
        NoneType
            None
        """

        print("Input parameters:")
        print(f"   - Number of principal components: {self.m_pca_number}")
        print(f"   - Figure of merit: {self.m_merit}")

        ndim = 3

        cpu = self._m_config_port.get_attribute("CPU")
        work_place = self._m_config_port.get_attribute("WORKING_PLACE")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")
        parang = self.m_image_in_port.get_attribute("PARANG")

        images = self.m_image_in_port.get_all()
        psf = self.m_psf_in_port.get_all()

        im_shape = self.m_image_in_port.get_shape()[-2:]

        self.m_image_in_port.close_port()
        self.m_psf_in_port.close_port()

        if psf.shape[0] != 1 and psf.shape[0] != images.shape[0]:
            raise ValueError(
                "The number of frames in psf_in_tag does not match with the number of "
                "frames in image_in_tag. The DerotateAndStackModule can be used to "
                "average the PSF frames (without derotating) before applying the "
                "MCMCsamplingModule."
            )

        if self.m_mask[0] is not None:
            self.m_mask = (self.m_mask[0] / pixscale, self.m_mask[1])

        if self.m_mask[1] is not None:
            self.m_mask = (self.m_mask[0], self.m_mask[1] / pixscale)

        # create the mask and get the unmasked image indices
        mask = create_mask(im_shape[-2:], self.m_mask)
        indices = np.where(mask.reshape(-1) != 0.0)[0]

        if isinstance(self.m_aperture, float):
            yx_pos = polar_to_cartesian(
                images, self.m_param[0] / pixscale, self.m_param[1]
            )
            aperture = (round(yx_pos[0]), round(yx_pos[1]), self.m_aperture / pixscale)

        elif isinstance(self.m_aperture, tuple):
            aperture = (
                self.m_aperture[1],
                self.m_aperture[0],
                self.m_aperture[2] / pixscale,
            )

        print(f"   - Aperture position (x, y): ({aperture[1]}, {aperture[0]})")
        print(f"   - Aperture radius (pixels): {int(aperture[2])}")

        if self.m_merit == "poisson":
            var_noise = None

        elif self.m_merit in ["gaussian", "hessian"]:
            var_noise = pixel_variance(
                var_type=self.m_merit,
                images=images,
                parang=parang,
                cent_size=self.m_mask[0],
                edge_size=self.m_mask[1],
                pca_number=self.m_pca_number,
                residuals=self.m_residuals,
                aperture=aperture,
                sigma=0.0,
            )

            if self.m_merit == "gaussian":
                print(f"Gaussian standard deviation (counts): {np.sqrt(var_noise):.2e}")
            elif self.m_merit == "hessian":
                print(f"Hessian standard deviation: {np.sqrt(var_noise):.2e}")

        initial = np.zeros((self.m_nwalkers, ndim))

        initial[:, 0] = self.m_param[0] + np.random.normal(
            0, self.m_sigma[0], self.m_nwalkers
        )
        initial[:, 1] = self.m_param[1] + np.random.normal(
            0, self.m_sigma[1], self.m_nwalkers
        )
        initial[:, 2] = self.m_param[2] + np.random.normal(
            0, self.m_sigma[2], self.m_nwalkers
        )

        print("Sampling the posterior distributions with MCMC...")

        backend = emcee.backends.HDFBackend(
            os.path.join(work_place, "PynPoint_database.hdf5"),
            name=self.m_chain_out_port.tag + "_backend",
            read_only=False,
        )

        if not self.m_resume:
            print("Reset backend of emcee...", end="", flush=True)
            backend.reset(self.m_nwalkers, ndim)
            print(" [DONE]")


        if self.m_reference_in_port is None:
            with Pool(processes=cpu) as pool:
                sampler = emcee.EnsembleSampler(
                    self.m_nwalkers,
                    ndim,
                    lnprob,
                    pool=pool,
                    args=(
                        [
                            self.m_bounds,
                            images,
                            psf,
                            mask,
                            parang,
                            self.m_psf_scaling,
                            pixscale,
                            self.m_pca_number,
                            self.m_extra_rot,
                            aperture,
                            indices,
                            self.m_merit,
                            self.m_residuals,
                            var_noise,
                        ]
                    ),
                    backend=backend,
                )

                sampler.run_mcmc(initial, self.m_nsteps, progress=True)

        else:
            with Pool(processes=cpu) as pool:
                sampler = emcee.EnsembleSampler(
                    self.m_nwalkers,
                    ndim,
                    lnprob_rdi,
                    pool=pool,
                    args=(
                        [
                            self.m_bounds,
                            images,
                            psf,
                            mask,
                            parang,
                            self.m_psf_scaling,
                            pixscale,
                            self.m_pca_number,
                            self.m_extra_rot,
                            aperture,
                            indices,
                            self.m_merit,
                            self.m_residuals,
                            var_noise,
                        ]
                    ),
                    backend=backend,
                )

                sampler.run_mcmc(initial, self.m_nsteps, progress=True)

        samples = sampler.get_chain()

        self.m_image_in_port._check_status_and_activate()
        self.m_chain_out_port._check_status_and_activate()

        self.m_chain_out_port.set_all(samples)
        print(f"Number of samples stored: {samples.shape[0]*samples.shape[1]}")

        burnin = int(0.2 * samples.shape[0])
        samples = samples[burnin:, :, :].reshape((-1, ndim))

        sep_percen = np.percentile(samples[:, 0], [16.0, 50.0, 84.0])
        ang_percen = np.percentile(samples[:, 1], [16.0, 50.0, 84.0])
        mag_percen = np.percentile(samples[:, 2], [16.0, 50.0, 84.0])

        print("Median and uncertainties (20% removed as burnin):")

        print(
            f"Separation (mas) = {1e3*sep_percen[1]:.2f} "
            f"(-{1e3*sep_percen[1]-1e3*sep_percen[0]:.2f} "
            f"+{1e3*sep_percen[2]-1e3*sep_percen[1]:.2f})"
        )

        print(
            f"Position angle (deg) = {ang_percen[1]:.2f} "
            f"(-{ang_percen[1]-ang_percen[0]:.2f} "
            f"+{ang_percen[2]-ang_percen[1]:.2f})"
        )

        print(
            f"Contrast (mag) = {mag_percen[1]:.2f} "
            f"(-{mag_percen[1]-mag_percen[0]:.2f} "
            f"+{mag_percen[2]-mag_percen[1]:.2f})"
        )

        history = f"walkers = {self.m_nwalkers}, steps = {self.m_nsteps}"
        self.m_chain_out_port.copy_attributes(self.m_image_in_port)
        self.m_chain_out_port.add_history("MCMCsamplingModule", history)

        mean_accept = np.mean(sampler.acceptance_fraction)
        print(f"Mean acceptance fraction: {mean_accept:.3f}")
        self.m_chain_out_port.add_attribute("ACCEPTANCE", mean_accept, static=True)

        try:
            autocorr = emcee.autocorr.integrated_time(sampler.get_chain())
            print(f"Integrated autocorrelation time = {autocorr}")

        except emcee.autocorr.AutocorrError:
            autocorr = [np.nan, np.nan, np.nan]
            print(
                "The chain is too short to reliably estimate the autocorrelation time. [WARNING]"
            )

        self.m_chain_out_port.add_attribute("AUTOCORR_0", autocorr[0], static=True)
        self.m_chain_out_port.add_attribute("AUTOCORR_1", autocorr[1], static=True)
        self.m_chain_out_port.add_attribute("AUTOCORR_2", autocorr[2], static=True)

        self.m_chain_out_port.close_port()
