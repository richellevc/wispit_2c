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

