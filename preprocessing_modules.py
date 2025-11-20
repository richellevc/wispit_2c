"""
Adjusted Pypeline Modules used for preprocessing.
"""

import sys
import time
import warnings

from typeguard import typechecked
from typing import Optional, Tuple, Union
import numpy as np
from photutils.aperture import RectangularAperture
from photutils import CircularAnnulus

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.image import rotate_images, scale_image, create_mask, pixel_distance
from pynpoint.util.module import memory_frames, progress
from pynpoint.util.module import progress, memory_frames, stack_angles, angle_average
import matplotlib.pyplot as plt

from astropy.coordinates import EarthLocation
from astropy.time import Time


class BadPixelMapModuleIRDIS(ProcessingModule):
    """
    Pipeline module to create a bad pixel map from the dark frames and flat fields.
    """

    @typechecked
    def __init__(self,
                 name_in: str,
                 dark_in_tag: Optional[str],
                 flat_in_tag: Optional[str],
                 bp_map_out_tag: str,
                 dark_threshold: float = 0.2,
                 flat_threshold: float = 0.2) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        dark_in_tag : str, None
            Tag of the database entry with the dark frames that are read as input. Not read if set
            to None.
        flat_in_tag : str, None
            Tag of the database entry with the flat fields that are read as input. Not read if set
            to None.
        bp_map_out_tag : str
            Tag of the database entry with the bad pixel map that is written as output.
        dark_threshold : float
            Fractional threshold with respect to the maximum pixel value in the dark frame to flag
            bad pixels. Pixels `brighter` than the fractional threshold are flagged as bad.
        flat_threshold : float
            Fractional threshold with respect to the -minimum- pixel value in the flat field to flag
            bad pixels. Pixels `fainter` than the fractional threshold are flagged as bad.

        Additional edit compared to original (Alex): mask pixels that are in the unimportant part of IRDIS

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        if dark_in_tag is None:
            self.m_dark_port = None
        else:
            self.m_dark_port = self.add_input_port(dark_in_tag)

        if flat_in_tag is None:
            self.m_flat_port = None
        else:
            self.m_flat_port = self.add_input_port(flat_in_tag)

        self.m_bp_map_out_port = self.add_output_port(bp_map_out_tag)

        self.m_dark_threshold = dark_threshold
        self.m_flat_threshold = flat_threshold

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Collapses a cube of dark frames and flat fields if needed, flags
        bad pixels by comparing the pixel values with the threshold times the maximum value, and
        writes a bad pixel map to the database. For the dark frame, pixel values larger than the
        threshold will be flagged while for the flat frame pixel values smaller than the threshold
        will be flagged.

        Returns
        -------
        NoneType
            None
        """

        if self.m_dark_port is not None:
            dark = self.m_dark_port.get_all()

            if dark.ndim == 3:
                dark = np.mean(dark, axis=0)

            max_dark = np.max(dark)

            print(f'Threshold dark frame = {max_dark*self.m_dark_threshold}')

            bpmap = np.ones(dark.shape)
            bpmap[np.where(dark > max_dark*self.m_dark_threshold)] = 0

        if self.m_flat_port is not None:
            flat = self.m_flat_port.get_all()

            if flat.ndim == 3:
                flat = np.mean(flat, axis=0)

            if self.m_dark_port is None:
                bpmap = np.ones(flat.shape)

            min_flat = np.min(flat)

            if min_flat < 0:
                print(f'Threshold flat field (ADU) = {min_flat*self.m_flat_threshold:.2e}')
                bpmap[np.where(flat < min_flat*self.m_flat_threshold)] = 0
            else:
                print(f'Threshold flat field (ADU) = {min_flat*(1.0/self.m_flat_threshold):.2e}')
                bpmap[np.where(flat < min_flat * (1.0 / self.m_flat_threshold))] = 0

        # Mask unimportant part of IRDIS
        rect_apertures = RectangularAperture(positions=[(487,520),(1513,509)],
                                             w=883,
                                             h=996,
                                             theta=0.)

        rect_apertures_mask = (rect_apertures.to_mask(method="center")[0]).to_image((1024, 2048)) + \
                              (rect_apertures.to_mask(method="center")[1]).to_image((1024, 2048))

        bpmap[rect_apertures_mask == 0.] = 1

        if self.m_dark_port is not None and self.m_flat_port is not None:
            if not dark.shape == flat.shape:
                raise ValueError('Dark and flat images should have the same shape.')

        self.m_bp_map_out_port.set_all(bpmap, data_dim=3)

        if self.m_dark_port is not None:
            self.m_bp_map_out_port.copy_attributes(self.m_dark_port)
        elif self.m_flat_port is not None:
            self.m_bp_map_out_port.copy_attributes(self.m_flat_port)

        history = f'dark = {self.m_dark_threshold}, flat = {self.m_flat_threshold}'
        self.m_bp_map_out_port.add_history('BadPixelMapModule', history)

        self.m_bp_map_out_port.close_port()


class CorrectDistortionModuleSPHERE(ProcessingModule):
    """
    Pynpoint pipeline module for correcting the distortion in y direction in SPHERE data
    """

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: Optional[str],
                 image_out_tag: Optional[str]) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the images that are read as input.
        image_out_tag : str
            Tag of the database entry that is written as output.

        Returns
        -------
        NoneType
            None
        """
        super().__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Crops the images in y direction to 967 lines.
        Scales y axis by 1.0062 according to SPHERE manual

        :return: None
        """

        def _remove_lines(image_in,
                          lines):
            shape_in = image_in.shape

            return image_in[int(lines[2]):shape_in[0]-int(lines[3]),
                   int(lines[0]):shape_in[1]-int(lines[1])]

        def _add_lines(image_in,
                       lines):
            image_out = np.zeros((1024, 2048))

            image_out[int(lines[2]):int(image_out.shape[0] - lines[3]),
            int(lines[0]):int(image_out.shape[1] - lines[1])] = image_in

            return image_out

        def _correct_distortion(image_in, im_index):

            im_removed = _remove_lines(image_in,
                                       [0, 0, (1024 - 968) / 2, (1024 - 968) / 2])

            im_scaled = scale_image(im_removed,scaling_x=1.,scaling_y=1.0062)

            im_corrected = _add_lines(im_scaled,
                                      [0, 0, 25, 25])

            return im_corrected

        self.apply_function_to_images(_correct_distortion,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running CorrectDistortionModuleSPHERE...")

        sys.stdout.flush()

        self.m_image_out_port.add_history("Corrected image distortion", "Multiplied y axis by 1.0062")
        self.m_image_out_port.copy_attributes(self.m_image_in_port)

        self.m_image_out_port.close_port()


class AnnulusBackgroundSubtractionModule(ProcessingModule):
    """Subtract background from annulus.
    """

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: Optional[str],
                 image_out_tag: Optional[str],
                 annulus_size: float = 5.0) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the images that are read as input.
        image_out_tag : str
            Tag of the database entry that is written as output.
        annulus_size : float
            Size of the annulus in pixels.

        Returns
        -------
        NoneType
            None
        """
        super().__init__(name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        # Parameter
        self.m_annulus_size = annulus_size

    @typechecked
    def run(self) -> None:

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        def subtract_background(image_in, im_index,
                                annulus_size):

            edge_offset = 5
            im_shape=image_in.shape

            # Define circular sky annulus for background subtraction
            annulus = CircularAnnulus(positions=(im_shape[0] / 2. - .5, im_shape[1] / 2. - .5),
                                      r_in=np.min(im_shape) / 2. - .5 - annulus_size - edge_offset,
                                      r_out=np.min(im_shape) / 2. - .5 - edge_offset)

            # Determine median flux per pixel inside annulus
            annulus_mask = annulus.to_mask(method="center")
            weighted_data = annulus_mask.multiply(image_in)
            bg_flux = np.median(weighted_data[weighted_data != 0])

            return image_in - bg_flux

        # align all science data
        self.apply_function_to_images(func=subtract_background,
                                      image_in_port=self.m_image_in_port,
                                      image_out_port=self.m_image_out_port,
                                      message="Running AnnulusBackgroundSubtractionModule...",
                                      func_args=(self.m_annulus_size,))

        self.m_image_out_port.copy_attributes(self.m_image_in_port)

        history = "annulus method"
        self.m_image_out_port.add_history("Background subtraction",history)

        self.m_image_out_port.close_port()

        sys.stdout.flush()


class TransmissionModule(ProcessingModule):
    """
    Pipeline module for applying a (coronagraph) transmission correction to the input images. For a given
    input transmission curve, the module scales the flux of the input images by 1.0 / transmission.
    """

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 transmission: Union[np.ndarray, list],
                 cent_size: Optional[float] = None,
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
        transmission : np.ndarray
            Transmission curve to be applied to the input images. Should be a 2D array or list of
            [sep [arcsec], transmission], expecting a (2, n) array.
        cent_size : float, None, optional
            Radius of the central mask (in arcsec). No mask is used when set to None.
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
        self.m_transmission = transmission
        self.m_cent_size = cent_size
        self.m_edge_size = self.m_transmission[0, :].max()

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

        # Convert m_cent_size, m_edge_size and m_transmission[0,:] from arcseconds to pixels
        if self.m_cent_size is not None:
            self.m_cent_size /= pixscale

        self.m_edge_size /= pixscale
        self.m_transmission[0, :] /= pixscale

        # Create 2D disk mask which will be applied to every frame
        mask = create_mask((int(im_shape[-2]), int(im_shape[-1])),
                           (self.m_cent_size, self.m_edge_size)).astype(bool)

        # Calculate the transmission correction factor for all pixels within the mask
        distance_grid, xx_grid, yy_grid = pixel_distance((int(im_shape[-2]), int(im_shape[-1])))
        distances = distance_grid[mask]
        transmission_interp = np.interp(distances, self.m_transmission[0, :], self.m_transmission[1, :])
        correction_interp = 1.0 / transmission_interp

        start_time = time.time()
        # Run the TransmissionModule for each subset of frames
        for i in range(frames[:-1].size):
            # Print progress to command line
            progress(i, len(frames[:-1]), 'Applying transmission correction to frames...', start_time)

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

            # Apply the mask, i.e., multiply all pixels in the mask with the corresponding transmission correction
            images[:, mask] *= correction_interp

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

        # Save cent_size and edge_size as attributes to the output port
        if self.m_cent_size is not None:
            self.m_image_out_port.add_attribute(name='cent_size',
                                                value=self.m_cent_size * pixscale,
                                                static=True)
        if self.m_edge_size is not None:
            self.m_image_out_port.add_attribute(name='edge_size',
                                                value=self.m_edge_size * pixscale,
                                                static=True)


class AngleCalculationModule(ProcessingModule):
    """
    Module for calculating the parallactic angles. The start time of the observation is taken and
    multiples of the exposure time are added to derive the parallactic angle of each frame inside
    the cube. Instrument specific overheads are included.
    The true North offset for SPHERE is automatically added to the derotation angles. The true North
    correction angle is -1.76 degrees: -1.76 +/- 0.04 deg from Maire et al. 2021.

    """

    __author__ = 'Alexander Bohn, Tomas Stolker, Richelle van Capelleveen'

    @typechecked
    def __init__(self,
                 name_in: str,
                 data_tag: str,
                 instrument: str = 'NACO') -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        data_tag : str
            Tag of the database entry for which the parallactic angles are written as attributes.
        instrument : str
            Instrument name ('NACO', 'SPHERE/IRDIS', or 'SPHERE/IFS').

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        # Parameters
        self.m_instrument = instrument

        # Set parameters according to choice of instrument
        if self.m_instrument == 'NACO':

            # pupil offset in degrees
            self.m_pupil_offset = 0.            # No offset here

            # no overheads in cube mode, since cube is read out after all individual exposures
            # see NACO manual page 62 (v102)
            self.m_O_START = 0.
            self.m_DIT_DELAY = 0.
            self.m_ROT = 0.

            # rotator offset in degrees
            self.m_rot_offset = 89.44           # According to NACO manual page 65 (v102)

        elif self.m_instrument == 'SPHERE/IRDIS':

            # pupil offset in degrees
            self.m_pupil_offset = 135.99        # According to SPHERE manual page 71 (v114.1)

            self.m_true_north_offset = -1.76    # True North offset for SPHERE - Maire et al 2021

            # overheads in cube mode (several NDITS) in hours
            self.m_O_START = 0.3 / 3600.        # According to SPHERE manual page 90/91 (v102)
            self.m_DIT_DELAY = 0.1 / 3600.      # According to SPHERE manual page 90/91 (v102)
            self.m_ROT = 0.838 / 3600.          # According to SPHERE manual page 90/91 (v102)

            # rotator offset in degrees
            self.m_rot_offset = 0.              # no offset here

        elif self.m_instrument == 'SPHERE/IFS':

            # pupil offset in degrees
            self.m_pupil_offset = 135.99 - 100.48   # According to SPHERE manual page 71 (v114.1)

            self.m_true_north_offset = -1.76        # True North offset for SPHERE - Maire et al 2021

            # overheads in cube mode (several NDITS) in hours
            self.m_O_START = 0.3 / 3600.            # According to SPHERE manual page 90/91 (v102)
            self.m_DIT_DELAY = 0.2 / 3600.          # According to SPHERE manual page 90/91 (v102)
            self.m_ROT = 1.65 / 3600.               # According to SPHERE manual page 90/91 (v102)

            # rotator offset in degrees
            self.m_rot_offset = 0.                  # no offset here

        else:
            raise ValueError('The instrument argument should be set to either \'NACO\', '
                             '\'SPHERE/IRDIS\', or \'SPHERE/IFS\'.')

        self.m_data_in_port = self.add_input_port(data_tag)
        self.m_data_out_port = self.add_output_port(data_tag)

    @typechecked
    def _attribute_check(self,
                         ndit: np.ndarray,
                         steps: np.ndarray) -> None:

        if not np.all(ndit == steps):
            warnings.warn('There is a mismatch between the NDIT and NFRAMES values. A frame '
                          'selection should be applied after the parallactic angles are '
                          'calculated.')

        if self.m_instrument == 'SPHERE/IFS':
            warnings.warn('AngleCalculationModule has not been tested for SPHERE/IFS data.')

        if self.m_instrument in ('SPHERE/IRDIS', 'SPHERE/IFS'):

            if self._m_config_port.get_attribute('RA') != 'ESO INS4 DROT2 RA':

                warnings.warn('For SPHERE data it is recommended to use the header keyword '
                              '\'ESO INS4 DROT2 RA\' to specify the object\'s right ascension. '
                              'The input will be parsed accordingly. Using the regular '
                              '\'RA\' keyword will lead to wrong parallactic angles.')

            if self._m_config_port.get_attribute('DEC') != 'ESO INS4 DROT2 DEC':

                warnings.warn('For SPHERE data it is recommended to use the header keyword '
                              '\'ESO INS4 DROT2 DEC\' to specify the object\'s declination. '
                              'The input will be parsed accordingly. Using the regular '
                              '\'DEC\' keyword will lead to wrong parallactic angles.')

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Calculates the parallactic angles from the position of the object
        on the sky and the telescope location on earth. The start of the observation is used to
        extrapolate for the observation time of each individual image of a data cube. The values
        are written as PARANG attributes to *data_tag*.

        Returns
        -------
        NoneType
            None
        """

        # Load cube sizes
        steps = self.m_data_in_port.get_attribute('NFRAMES')
        ndit = self.m_data_in_port.get_attribute('NDIT')

        self._attribute_check(ndit, steps)

        # Load exposure time [hours]
        exptime = self.m_data_in_port.get_attribute('DIT')/3600.

        # Load telescope location
        tel_lat = self.m_data_in_port.get_attribute('LATITUDE')
        tel_lon = self.m_data_in_port.get_attribute('LONGITUDE')

        # Load temporary target position
        tmp_ra = self.m_data_in_port.get_attribute('RA')
        tmp_dec = self.m_data_in_port.get_attribute('DEC')

        # Parse to degree depending on instrument
        if 'SPHERE' in self.m_instrument:

            # get sign of declination
            tmp_dec_sign = np.sign(tmp_dec)
            tmp_dec = np.abs(tmp_dec)

            # parse RA
            tmp_ra_s = tmp_ra % 100
            tmp_ra_m = ((tmp_ra - tmp_ra_s) / 1e2) % 100
            tmp_ra_h = ((tmp_ra - tmp_ra_s - tmp_ra_m * 1e2) / 1e4)

            # parse DEC
            tmp_dec_s = tmp_dec % 100
            tmp_dec_m = ((tmp_dec - tmp_dec_s) / 1e2) % 100
            tmp_dec_d = ((tmp_dec - tmp_dec_s - tmp_dec_m * 1e2) / 1e4)

            # get RA and DEC in degree
            ra = (tmp_ra_h + tmp_ra_m / 60. + tmp_ra_s / 3600.) * 15.
            dec = tmp_dec_sign * (tmp_dec_d + tmp_dec_m / 60. + tmp_dec_s / 3600.)

        else:
            ra = tmp_ra
            dec = tmp_dec

        # Load start times of exposures
        obs_dates = self.m_data_in_port.get_attribute('DATE')

        # Load pupil positions during observations
        if self.m_instrument == 'NACO':
            pupil_pos = self.m_data_in_port.get_attribute('PUPIL')

        elif self.m_instrument == 'SPHERE/IRDIS':
            pupil_pos = np.zeros(steps.shape)

        elif self.m_instrument == 'SPHERE/IFS':
            pupil_pos = np.zeros(steps.shape)

        new_angles = np.array([])
        pupil_pos_arr = np.array([])

        start_time = time.time()

        # Calculate parallactic angles for each cube
        for i, tmp_steps in enumerate(steps):
            progress(i, len(steps), 'Calculating parallactic angles...', start_time)

            t = Time(obs_dates[i].decode('utf-8'),
                     location=EarthLocation(lat=tel_lat, lon=tel_lon))

            sid_time = t.sidereal_time('apparent').value

            # Extrapolate sideral times from start time of the cube for each frame of it
            sid_time_arr = np.linspace(sid_time+self.m_O_START,
                                       (sid_time+self.m_O_START) +
                                       (exptime+self.m_DIT_DELAY + self.m_ROT)*(tmp_steps-1),
                                       tmp_steps)

            # Convert to degrees
            sid_time_arr_deg = sid_time_arr * 15.

            # Calculate hour angle in degrees
            hour_angle = sid_time_arr_deg - ra[i]

            # Conversion to radians:
            hour_angle_rad = np.deg2rad(hour_angle)
            dec_rad = np.deg2rad(dec[i])
            lat_rad = np.deg2rad(tel_lat)

            p_angle = np.arctan2(np.sin(hour_angle_rad),
                                 (np.cos(dec_rad)*np.tan(lat_rad) -
                                  np.sin(dec_rad)*np.cos(hour_angle_rad)))

            new_angles = np.append(new_angles, np.rad2deg(p_angle))
            pupil_pos_arr = np.append(pupil_pos_arr, np.ones(tmp_steps)*pupil_pos[i])

        # Correct for rotator (SPHERE) or pupil offset (NACO)
        if self.m_instrument == 'NACO':
            # See NACO manual page 65 (v102)
            new_angles_corr = new_angles - (90. + (self.m_rot_offset-pupil_pos_arr))

        elif self.m_instrument == 'SPHERE/IRDIS':
            # See SPHERE manual page 71 (v114.1)
            new_angles_corr = new_angles + self.m_pupil_offset + self.m_true_north_offset

        elif self.m_instrument == 'SPHERE/IFS':
            # See SPHERE manual page 71 (v114.1)
            new_angles_corr = new_angles + self.m_pupil_offset + self.m_true_north_offset

        indices = np.where(new_angles_corr < -180.)[0]
        if indices.size > 0:
            new_angles_corr[indices] += 360.

        indices = np.where(new_angles_corr > 180.)[0]
        if indices.size > 0:
            new_angles_corr[indices] -= 360.

        self.m_data_out_port.add_attribute('PARANG', new_angles_corr, static=False)


class StackAndSubsetModule(ProcessingModule):
    """
    Pipeline module for stacking subsets of images and/or selecting a random sample of images.
    """

    @typechecked
    def __init__(
        self,
        name_in: str,
        image_in_tag: str,
        image_out_tag: str,
        random: Optional[int] = None,
        stacking: Optional[int] = None,
        combine: str = "mean",
        max_rotation: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be different from
            *image_in_tag*.
        random : int, None
            Number of random images. All images are used if set to None.
        stacking : int, None
            Number of stacked images per subset. No stacking is applied if set to None.
        combine : str
            Method for combining images ('mean', 'median' or 'sum'). The angles are always mean-combined.
        max_rotation : float, None
            Maximum allowed field rotation (deg) throughout each subset of stacked images when
            `stacking` is not None. No restriction on the field rotation is applied if set to
            None.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_random = random
        self.m_stacking = stacking
        self.m_combine = combine
        self.m_max_rotation = max_rotation

        if self.m_stacking is None and self.m_random is None:
            warnings.warn("Both 'stacking' and 'random' are set to None.")

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Stacks subsets of images and/or selects a random subset. Also
        the parallactic angles are mean-combined if images are stacked.

        Returns
        -------
        NoneType
            None
        """

        @typechecked
        def _stack_subsets(
            nimages: int, im_shape: Tuple[int, ...], parang: Optional[np.ndarray]
        ) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray]:

            im_new = None
            parang_new = None

            if self.m_stacking is not None:
                if self.m_max_rotation is not None:
                    frames = stack_angles(self.m_stacking, parang, self.m_max_rotation)
                else:
                    frames = memory_frames(self.m_stacking, nimages)

                nimages_new = np.size(frames) - 1

                if parang is None:
                    parang_new = None
                else:
                    parang_new = np.zeros(nimages_new)

                im_new = np.zeros((nimages_new, im_shape[1], im_shape[2]))

                start_time = time.time()

                for i in range(nimages_new):
                    progress(
                        i, nimages_new, "Stacking subsets of images...", start_time
                    )

                    if parang is not None:
                        # parang_new[i] = np.mean(parang[frames[i]:frames[i+1]])
                        parang_new[i] = angle_average(parang[frames[i] : frames[i + 1]])

                    im_subset = self.m_image_in_port[frames[i] : frames[i + 1],]

                    if self.m_combine == "mean":
                        im_new[i,] = np.mean(im_subset, axis=0)
                    elif self.m_combine == "median":
                        im_new[i,] = np.median(im_subset, axis=0)
                    elif self.m_combine == "sum":
                        im_new[i,] = np.sum(im_subset, axis=0)

                im_shape = im_new.shape

            else:
                if parang is not None:
                    parang_new = np.copy(parang)

            return im_shape, im_new, parang_new

        @typechecked
        def _random_subset(
            im_shape: Tuple[int, ...], im_new: np.ndarray, parang_new: np.ndarray
        ) -> Tuple[int, np.ndarray, np.ndarray]:

            if self.m_random is not None:
                choice = np.random.choice(im_shape[0], self.m_random, replace=False)
                choice = list(np.sort(choice))

                if parang_new is None:
                    parang_new = None
                else:
                    parang_new = parang_new[choice]

                if self.m_stacking is None:
                    im_new = self.m_image_in_port[list(choice),]
                else:
                    im_new = im_new[choice,]

            if self.m_random is None and self.m_stacking is None:
                nimages = 0
            elif im_new.ndim == 2:
                nimages = 1
            elif im_new.ndim == 3:
                nimages = im_new.shape[0]

            return nimages, im_new, parang_new

        non_static = self.m_image_in_port.get_all_non_static_attributes()

        im_shape = self.m_image_in_port.get_shape()
        nimages = im_shape[0]

        if self.m_random is not None:
            if self.m_stacking is None and im_shape[0] < self.m_random:
                raise ValueError(
                    "The number of images of the destination subset is larger than "
                    "the number of images in the source."
                )

            if (
                self.m_stacking is not None
                and int(float(im_shape[0]) / float(self.m_stacking)) < self.m_random
            ):
                raise ValueError(
                    "The number of images of the destination subset is larger than "
                    "the number of images in the stacked source."
                )

        if "PARANG" in non_static:
            parang = self.m_image_in_port.get_attribute("PARANG")
        else:
            parang = None

        im_shape, im_new, parang_new = _stack_subsets(nimages, im_shape, parang)
        nimages, im_new, parang_new = _random_subset(im_shape, im_new, parang_new)

        if self.m_random or self.m_stacking:
            self.m_image_out_port.set_all(im_new, keep_attributes=True)
            self.m_image_out_port.copy_attributes(self.m_image_in_port)
            self.m_image_out_port.add_attribute(
                "INDEX", np.arange(0, nimages, 1), static=False
            )

            if parang_new is not None:
                self.m_image_out_port.add_attribute("PARANG", parang_new, static=False)

            if "NFRAMES" in non_static:
                self.m_image_out_port.del_attribute("NFRAMES")

            history = f"stacking = {self.m_stacking}, random = {self.m_random}"
            self.m_image_out_port.add_history("StackAndSubsetModule", history)

        self.m_image_out_port.close_port()