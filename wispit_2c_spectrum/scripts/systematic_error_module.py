"""
Pipeline modules for photometric and astrometric measurements.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np

from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.image import (
    polar_to_cartesian,
    cartesian_to_polar,
    center_subpixel,
)

from pynpoint import FakePlanetModule, SimplexMinimizationModule


class SystematicErrorModule_rdi(ProcessingModule):
    """
    Pipeline module for estimating the systematic error of the flux and position measurement.
    """

    __author__ = "Tomas Stolker"

    @typechecked
    def __init__(
        self,
        name_in: str,
        image_in_tag: str,
        psf_in_tag: str,
        offset_out_tag: str,
        position: Tuple[float, float],
        magnitude: float,
        angles: Tuple[float, float, int] = (0.0, 359.0, 360),
        psf_scaling: float = 1.0,
        merit: str = "gaussian",
        aperture: float = 0.1,
        tolerance: float = 0.01,
        pca_number: int = 10,
        mask: Optional[
            Union[
                Tuple[float, float],
                Tuple[None, float],
                Tuple[float, None],
                Tuple[None, None],
            ]
        ] = None,
        extra_rot: float = 0.0,
        residuals: str = "median",
        reference_in_tag: Optional[str] = None,
        offset: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the science images for which the systematic error is
            estimated.
        psf_in_tag : str
            Tag of the database entry with the PSF template that is used as fake planet. Can be
            either a single image or a stack of images equal in size to ``image_in_tag``.
        offset_out_tag : str
            Tag of the database entry at which the differences are stored between the injected and
            and retrieved values of the separation (arcsec), position angle (deg), contrast
            (mag), x position (pix), and y position (pix).
        position : tuple(float, float)
            Separation (arcsec) and position angle (deg) that are used to remove the planet signal.
            The separation is also used to estimate the systematic error.
        magnitude : float
            Magnitude that is used to remove the planet signal and estimate the systematic error.
        angles : tuple(float, float, int)
            The start, end, and number of the position angles (linearly sampled) that are used to
            estimate the systematic errors (default: 0., 359., 360). The endpoint is also included.
        psf_scaling : float
            Additional scaling factor of the planet flux (e.g., to correct for a neutral density
            filter). Should be a positive value.
        merit : str
            Figure of merit for the minimization ('hessian', 'gaussian', or 'poisson'). Either the
            determinant of the Hessian matrix is minimized ('hessian') or the flux of each pixel
            ('gaussian' or 'poisson'). For the latter case, the estimate noise is assumed to follow
            a Poisson (see Wertz et al. 2017) or Gaussian distribution (see Wertz et al. 2017 and
            Stolker et al. 2020).
        aperture : float
            Aperture radius (arcsec) that is used for measuring the figure of merit.
        tolerance : float
            Absolute error on the input parameters, position (pix) and contrast (mag), that is used
            as acceptance level for convergence. Note that only a single value can be specified
            which is used for both the position and flux so tolerance=0.1 will give a precision of
            0.1 mag and 0.1 pix. The tolerance on the output (i.e., the chi-square value) is set to
            np.inf so the condition is always met.
        pca_number : int
            Number of principal components (PCs) used for the PSF subtraction.
        mask : tuple(float, float), None
            Inner and outer mask radius (arcsec) which is applied before the PSF subtraction. Both
            elements of the tuple can be set to None.
        extra_rot : float
            Additional rotation angle of the images in clockwise direction (deg).
        residuals : str
            Method for combining the residuals ('mean', 'median', 'weighted', or 'clipped').
        reference_in_tag : str, None
            Tag of the database entry with the reference images that
            are read as input. The data of the ``image_in_tag`` itself
            is used as reference data for the PSF subtraction if set to
            ``None``. Note that the mean is not subtracted from the
            data of ``image_in_tag`` and ``reference_in_tag`` in case
            the ``reference_in_tag`` is used, to allow for flux and
            position measurements in the context of RDI.
        offset : float, None
            Offset (pixels) by which the negative PSF may deviate from the positive injected PSF.
            No constraint on the position is applied if set to None. Only the contrast is optimized
            and the position is fixed to the injected value if ``offset=0``.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        self.m_image_in_tag = image_in_tag
        self.m_psf_in_tag = psf_in_tag
        self.m_reference_in_tag = reference_in_tag

        if reference_in_tag is None:
            self.m_reference_in_port = None
        else:
            self.m_reference_in_port = self.add_input_port(reference_in_tag)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_offset_out_port = self.add_output_port(offset_out_tag)

        self.m_position = position
        self.m_magnitude = magnitude
        self.m_angles = angles
        self.m_psf_scaling = psf_scaling
        self.m_merit = merit
        self.m_aperture = aperture
        self.m_tolerance = tolerance
        self.m_mask = mask
        self.m_extra_rot = extra_rot
        self.m_residuals = residuals
        self.m_pca_number = pca_number
        self.m_offset = offset

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Removes the planet signal, then artificial planets are injected
        (one at a time) at equally separated position angles and their position and contrast is
        determined with the :class:`~pynpoint.processing.fluxposition.SimplexMinimizationModule`.
        The differences between the injected and retrieved separation, position angle, and contrast
        are then stored as output.

        Returns
        -------
        NoneType
            None
        """

        print("Input parameters:")
        print(f"   - Number of principal components = {self.m_pca_number}")
        print(f"   - Figure of merit = {self.m_merit}")
        print(f"   - Residuals type = {self.m_residuals}")
        print(f"   - Absolute tolerance (pixels/mag) = {self.m_tolerance}")
        print(f"   - Maximum offset = {self.m_offset}")
        print(f"   - Aperture radius (arcsec) = {self.m_aperture}")

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")
        image = self.m_image_in_port[0,]

        module = FakePlanetModule(
            name_in=f"{self._m_name}_fake",
            image_in_tag=self.m_image_in_tag,
            psf_in_tag=self.m_psf_in_tag,
            image_out_tag=f"{self._m_name}_empty",
            position=(self.m_position[0], self.m_position[1] + self.m_extra_rot),
            magnitude=self.m_magnitude,
            psf_scaling=-self.m_psf_scaling,
        )

        module.connect_database(self._m_data_base)
        module._m_output_ports[f"{self._m_name}_empty"].del_all_data()
        module._m_output_ports[f"{self._m_name}_empty"].del_all_attributes()
        module.run()

        sep = float(self.m_position[0])

        angles = np.linspace(
            self.m_angles[0], self.m_angles[1], self.m_angles[2], endpoint=True
        )

        print("Testing the following parameters:")
        print(f"   - Contrast (mag) = {self.m_magnitude:.2f}")
        print(f"   - Separation (mas) = {sep*1e3:.1f}")
        print(f"   - Position angle range (deg) = {angles[0]} - {angles[-1]}")

        if angles.size > 1:
            print(f"     in steps of {np.mean(np.diff(angles)):.2f} deg")

        # Image center (y, x) with subpixel accuracy
        im_center = center_subpixel(image)

        if self.m_reference_in_port is not None and self.m_merit != "poisson":
            raise NotImplementedError(
                "The reference_in_tag can only be used in combination with "
                "the 'poisson' figure of merit."
            )

        for i, ang in enumerate(angles):
            print(f"\nProcessing position angle: {ang} deg...")

            # Convert the polar coordiantes of the separation and position angle that is tested
            # into cartesian coordinates (y, x)
            planet_pos_yx = polar_to_cartesian(image, sep / pixscale, ang)
            planet_pos_xy = (planet_pos_yx[1], planet_pos_yx[0])

            # Convert the planet position to polar coordinates
            planet_sep_ang = cartesian_to_polar(
                im_center, planet_pos_yx[0], planet_pos_yx[1]
            )

            # Change the separation units to arcsec
            planet_sep_ang = (planet_sep_ang[0] * pixscale, planet_sep_ang[1])

            # Inject the artifical planet

            module = FakePlanetModule(
                position=(planet_sep_ang[0], planet_sep_ang[1] + self.m_extra_rot),
                magnitude=self.m_magnitude,
                psf_scaling=self.m_psf_scaling,
                name_in=f"{self._m_name}_fake_{i}",
                image_in_tag=f"{self._m_name}_empty",
                psf_in_tag=self.m_psf_in_tag,
                image_out_tag=f"{self._m_name}_fake",
            )

            module.connect_database(self._m_data_base)
            module._m_output_ports[f"{self._m_name}_fake"].del_all_data()
            module._m_output_ports[f"{self._m_name}_fake"].del_all_attributes()
            module.run()

            # Retrieve the position and contrast of the artificial planet

            module = SimplexMinimizationModule(
                position=planet_pos_xy,
                magnitude=self.m_magnitude,
                psf_scaling=-self.m_psf_scaling,
                name_in=f"{self._m_name}_fake_{i}",
                image_in_tag=f"{self._m_name}_fake",
                psf_in_tag=self.m_psf_in_tag,
                res_out_tag=f"{self._m_name}_simplex",
                flux_position_tag=f"{self._m_name}_fluxpos",
                merit=self.m_merit,
                aperture=self.m_aperture,
                sigma=0.0,
                tolerance=self.m_tolerance,
                pca_number=self.m_pca_number,
                cent_size=self.m_mask[0],
                edge_size=self.m_mask[1],
                extra_rot=self.m_extra_rot,
                residuals=self.m_residuals,
                reference_in_tag=self.m_reference_in_tag,
                offset=self.m_offset,
            )

            module.connect_database(self._m_data_base)
            module._m_output_ports[f"{self._m_name}_simplex"].del_all_data()
            module._m_output_ports[f"{self._m_name}_simplex"].del_all_attributes()
            module._m_output_ports[f"{self._m_name}_fluxpos"].del_all_data()
            module._m_output_ports[f"{self._m_name}_fluxpos"].del_all_attributes()
            module.run()

            # Add the input port to collect the results of SimplexMinimizationModule
            fluxpos_out_port = self.add_input_port(f"{self._m_name}_fluxpos")

            # Create a list with the offset between the injected and retrieved values of the
            # separation (arcsec), position angle (deg), contrast (mag), x position (pixels),
            # and y position (pixels).
            data = [
                planet_sep_ang[0] - fluxpos_out_port[-1, 2],  # Separation (arcsec)
                planet_sep_ang[1] - fluxpos_out_port[-1, 3],  # Position angle (deg)
                self.m_magnitude - fluxpos_out_port[-1, 4],  # Contrast (mag)
                planet_pos_xy[0] - fluxpos_out_port[-1, 0],  # Position x (pixels)
                planet_pos_xy[1] - fluxpos_out_port[-1, 1],
            ]  # Position y (pixels)

            if data[1] > 180.0:
                data[1] -= 360.0

            elif data[1] < -180.0:
                data[1] += 360.0

            print(
                f"Offset: {data[0]*1e3:.2f} mas, {data[1]:.2f} deg, {data[2]:.2f} mag"
            )

            self.m_offset_out_port.append(data, data_dim=2)

        offset_in_port = self.add_input_port(self.m_offset_out_port.tag)
        offsets = offset_in_port.get_all()

        sep_percen = np.percentile(offsets[:, 0], [16.0, 50.0, 84.0])
        ang_percen = np.percentile(offsets[:, 1], [16.0, 50.0, 84.0])
        mag_percen = np.percentile(offsets[:, 2], [16.0, 50.0, 84.0])
        x_pos_percen = np.percentile(offsets[:, 3], [16.0, 50.0, 84.0])
        y_pos_percen = np.percentile(offsets[:, 4], [16.0, 50.0, 84.0])

        print("\nMedian offset and uncertainties:")

        print(
            f"   - Position x (pixels) = {x_pos_percen[1]:.2f} "
            f"(-{x_pos_percen[1]-x_pos_percen[0]:.2f} "
            f"+{x_pos_percen[2]-x_pos_percen[1]:.2f})"
        )

        print(
            f"   - Position y (pixels) = {y_pos_percen[1]:.2f} "
            f"(-{y_pos_percen[1]-y_pos_percen[0]:.2f} "
            f"+{y_pos_percen[2]-y_pos_percen[1]:.2f})"
        )

        print(
            f"   - Separation (mas) = {1e3*sep_percen[1]:.2f} "
            f"(-{1e3*sep_percen[1]-1e3*sep_percen[0]:.2f} "
            f"+{1e3*sep_percen[2]-1e3*sep_percen[1]:.2f})"
        )

        print(
            f"   - Position angle (deg) = {ang_percen[1]:.2f} "
            f"(-{ang_percen[1]-ang_percen[0]:.2f} "
            f"+{ang_percen[2]-ang_percen[1]:.2f})"
        )

        print(
            f"   - Contrast (mag) = {mag_percen[1]:.2f} "
            f"(-{mag_percen[1]-mag_percen[0]:.2f} "
            f"+{mag_percen[2]-mag_percen[1]:.2f})"
        )

        history = (
            f"sep = {self.m_position[0]:.3f}, "
            f"pa = {self.m_position[1]:.1f}, "
            f"mag = {self.m_magnitude:.1f}"
        )

        self.m_offset_out_port.copy_attributes(self.m_image_in_port)
        self.m_offset_out_port.add_history("SystematicErrorModule", history)
        self.m_offset_out_port.close_port()