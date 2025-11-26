
"""
Adjusted Pypline Modules used for reduction.
"""

import os

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
        print("imshape", im_shape)

        if self.m_reference_in_port is not None:
            ref_data = self.m_reference_in_port.get_all()
            if ref_data.shape[1:] != images.shape[1:]:
                raise ValueError(
                    "The image size of the science data and the reference data "
                    "should be identical."
                )

        self.m_image_in_port.close_port()
        self.m_reference_in_port.close_port()
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
                            ref_data,
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
        self.m_reference_in_port._check_status_and_activate()
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
