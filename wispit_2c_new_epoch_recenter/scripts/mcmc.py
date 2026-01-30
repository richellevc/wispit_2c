"""
Functions for MCMC sampling.
"""

import math

from typing import Optional, Tuple

import numpy as np

from typeguard import typechecked

from pynpoint.util.analysis import fake_planet, merit_function
from pynpoint.util.psf import pca_psf_subtraction
from pynpoint.util.residuals import combine_residuals
from sklearn.decomposition import PCA


@typechecked
def lnprob_rdi(
    param: np.ndarray,
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    images: np.ndarray,
    ref_data: np.ndarray,
    psf: np.ndarray,
    mask: np.ndarray,
    parang: np.ndarray,
    psf_scaling: float,
    pixscale: float,
    pca_number: int,
    extra_rot: float,
    aperture: Tuple[int, int, float],
    indices: np.ndarray,
    merit: str,
    residuals: str,
    var_noise: Optional[float],
) -> float:
    """
    Function for the log posterior function. Should be placed at the highest level of the
    Python module to be pickable for the multiprocessing.

    Parameters
    ----------
    param : numpy.ndarray
        The separation (arcsec), angle (deg), and contrast (mag). The angle is measured in
        counterclockwise direction with respect to the positive y-axis.
    bounds : tuple(tuple(float, float), tuple(float, float), tuple(float, float))
        The boundaries of the separation (arcsec), angle (deg), and contrast (mag). Each set of
        boundaries is specified as a tuple.
    images : numpy.ndarray
        Stack with images.
    ref_data : numpy.ndarray
        Stack with reference image data
    psf : numpy.ndarray
        PSF template, either a single image (2D) or a cube (3D) with the dimensions equal to
        *images*.
    mask : numpy.ndarray
        Array with the circular mask (zeros) of the central and outer regions.
    parang : numpy.ndarray
        Array with the angles for derotation.
    psf_scaling : float
        Additional scaling factor of the planet flux (e.g., to correct for a neutral density
        filter). Should be negative in order to inject negative fake planets.
    pixscale : float
        Additional scaling factor of the planet flux (e.g., to correct for a neutral density
        filter). Should be negative in order to inject negative fake planets.
    pca_number : int
        Number of principal components used for the PSF subtraction.
    extra_rot : float
        Additional rotation angle of the images (deg).
    aperture : tuple(int, int, float)
        Position (y, x) of the aperture center (pix) and aperture radius (pix).
    indices : numpy.ndarray
        Non-masked image indices.
    merit : str
        Figure of merit that is used for the likelihood function ('gaussian' or 'poisson').
        Pixels are assumed to be independent measurements which are expected to be equal to
        zero in case the best-fit negative PSF template is injected. With 'gaussian', the
        variance is estimated from the pixel values within an annulus at the separation of
        the aperture (but excluding the pixels within the aperture). With 'poisson', a
        Poisson distribution is assumed for the variance of each pixel value.
    residuals : str
        Method used for combining the residuals ('mean', 'median', 'weighted', or 'clipped').
    var_noise : float, None
        Variance of the noise which is required when `merit` is set to 'gaussian' or 'hessian'.

    Returns
    -------
    float
        Log posterior probability.
    """

    @typechecked
    def _lnprior() -> float:
        """
        Internal function for the log prior function.

        Returns
        -------
        float
            Log prior.
        """

        if (
            bounds[0][0] <= param[0] <= bounds[0][1]
            and bounds[1][0] <= param[1] <= bounds[1][1]
            and bounds[2][0] <= param[2] <= bounds[2][1]
        ):

            ln_prior = 0.0

        else:

            ln_prior = -np.inf

        return ln_prior

    @typechecked
    def _lnlike() -> float:
        """
        Internal function for the log likelihood function.

        Returns
        -------
        float
            Log likelihood.
        """

        sep, ang, mag = param
        im_shape = images.shape
        ref_shape = ref_data.shape

        # reshape reference data and select the unmasked pixels
        ref_reshape = ref_data.reshape(
            ref_shape[0], ref_shape[1] * ref_shape[2]
        )

        mean_ref = np.mean(ref_reshape, axis=0)
        ref_reshape -= mean_ref

        # create the PCA basis
        sklearn_pca = PCA(n_components=pca_number, svd_solver="arpack")
        sklearn_pca.fit(ref_reshape)

        # add mean of reference array as 1st PC and orthogonalize it to the PCA basis
        mean_ref_reshape = mean_ref.reshape((1, mean_ref.shape[0]))

        q_ortho, _ = np.linalg.qr(
            np.vstack((mean_ref_reshape, sklearn_pca.components_[:-1,])).T
        )

        sklearn_pca.components_ = q_ortho.T

        # Add fake planet
        fake = fake_planet(
            images=images,
            psf=psf,
            parang=parang - extra_rot,
            position=(sep / pixscale, ang),
            magnitude=mag,
            psf_scaling=psf_scaling,
        )

        im_reshape = np.reshape(
            fake * mask, (im_shape[0], im_shape[1] * im_shape[2])
        )

        # NO RDI
        #im_res_rot, im_res_derot = pca_psf_subtraction(
        #    images=fake * mask,
        #    angles=-1.0 * parang + extra_rot,
        #    pca_number=pca_number,
        #    indices=indices,
        #)

        #RDI
        im_res_rot, im_res_derot = pca_psf_subtraction(
            images=im_reshape,
            angles=-1.0 * parang + extra_rot,
            pca_number=pca_number,
            pca_sklearn=sklearn_pca,
            im_shape=im_shape,
            #indices=indices,    # maybe this should be none
            indices=None,  # maybe this should be none
        )

        res_stack = combine_residuals(
            method=residuals, res_rot=im_res_derot, residuals=im_res_rot, angles=parang
        )

        chi_square = merit_function(
            residuals=res_stack[0,],
            merit=merit,
            aperture=aperture,
            sigma=0.0,
            var_noise=var_noise,
        )

        return -0.5 * chi_square

    ln_prior = _lnprior()

    if math.isinf(ln_prior):
        ln_prob = -np.inf
    else:
        ln_prob = ln_prior + _lnlike()

    return ln_prob

@typechecked
def residuals_rdi(
        param: np.ndarray,
        images: np.ndarray,
        ref_data: np.ndarray,
        psf: np.ndarray,
        mask: np.ndarray,
        parang: np.ndarray,
        psf_scaling: float,
        pixscale: float,
        pca_number: int,
        extra_rot: float,
        residuals: str,
) -> np.ndarray:
    """
    Function for storing residuals. Should be placed at the highest level of the
    Python module to be pickable for the multiprocessing.

    Parameters
    ----------
    param : numpy.ndarray
        The separation (arcsec), angle (deg), and contrast (mag). The angle is measured in
        counterclockwise direction with respect to the positive y-axis.
    images : numpy.ndarray
        Stack with images.
    ref_data : numpy.ndarray
        Stack with reference image data
    psf : numpy.ndarray
        PSF template, either a single image (2D) or a cube (3D) with the dimensions equal to
        *images*.
    mask : numpy.ndarray
        Array with the circular mask (zeros) of the central and outer regions.
    parang : numpy.ndarray
        Array with the angles for derotation.
    psf_scaling : float
        Additional scaling factor of the planet flux (e.g., to correct for a neutral density
        filter). Should be negative in order to inject negative fake planets.
    pixscale : float
        Additional scaling factor of the planet flux (e.g., to correct for a neutral density
        filter). Should be negative in order to inject negative fake planets.
    pca_number : int
        Number of principal components used for the PSF subtraction.
    extra_rot : float
        Additional rotation angle of the images (deg).
    residuals : str
        Method used for combining the residuals ('mean', 'median', 'weighted', or 'clipped').

    Returns
    -------
    numpy.ndarray
        Residuals.
    """

    sep, ang, mag = param
    im_shape = images.shape
    ref_shape = ref_data.shape

    # reshape reference data and select the unmasked pixels
    ref_reshape = ref_data.reshape(
        ref_shape[0], ref_shape[1] * ref_shape[2]
    )

    mean_ref = np.mean(ref_reshape, axis=0)
    ref_reshape -= mean_ref

    # create the PCA basis
    sklearn_pca = PCA(n_components=pca_number, svd_solver="arpack")
    sklearn_pca.fit(ref_reshape)

    # add mean of reference array as 1st PC and orthogonalize it to the PCA basis
    mean_ref_reshape = mean_ref.reshape((1, mean_ref.shape[0]))

    q_ortho, _ = np.linalg.qr(
        np.vstack((mean_ref_reshape, sklearn_pca.components_[:-1,])).T
    )

    sklearn_pca.components_ = q_ortho.T

    # Add fake planet
    fake = fake_planet(
        images=images,
        psf=psf,
        parang=parang - extra_rot,
        position=(sep / pixscale, ang),
        magnitude=mag,
        psf_scaling=psf_scaling,
    )

    im_reshape = np.reshape(
        fake * mask, (im_shape[0], im_shape[1] * im_shape[2])
    )

    im_res_rot, im_res_derot = pca_psf_subtraction(
        images=im_reshape,
        angles=-1.0 * parang + extra_rot,
        pca_number=pca_number,
        pca_sklearn=sklearn_pca,
        im_shape=im_shape,
        indices=None,  # maybe this should be none
    )

    res_stack = combine_residuals(
        method=residuals, res_rot=im_res_derot, residuals=im_res_rot, angles=parang
    )

    return res_stack

