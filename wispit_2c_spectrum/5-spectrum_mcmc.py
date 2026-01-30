from species import SpeciesInit
from species.data.database import Database
from species.fit.fit_model import FitModel
from species.plot.plot_mcmc import plot_posterior
from species.read.read_model import ReadModel
from species.plot.plot_spectrum import plot_spectrum
from species.util.fit_util import get_residuals
#import matplotlib.pyplot as plt
from uncertainties import ufloat
from species.phot.syn_phot import SyntheticPhotometry

spec_file = f'./input/spectrum.fits'
modelname = 'bt-settl'

parallax = ufloat(7.464925176349439, 0.021432146)
# star magnitude
app_star = ufloat(8.591, 0.078)
contrast_2c = ufloat(7.39, 0.21)
app_2c = app_star + contrast_2c
# MagAO-X
app_2c_zp = ufloat(19.40, 0.65)
# LBT
app_2c_l = ufloat(14.80, 0.76)
# add sphere data point
synphot = SyntheticPhotometry('2MASS/2MASS.H')
flux, error = synphot.magnitude_to_flux(app_2c.n, error=app_2c.s)
print(f'Flux (W m-2 um-1) = {flux:.2e} +/- {error:.2e}')

SpeciesInit()
db = Database()

db.add_filter('2MASS/2MASS.Ks')
db.add_filter('2MASS/2MASS.H')
db.add_filter('LBT/LMIRCam.L_77K')
db.add_filter('LCO/MagAOX.Sci1-zp')

db.add_object(
    object_name='Candidate-1',
    app_mag={'2MASS/2MASS.H':(app_2c.n, app_2c.s),
             'LCO/MagAOX.Sci1-zp':(app_2c_zp.n,app_2c_zp.s),
             'LBT/LMIRCam.L_77K':(app_2c_l.n,app_2c_l.s)},
    parallax=(parallax.n, parallax.s),
    spectrum={
        'MyInstr': (
            spec_file,
            None,
            500.0  # instrument resolution
        )
    }
)

fit = FitModel(
    object_name='Candidate-1',
    model=modelname,
    bounds={
        'teff': (1000., 2000.),
        'logg': (4.0, 4.0),
        'radius': (0.5, 2.5),
        'feh': (-1.0, 0.0),  # dont think exorem goes below -0.5 in grid
        'c_o_ratio': (0.0, 1.0)  # dont think exorem goes above 1.0 in grid
    },
    normal_prior={

    },
    inc_phot=True,
    inc_spec=True,
    apply_weights=True,  # Relative weighting photometry vs spectra
)

TAG = 'cand1_nest_1'
fit.run_dynesty(tag=TAG, n_live_points=int(1e4),
                sample_method='rwalk', evidence_tolerance=0.01)

# Corner plot from posterior
plot_posterior(
    tag=TAG,
    inc_mass=True,
    inc_luminosity=True,
    offset=(-0.25, -0.25),
    output='candidate_corner.pdf',
)
print("Saved corner plot to candidate_corner.pdf")

best = db.get_median_sample(tag=TAG)

wanted_filters = [
    '2MASS/2MASS.H',
    'LBT/LMIRCam.L_77K',
    'LCO/MagAOX.Sci1-zp'
]

# Read the model spectrum at best-fit params
readmodel = ReadModel(modelname, teff_range=(1000, 2200))

modelbox = readmodel.get_model(
    model_param=best,  # Directly using posterior median
    spec_res=500.0,
    smooth=True
)

objectbox = db.get_object(
    object_name='Candidate-1',
    inc_phot=wanted_filters,
    inc_spec=True
)

readmodel = ReadModel(modelname, wavel_range=(0.5, 5))
modelbox = readmodel.get_model(best, spec_res=500.0, smooth=True)

residuals = get_residuals(
    tag=TAG,
    spectrum=modelname,
    parameters=best,
    objectbox=objectbox,
    inc_phot=True,  # include photometric residuals
    inc_spec=True  # include spectral residuals
)

line_colors = [
    "#4A90E2",  # sky blue
    "#E77CB4",  # rose pink
    "#93E2D5",  # mint
    "#C5A3FF",  # lavender
    "#8EC9F3",  # baby blue
    "#A57CE6",  # lilac
]

photo_style = {f: {'marker': 'D', 'ms': 4., 'ls': 'none'} for f in wanted_filters}

fig1 = plot_spectrum(
    boxes=[modelbox, objectbox],
    # filters=wanted_filters,
    residuals=residuals,
    plot_kwargs=[
        # Model spectrum
        {'ls': '-', 'lw': 1., 'color': 'black', 'zorder': 1},

        # Object data
        {
            'MyInstr': {
                'ls': '-',
                'lw': 0.8,
                'marker': None,
                'alpha': 0.9,
                'color': '#E77CB4',
                'label': 'GRAVITY K-band'
            },
            'LBT/LMIRCam.L_77K': {
                'marker': 'D',
                'ms': 5.,
                'color': '#4A90E2',
                'ls': 'none',
                'label': 'LMIRCam L-band'
            },
            '2MASS/2MASS.H': {
                'marker': 'D',
                'ms': 5.,
                'color': '#93E2D5',
                'ls': 'none',
                'label': 'SPHERE H-band'
            },
            'LCO/MagAOX.Sci1-zp': {
                'marker': 'D',
                'ms': 5.,
                'color': '#C5A3FF',
                'ls': 'none',
                'label': "MagAO-X z'-band"
            }
        }
    ],
    xlim=(0.8, 4.1),
    ylim=(-0.5e-16, 8e-16),
    legend=[None,
            {'loc': 'upper right', 'frameon': False, 'fontsize': 12.}],
    units=('um', 'W m-2 um-1'),
    figsize=(6, 3.5),
    output=None
)

ax_main, ax_resid = fig1.axes
ax_main.tick_params(labelbottom=False)
ax_resid.set_xlabel("Wavelength (µm)")

fig1.savefig("Spectrum.png", bbox_inches="tight", dpi=1000)

# Zoomed in fig
fig2 = plot_spectrum(
    boxes=[modelbox, objectbox],
    residuals=None,
    plot_kwargs=[
        {'ls': '-', 'lw': 1., 'color': 'black', 'zorder': 1},
        {
            'MyInstr': {
                'ls': '-',
                'lw': 0.8,
                'marker': None,
                'alpha': 0.9,
                'color': '#E77CB4',
                'label': 'GRAVITY K-band'
            }
        }
    ],
    xlim=(2.2, 2.4),
    ylim=(-0.5e-16, 3e-16),
    legend=[None,
            {'loc': 'upper right', 'frameon': False, 'fontsize': 12.}],
    units=('um', 'W m-2 um-1'),
    figsize=(6, 3.5),
    output=None
)

fig2.patch.set_facecolor('none')
fig2.axes[0].set_facecolor('none')
for spine in fig2.axes[0].spines.values():
    spine.set_linewidth(1.2)

fig2.savefig("Zoomed_spectrum.png", dpi=1000, bbox_inches="tight")
print("best fit parameters:", best)