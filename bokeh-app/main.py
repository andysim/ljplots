import numpy as np
from os.path import join, dirname

from bokeh.io import curdoc, show 
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Range1d, Select, Slider
from bokeh.plotting import figure, show

DATADIR=join(dirname(__file__), 'data/')

# Parse the parameter files
params = {}
with open(DATADIR+'nbdata.dat') as fp:
    for line in fp.readlines():
        sl = line.split()
        params[sl[0]] = list(map(float, sl[1:3]))

nbfixes = {}
with open(DATADIR+'nbfix.dat') as fp:
    for line in fp.readlines():
        sl = line.split()
        nbfixes[sl[0]+sl[1]] = list(map(float, sl[2:4]))


# Define C6 parameters
c6s = {}
for attype, vals in params.items():
    epsn = vals[0]
    rmin = vals[1]
    c6s[attype] = 8 * np.power(rmin, 3) * np.sqrt(2 * np.abs(epsn))


def lookup_nbfix(type1, type2):
    try:
        return nbfixes[type1 + type2]
    except KeyError:
        try:
            return nbfixes[type2 + type1]
        except KeyError:
            return None
    return type1+type2 if type1 > type2 else type2 + type1


def get_combined_params(type1, type2, allow_nbfix=False):
    nbfix_vals = lookup_nbfix(type1, type2)
    if nbfix_vals and allow_nbfix:
        epsn = -nbfix_vals[0]
        rmin = nbfix_vals[1]
    else:
        epsn = np.sqrt(np.abs(params[type1][0] * params[type2][0]))
        rmin = params[type1][1] + params[type2][1]
    return epsn, rmin


def compute_regular(R12invs, R6invs, epsn, rmin):
    rmin6 = np.power(rmin, 6)
    rmin12 = rmin6 * rmin6
    ljpot = epsn * (rmin12 * R12invs - 2 * rmin6 * R6invs)
    return ljpot


def compute_reciprocal(Rvals, R6invs, kappa, c6i, c6j):
    K2R2 = np.square(kappa * Rvals)
    K4R4 = np.square(K2R2)
    expterm = np.exp(-K2R2)
    return -c6i * c6j * (1.0 - expterm * (1.0 + K2R2 + 0.5 * K4R4)) * R6invs


def make_plot(plotmin):
    plot = figure(x_range=[Rmin,Rmax], y_range=[plotmin, 0.001], 
                  x_axis_label="Bond length (R)",
                  y_axis_label="Pair VDW energy (kcal/mol)",
                  plot_width=700)
    plot.title.text = "Click the legend titles to show/hide individual plots"

    x = np.linspace(Rmin,Rmax, 100)
    y = np.zeros_like(x)
    plot.line(x, y, color='gray', line_width=0.5)
    plot.line('x', 'y', color='blue', line_width=1.5, source=regularDdata,
              legend_label="Regular LJ")
    plot.line('x', 'y', color='green', line_width=1.5, source=switchedDdata,
              legend_label="Switched Regular LJ")
    plot.line('x', 'y', color='purple', line_width=1.5, source=nbfixedDdata,
              legend_label="NBFIXed LJ")
    plot.line('x', 'y', color='red', line_width=1.5, source=switchednbfixDdata,
              legend_label="Switched and NBFIXed LJ")
    plot.line('x', 'y', color='black', line_width=1.5, source=reciprocaldata,
              legend_label="Reciprocal space PME term")
    plot.legend.location = "bottom_right"
    plot.legend.click_policy="hide"
    return plot


def compute_data(type1, type2, Rcut, Rwin, kappa):
    data = {}
    epsn_fix, rmin_fix = get_combined_params(type1, type2, True)
    epsn_reg, rmin_reg = get_combined_params(type1, type2, False)
    Rvals = np.linspace(Rmin, Rmax, npts)
    Ron = Rcut - Rwin
    data['RvalsD'] = RvalsD = Rvals[Rvals <= Rcut] # Direct space range
    data['RvalsR'] = RvalsR = Rvals[Rvals >= Rcut] # Reciprocal space range
    R6invsD = np.power(RvalsD, -6)
    R6invsR = np.power(RvalsR, -6)
    R12invsD = np.square(R6invsD)
    data['regularlj'] = compute_regular(R12invsD, R6invsD, epsn_reg, rmin_reg)
    data['nbfixed'] = compute_regular(R12invsD, R6invsD, epsn_fix, rmin_fix)
    data['reciprocal'] = compute_reciprocal(RvalsR, R6invsR, kappa, c6s[type1], c6s[type2])
    shortrec = compute_reciprocal(RvalsD, R6invsD, kappa, c6s[type1], c6s[type2])
    denom = 1.0/np.max((np.power(Rwin,3), 0.0000001))
    s = np.where(RvalsD <= Ron, 1, np.square(Rcut-RvalsD)*(Rcut + 2*RvalsD - 3*Ron) *denom)
    data['switchedlj'] = s * data['regularlj'] + (1 - s)*shortrec
    data['switchednbfix'] = s * data['nbfixed'] + (1 - s)*shortrec

     # Evaluate values at the cutoff, just to set the plot's y range
    R6cutinv = np.power(1.0*Rcut, -6)
    R12cutinv = np.square(R6cutinv)
    regularmin = np.amin(data['regularlj'])
    nbfixedmin =  np.amin(data['nbfixed'])
    data['plotmin'] = 1.1*np.amin((regularmin, nbfixedmin))

    return data


def update_plot(attr, old, new):
    # Get the various potential functions
    type1 = type1_select.value
    type2 = type2_select.value
    Rcut = rcut_slider.value
    Rwin = rwin_slider.value
    kappa = kappa_slider.value
    d = compute_data(type1, type2, Rcut, Rwin, kappa)
    regularDdata.data = ColumnDataSource(data=dict(x=d['RvalsD'], y=d['regularlj'])).data
    switchedDdata.data = ColumnDataSource(data=dict(x=d['RvalsD'], y=d['switchedlj'])).data
    nbfixedDdata.data = ColumnDataSource(data=dict(x=d['RvalsD'], y=d['nbfixed'])).data
    switchednbfixDdata.data = ColumnDataSource(data=dict(x=d['RvalsD'], y=d['switchednbfix'])).data
    reciprocaldata.data = ColumnDataSource(data=dict(x=d['RvalsR'], y=d['reciprocal'])).data
    plot.y_range.start = d['plotmin']



# Plot range and resolution settings
Rmin = 2
Rmax = 14
npts = 200

# Initial values to use
type1 = "SODD";
type2 = "CLAD"
Rcut = 8
Rwin = 2
kappa = 0.4

type1_select = Select(value=type1, title='Atom1 type', options=sorted(params.keys()))
type2_select = Select(value=type2, title='Atom2 type', options=sorted(params.keys()))
rcut_slider = Slider(title="Real space cutoff (A)", value=Rcut, start=Rmin, end=Rmax, step=0.5)
rwin_slider = Slider(title="Switching window size (A)", value=Rwin, start=0, end=6, step=0.5)
kappa_slider = Slider(title="Ewald attenuation parameter (1/A)", value=kappa, start=0, end=1, step=0.01)

type1_select.on_change('value', update_plot)
type2_select.on_change('value', update_plot)
rcut_slider.on_change('value', update_plot)
rwin_slider.on_change('value', update_plot)
kappa_slider.on_change('value', update_plot)

d = compute_data(type1, type2, Rcut, Rwin, kappa)

regularDdata = ColumnDataSource(data=dict(x=d['RvalsD'], y=d['regularlj']))
switchedDdata = ColumnDataSource(data=dict(x=d['RvalsD'], y=d['switchedlj']))
nbfixedDdata = ColumnDataSource(data=dict(x=d['RvalsD'], y=d['nbfixed']))
switchednbfixDdata = ColumnDataSource(data=dict(x=d['RvalsD'], y=d['switchednbfix']))
reciprocaldata = ColumnDataSource(data=dict(x=d['RvalsR'], y=d['reciprocal']))
plot = make_plot(d['plotmin'])

controls = column(type1_select, type2_select, rcut_slider, rwin_slider, kappa_slider)
curdoc().add_root(row(plot, controls))
