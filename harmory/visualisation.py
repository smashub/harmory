"""
Basic utilities for plotting structural elements and annotations.

Notes: XXX this will be moved to a separate [library currently in preparation].
"""
import random

import jams
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def preprocess_annotation(annotation, sr=1, remove_digits=False, index=False):
    """
    Preprocess annotations by resampling, removal of digits and reindexing.

    Parameters
    ----------
    ann : list of lists
        A LAB-like annotation with start and end time
    sr : float
        Sampling rate (Default value = 1)
    remove_digits : bool
        Whether digits in the labels will be removed or not.
    index : int
        Whether to round to nearest integer.

    Returns
    -------
    new_annotation : list of lists
        The preprocessed annotation as a new list.

    """
    new_annotation = []
    for observation in annotation:
        s, t = observation[0] * sr, observation[1] * sr
        if index:  # round to the nearest integer, if needed
            s, t = int(np.round(s)), int(np.round(t))
        # Removing numbered occurrences from segment labels
        label = ''.join([i for i in observation[2] if not i.isdigit()]) \
            if remove_digits else observation[2]
        label = label.replace("Silence", "Z").replace("silence", "z")
        new_annotation.append([s, t, label])

    return new_annotation


def set_plotting_preferences(ax, **kwargs):
    """
    Sets the given axes object according to the given preferences, which are
    provided as named parameters to this function.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object that will be set.
    **kwargs : dict
        Named parameters as plotting preferences.

    """
    def check_kwarg(keyword):
        return keyword in kwargs \
            and kwargs[keyword] is not None

    if check_kwarg('xlabel'):
        ax.set_xlabel(kwargs['xlabel'])
    if check_kwarg('ylabel'):
        ax.set_ylabel(kwargs['ylabel'])
    if check_kwarg('title'):
        ax.set_title(kwargs['title'])
    if check_kwarg('xlim'):
        ax.set_xlim(kwargs['xlim'])
    if check_kwarg('ylim'):
        ax.set_ylim(kwargs['ylim'])

    return ax


def assign_segment_colours(segment_annotation):
    """
    Generate a mapping from segment labels to colours for plotting.

    Notes
    -----
    - Parameterise the choice of the colourmap;
    - Parameterise the random behaviour;
    
    """
    colours = list(matplotlib.cm.get_cmap('tab20c').colors)
    random.shuffle(colours)  # avoid similar colours for labels
    label_set = sorted(set([label for _, _, label in segment_annotation]))
    # TODO Remove numbers from labels to reduce them
    assert len(label_set) <= len(colours), "Not enough colours for labels"
    colour_dict = {label: col for label, col in zip(label_set, colours)}
    return colour_dict


def plot_segments(annotation, ax=None, figsize=(6, 1), direction='h',
    time_min=None, time_max=None, nontime_min=0, nontime_max=1, 
    time_axis=True, nontime_axis=False, time_label=None, swap_time_ticks=False,
    colour_dict=None, edgecolor='k', axis_off=False, dpi=72,
    adjust_time_axislim=True, adjust_nontime_axislim=True, alpha=None,
    print_labels=True, label_ticks=False, tick_boundaries=False, **kwargs):
    """
    Creates a multi-line plot to temporally  display structural annotations.

    Parameters
    ----------
    annotation : list
        A List of tuples like ``[(start_position, end_position, label), ...]``
    ax : matplotlib.axes.Axes
        Optional Axes instance to reuse to plot on. If None, new figure and axes
        will be created and used for this plot, using the other parameters.
    figsize : tuple
        Size of the figure as a (width, height) tuple, expressed in inches.
    direction: str
        Orientation of the figure, either 'v' (vertical) or 'h' (horizontal).
    colour_dict : dict
        Optional mapping from unique segment labels to RGB colours.
    time_min : float 
        Optional minimal limit for time axis. If `None`, will be min annotation.
    time_max : float 
        Optional maximal limit for time axis. If `None`, will be max annotation.
    nontime_min : float
        Minimal limit for non-time axis (analogous to time_min).
    nontime_max : float
        Maximal limit for non-time axis (analogous to time_min).
    time_axis : bool  
        Whether to display the time axis ticks or not.
    nontime_axis : bool 
        Whether to display the non-time axis ticks or not.
    time_label : str
        Label that will be displayed alognside the time axes.
    swap_time_ticks : bool
        Orientation of the time ticks; up for horizontal and left for vertical.
    edgecolor : str
        Name of the colour to use for edgelines of segment box.
    axis_off : bool
        Whether to switch off the axis frame (calls `ax.axis('off')` if true).
    dpi : int
        Resolution of the figure, expressed in dots per inch.
    adjust_time_axislim : bool
        Whether to adjust the time-axis. Usually `True` for plotting on
        standalone axes and `False` for overlay plotting.
    adjust_nontime_axislim : bool
        Whether to adjust non-time-axis (analogous to `adjust_time_axislim`).
    alpha : int
        Alpha value for the rectangles used as bounding boxes for segments.
    print_labels : bool
        Whether to print segment labels inside their rectangles.
    label_ticks : bool
        Whether to print labels as ticks with onset alignment.

    Returns
    -------
    fig : matplotlib.figure.Figure
        A new matplotlib figure, if new ax created, or None if ax was given.
    ax : matplotlib.axes.Axes
        The Axes object that was eventually used in this function.

    """
    if direction not in ['v', 'h']:
        raise ValueError("Direction can be vertical (v) or horizontal (h)")
    # annot = check_segment_annotations(annot)

    if 'color' not in kwargs:
        kwargs['color'] = 'k'
    if 'weight' not in kwargs:
        kwargs['weight'] = 'bold'
        # kwargs['weight'] = 'normal'
    if 'fontsize' not in kwargs:
        kwargs['fontsize'] = 12
    if 'ha' not in kwargs:
        kwargs['ha'] = 'center'
    if 'va' not in kwargs:
        kwargs['va'] = 'center'

    if colour_dict is None:  # assign colour to labels from palette
        colour_dict = assign_segment_colours(annotation)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi) \
        if ax is None else (None, ax)

    nontime_width = nontime_max - nontime_min
    nontime_middle = nontime_min + nontime_width / 2
    all_time_middles = []

    for start, end, label in annotation:
        time_width = end - start
        time_middle = start + time_width / 2
        all_time_middles.append(time_middle)

        if direction == 'h':
            rect = matplotlib.patches.Rectangle(
                (start, nontime_min), time_width, nontime_width,
                facecolor=colour_dict[label], edgecolor=edgecolor, alpha=alpha)
            ax.add_patch(rect)
            if print_labels:
                ax.annotate(label, (time_middle, nontime_middle), **kwargs)
        else:
            rect = matplotlib.patches.Rectangle(
                (nontime_min, start), nontime_width, time_width,
                facecolor=colour_dict[label], edgecolor=edgecolor, alpha=alpha)
            ax.add_patch(rect)
            if print_labels:
                ax.annotate(label, (nontime_middle, time_middle), **kwargs)

    if time_min is None:
        time_min = min(start for start, end, label in annotation)
    if time_max is None:
        time_max = max(end for start, end, label in annotation)

    if direction == 'h':
        if adjust_time_axislim:
            ax.set_xlim([time_min, time_max])
        if adjust_nontime_axislim:
            ax.set_ylim([nontime_min, nontime_max])
        if not nontime_axis:
            ax.set_yticks([])
        if not time_axis:
            ax.set_xticks([])
        if swap_time_ticks:
            ax.xaxis.tick_top()
        if time_label:
            ax.set_xlabel(time_label)
        if label_ticks:
            ax.set_xticks(all_time_middles)
            ax.set_xticklabels([label for start, end, label in annotation])
        if tick_boundaries:
            ax.set_xticks([s for s, e, l in annotation])

    else:  # vertical orientation expected otherwise
        if adjust_time_axislim:
            ax.set_ylim([time_min, time_max])
        if adjust_nontime_axislim:
            ax.set_xlim([nontime_min, nontime_max])
        if not nontime_axis:
            ax.set_xticks([])
        if not time_axis:
            ax.set_yticks([])
        if swap_time_ticks:
            ax.yaxis.tick_right()
        if time_label:
            ax.set_ylabel(time_label)
        if label_ticks:
            ax.set_yticks(all_time_middles)
            ax.set_yticklabels([label for start, end, label in annotation])
        if tick_boundaries:
            ax.set_yticks([s for s, e, l in annotation])

    if axis_off:
        ax.axis('off')
    if fig is not None:
        plt.tight_layout()

    return fig, ax


def plot_segments_overlay(*args, **kwargs):
    """
    Plot segment annotations as overlay on the given/current axis.
    """
    assert 'ax' in kwargs
    ax = kwargs['ax']

    if 'adjust_time_axislim' not in kwargs:
        kwargs['adjust_time_axislim'] = False
    if 'adjust_nontime_axislim' not in kwargs:
        kwargs['adjust_nontime_axislim'] = False
    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.3
    if 'edgecolor' not in kwargs:
        kwargs['edgecolor'] = None
    if 'nontime_axis' not in kwargs:
        kwargs['nontime_axis'] = True

    if 'direction' in kwargs and kwargs['direction'] == 'vertical':
        kwargs['nontime_min'], kwargs['nontime_max'] = ax.get_xlim()
    else:
        kwargs['nontime_min'], kwargs['nontime_max'] = ax.get_ylim()

    fig, ax = plot_segments(*args, **kwargs)

    return fig, ax


def compressed_gray_cmap(alpha=5, rgb_n=256, reverse=False):
    """
    Creates a logarithmically or exponentially compressed grayscale colormap.

    Parameters
    ----------
    alpha : float
        The compression factor. If alpha > 0, it performs log compression
        (enhancing black colors). If alpha < 0, it performs exp compression
        (enhancing white colors). Raises an error if alpha = 0.
    rgb_n : int
        The number of RGB quantisation levels (matplotlib uses 256).
    reverse : bool
        Whether to invert the cmap (black to white) or not (black to white).

    Returns
    -------
    color_wb (mpl.colors.LinearSegmentedColormap): the compressed colormap.

    """
    if alpha == 0:  # sanity check of alpha: either positive or negative
        raise ValueError("Parameter alpha cannot be null.")

    gray_values = np.log(1 + abs(alpha) * np.linspace(0, 1, rgb_n))
    gray_values /= gray_values.max()  # max normalisation
    gray_values = 1 - gray_values if alpha > 0 else gray_values[::-1]
    gray_values = gray_values[::-1] if reverse else gray_values

    gray_values_rgb = np.repeat(gray_values.reshape(rgb_n, 1), 3, axis=1)
    color_wb = matplotlib.colors.LinearSegmentedColormap.from_list(
        'color_wb', gray_values_rgb, N=rgb_n) # create a linear colourmap

    return color_wb


def plot_univariate_signal(x, sr=1, times=None, ax=None, figsize=(6, 2), dpi=72,
        xlabel='Time (seconds)', ylabel='', title='', ylim=True, color="gray"):
    """
    Plot a signal, e.g. a waveform or a novelty function, as a line plot.

    Parameters
    ----------
    x : np.array
        The monovariate (1D) input signal to visualise.
    sr : int
        The sample rate used to obtain x (Default value = 1).
    times : list
        Optional time coeffients to use as labels; otherwise, the coefficients
        will be inferred from the sample rate.
    ax : matplotlib.axes.Axes
        The Axes instance to plot on, if provided; a new figure with axes will
        be created otherwise (its handles are returned).
    figsize : tuple
        Size of the figure as a (width, height) tuple, expressed in inches.
    xlabel (str)
        An optional label to use for the x-axis, time assumed otherwise.
    ylabel (str)
        An optional label to use for the y-axis, empty otherwse.
    title (str):
        An optional title to use for the plot, no title otherwise.
    dpi (int):
        Resolution of the figure, expressed in dots per inch.
    ylim : bool or tuple
        `True` if auto adjusting ylim, or use a custom tuple with actual ylim.
    **kwargs: dict
        Keyword arguments for matplotlib.pyplot.plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure or None if ax was given.
    ax : matplotlib.axes.Axes
        The used axes.
    line : list of `matplotlib.lines.Line2D`
        The created line plot for the given signal.

    """
    # Creating the plot and instantiating times, if needed
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi) \
        if ax is None else (None, ax)
    times = np.arange(x.shape[0]) / sr \
        if times is None else times
    ax.set_xlim([times[0], times[-1]])

    line = ax.plot(times, x, color=color)

    if ylim is True:  # auto-setting of y-limits based on x
        ylim_x = x[np.isfinite(x)]
        x_min, x_max = ylim_x.min(), ylim_x.max()
        x_max = x_max + 1 if x_max == x_min else x_max
        #
        ax.set_ylim([min(1.1 * x_min, 0.9 * x_min),
                     max(1.1 * x_max, 0.9 * x_max)])
    elif len(ylim) == 2:  # custom limits are provided as tuple
        ax.set_ylim(ylim)

    ax = set_plotting_preferences(
        ax, title=title, xlabel=xlabel, ylabel=ylabel)

    if fig is not None:
        plt.tight_layout()

    return fig, ax, line


def plot_matrix(X, x_sr=1, y_sr=1, x_coeff=None, y_coeff=None, ax=None,
    xlabel='Time (s)', ylabel='Frequency (Hz)', xlim=None, ylim=None, clim=None,
    dpi=72, figsize=(6, 3), cbar=False, cbar_aspect=20.0, cbar_label='', **kwargs):
    """
    Plot a feature matrix (e.g spectrogram, tempogram) or a simil matrix.

    Parameters
    ----------
    X : np.array
        The matrix to visualise given as 2D numpy array.
    x_sr : int
        Sample rate of the feature in axis 1 (e.g. for temporal frames).
    y_sr : int
        Sample rate of the feature in axis 0 (e.g. for frequency bands).
    x_coeff: list or np.array
        Temporal coeffients to use for the dimension represented in axis 1;
        if not provided, it will be computed based on `x_sr`.
    y_coeff : list or np.array
        Temporal coeffients to use for the dimension represented in axis 1;
        if not provided, it will be computed based on `y_sr`.
    xlabel : str
        An optional label the for x-axis, e.g. 'Time (seconds)'.
    ylabel : str
        An optional label the for y-axis, e.g. 'Frequency (Hz)'.
    xlim : 2-tuple
        Custom limits for the x-axis, none are used otherwise.
    ylim : 2-tuple
        Custom limits for the y-axis,  none are used otherwise.
    clim : 2-tuple
        Color limits for the current image, in the form (vmin, vmax); as per
        matplotlib, if either vmin or vmax is None, the image min/max
        respectively will be used for color scaling.
    dpi : int
        Resolution of the figure, expressed in dots per inch.
    cbar : bool
        Whether to create a colorbar next to the matrix plot.
    cbar_aspect : float
        The aspect of the colourbar if the colourbar is requested (`cbar`).
    cbar_label : str
        An optional label for the colourbar, none is used otherwise.
    ax : plt.Axes, or a collection of them
        Whether to use the given axes to plot on. Possible argument types
        supported: (i) a collection of two axes (the first is used to plot
        the matrix, the second for the colourbar); (ii) a collection with a
        single axes (used for matrix); if no axes is provided, a new is created.
    figsize : 2-tuple
        Size of the figure as a (width, height) tuple, expressed in inches.
    **kwargs : dict
        Keyword arguments that will be passed to the plotting function.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure, or `None` if `ax` was already given.
    ax : matplotlib.axes.Axes
        The used axes, either created from scratch or the same that was given.

    """
    # Creating the plot and instantiating coefficients, if needed.
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi) \
        if ax is None else (None, ax)
    ax = [ax] if not isinstance(ax, list) else ax
    if x_coeff is None:
        x_coeff = np.arange(X.shape[1]) / x_sr
    if y_coeff is None:
        y_coeff = np.arange(X.shape[0]) / y_sr

    # Setting default plotting preferences for a set of controls
    # passed to imshow (more can be provided).
    if 'extent' not in kwargs:
        x_start_ext = (x_coeff[1] - x_coeff[0]) / 2
        x_end_ext = (x_coeff[-1] - x_coeff[-2]) / 2
        y_ext1 = (y_coeff[1] - y_coeff[0]) / 2
        y_ext2 = (y_coeff[-1] - y_coeff[-2]) / 2
        kwargs['extent'] = [
            x_coeff[0] - x_start_ext, x_coeff[-1] + x_end_ext,
            y_coeff[0] - y_ext1, y_coeff[-1] + y_ext2]

    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'gray_r'
    if 'aspect' not in kwargs:
        kwargs['aspect'] = 'auto'
    if 'origin' not in kwargs:
        kwargs['origin'] = 'lower'

    im = ax[0].imshow(X, **kwargs)  # main plot, on first axes
    # Plotting the colourbar, if required.
    if len(ax) == 2 and cbar:
        cbar = plt.colorbar(im, cax=ax[1])
        cbar.set_label(cbar_label)
    elif len(ax) == 1 and cbar:
        plt.sca(ax[0])
        cbar = plt.colorbar(im, aspect=cbar_aspect)
        cbar.set_label(cbar_label)

    # Setting additional plotting preferences
    ax = set_plotting_preferences(
        ax[0], xlim=xlim, ylim=ylim,
        xlabel=xlabel, ylabel=ylabel
    )
    if clim is not None:
        im.set_clim(clim)
    if fig is not None:
        plt.tight_layout()

    return fig, ax


def plot_feature_ssm(X, sr_X, S, sr_S, ann, duration, colour_dict=None,
    label='Time (s)', time=True, figsize=(5, 6), fontsize=10,
    clim_X=None, clim=None, tick_boundaries=False):
    """
    Plot the simil matrix (SM) alongside the corresponding feature
    representation and the given annotations.

    Parameters
    ----------
    X : np.array (or list?)
        Feature representation on which the SM was computed. If not provided, no
        header is displayed on top of the SM.
    sr_X : int
        Sample rate to consider for the given signal/feature X.
    S : np.array
        The simil matrix to display, given as a 2D numpy array.
    sr_S : int
        Sample rate of the given simil matrix.
    ann : list
        List of segment annotations, in the form (... ... label).
    duration : float
        The duration spanned by the feature representation and the SM.
    colour_dict : dict
        Optional mapping from segment labels to colours for side plots.
    label : str
        Label to use for time axes, defaulting to 'Time (seconds)'.
    time : bool
        Whether to tisplay time axis ticks or not.
    figsize : tuple
        Size of the figure as a (width, height) tuple, expressed in inches.
    fontsize : int
        Font size to use for segment labels.
    clim_X :
        Optional color limits for matrix `X`, in the form (vmin, vmax); as per
        `matplotlib`, if either vmin or vmax is None, the image min/max
        respectively will be used for color scaling.
    clim : 
        Optional color limits for matrix `S` (analogous to `clim_X`).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the created figure object.
    ax: matplotlib.axes.Axes
        Handle for the created axes object.

    Notes
    -----
    - X Can be generalised to any arbitrary signal, rather than acoustic ones.

    """
    gridspec_kw={'width_ratios': [0.1, 1, 0.05],
                 'wspace': 0.2,
                 'height_ratios': [0.3, 1, 0.1]}

    cmap = compressed_gray_cmap(alpha=-10)  # generate logarithmic colourmap
    fig, ax = plt.subplots(3, 3, gridspec_kw=gridspec_kw, figsize=figsize)
    # First to be plotted: the feature matrix on top-centre location
    plot_matrix(X, x_sr=sr_X, ax=[ax[0, 1], ax[0, 2]],
                clim=clim_X, xlabel='', ylabel='')

    ax[0, 0].axis('off')  # deactivate empty axes area (top-left)
    # Now time to plot the actual simil matrix at the centre + cmap?
    plot_matrix(S, x_sr=sr_S, y_sr=sr_S, ax=[ax[1, 1], ax[1, 2]], cmap=cmap,
                clim=clim, xlabel='', ylabel='', cbar=True)
    # Finally, plot the segment annotations alongside the SM and drop empty axes
    ax[1, 1].set_xticks([]), ax[1, 1].set_yticks([])
    plot_segments(ann, ax=ax[2, 1], time_axis=time, fontsize=fontsize,
                  direction='h', colour_dict=colour_dict,
                  time_label=label, time_max=duration*sr_X,
                  tick_boundaries=tick_boundaries)
    ax[2, 2].axis('off'), ax[2, 0].axis('off')
    plot_segments(ann, ax=ax[1, 0], time_axis=time, fontsize=fontsize,
                  direction='v', colour_dict=colour_dict,
                  time_label=label, time_max=duration*sr_X,
                  tick_boundaries=tick_boundaries)

    return fig, ax


def plot_peak_positions_overlay(x, sr_x, peaks, title='', figsize=(8,2)):
    """
    Plot peak positions on top of the given novelty curve.

    Parameters
    ----------
    x : np.ndarray
        The novelty curve that will be displayed under the peaks.
    sr_x : int
        The sample rate of the novelty curve `x`.
    peaks : np.ndarray
       A collection or array of peak positions relative to `x`.
    title : str, optional
        Title for the plot, usually corresponding to the peak detection method.
    figsize : tuple, optional
        Size of the figure as a (width, height) tuple, expressed in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the created figure object.
    ax: matplotlib.axes.Axes
        Handle for the created axes object.
    
    """
    peaks_sec = peaks / sr_x
    fig, ax, line = plot_univariate_signal(
        x, sr_x, figsize=figsize, color='k', title=title)
    plt.vlines(peaks_sec, 0, 1.1, color='r', linestyle=':', linewidth=1)

    return fig, ax