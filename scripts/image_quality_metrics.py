"""Apply thesis section 2.2.4 to the 37 saved final migration cases.

Run with ``python -m scripts.image_quality_metrics``. Edit the generated
regions.json and rerun with ``--cases 8 12`` to revise particular cases.
All measurements use full-resolution arrays; PNGs are review previews.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.signal import hilbert


PROJECT = Path(__file__).resolve().parents[1]
SIMULATIONS = PROJECT / 'outputs/simulations'
OUTPUT = PROJECT / 'outputs/analysis/image_quality_20261005'
BOUNDARIES = ('reflect_sides_bottom', 'reflect_bottom', 'reflect_sides', 'absorb_all')
MODELS = ('t_b1', 'y_b1', 'y_b2', 'y_b3', 'y_b4')
THRESHOLD = 10 ** (-6 / 20)
COLORS = {'api': '#ff43cf', 'signal': '#37eb62', 'noise': '#27d9ef'}
CONVENTIONS = {
    'source_pdf': '/home/heitor/repos/tcc/build/main.pdf',
    'source_section': '2.2.4, equations 23–27',
    'input': 'Final source-energy-normalized coherent FMC stacks only.',
    'amplitude': 'abs(scipy.signal.hilbert(signed_image, axis=0))',
    'power': 'Region statistics use amplitude**2 before converting the final ratios to dB.',
    'contrast': 'mean(signal power) / mean(noise power)',
    'contrast_db': '10 * log10(contrast); reported contrast in dB. contrast retains the linear power ratio.',
    'cnr': 'abs(mean(signal power) - mean(noise power)) / std(noise power, ddof=0)',
    'cnr_db': '20 * log10(cnr); reported CNR in dB. cnr retains the linear ratio.',
    'db_nonfinite': 'Zero or undefined linear ratios have null dB values. A zero ratio corresponds to -infinity dB.',
    'noise_region': 'Nearby rectangular background patches. Pixels from multiple patches are pooled for the background mean and standard deviation.',
    'sigma_o_interpretation': 'Standard deviation of background POWER, consistent with the power means in equations 24–26.',
    'a_minus_6db': 'All amplitude >= reflector_window_peak * 10**(-6/20) within api_roi_mm; disconnected areas are included.',
    'api': 'a_minus_6db_mm2 / (sound_speed_m_s / frequency_hz * 1000)**2',
    'api_domain': 'User-selected reflector window (api_roi_mm). Full-image area/API are retained separately. Real-data PML padding is cropped as in the existing workflow.',
    'api_interpretation': 'Multi-target images, cracks and extended interfaces yield a total footprint, not a single point-reflector resolution measurement.',
    'region_status': 'Proposed until reviewed by the user. Standard and Poynting share the same signal/noise masks in each case.',
}


def cases():
    """Keep the case numbers shown to the user stable."""
    folders = ['synthetic', 'synthetic/poynting_good_result']
    folders += [f'synthetic/td_rtm_full_fmc_20260917/{model}' for model in MODELS]
    folders += [
        'real/immersion_fmc_rot_Xdeg.m2k',
        'real/meia_lua_fmc.m2k',
        'real/before_refinement_20260915/immersion_fmc_rot_Xdeg.m2k',
        'real/before_refinement_20260915/meia_lua_fmc.m2k',
        'real/timing_corrected_20260916/immersion_fmc_rot_Xdeg.m2k',
        'real/bottom_reflector_20260917/immersion_fmc_rot_Xdeg.m2k',
    ]
    folders += [f'synthetic/td_rtm_boundary_comparison_20260918/{model}/{boundary}'
                for model in MODELS for boundary in BOUNDARIES]
    folders += [f'real/panther_boundary_comparison_20260924/{boundary}'
                for boundary in BOUNDARIES]
    return {str(i): folder for i, folder in enumerate(folders, 1)}


def result_paths(folder):
    folder = Path(folder)
    if (folder / 'standard_normalized.npy').exists():
        return {method: folder / f'{method}_normalized.npy'
                for method in ('standard', 'poynting')}
    if (folder / 'rtm_normalized.npy').exists():
        return {'standard': folder / 'rtm_normalized.npy'}
    data = folder / ('SyntheticRTM' if 'synthetic' in folder.parts else 'ReverseTimeMigration') / 'data'
    return {'standard': data / 'accumulated_product_normalized_fmc.npy',
            'poynting': data / 'accumulated_product_poynting_normalized_fmc.npy'}


def grid(folder, shape):
    """Return the same physical image domain used by the saved workflow plots."""
    if 'td_rtm_full_fmc_20260917' in folder.parts or 'td_rtm_boundary_comparison_20260918' in folder.parts:
        config_root = next(p for p in folder.parents if (p / 'config.json').exists())
        config = json.loads((config_root / 'config.json').read_text())
        config = config.get('input_config', config)
        dx = config['dx_m'] * 1000
        dz = config['dz_m'] * 1000
        return np.arange(shape[1]) * dx - 20, np.arange(shape[0]) * dz, (slice(None), slice(None))
    if 'real' in folder.parts:
        root = folder if (folder / 'x_m.npy').exists() else folder.parent
        crop = (slice(45, -45), slice(45, -45))
        return (np.load(root / 'x_m.npy')[crop[1]] * 1000,
                np.load(root / 'z_m.npy')[crop[0]] * 1000, crop)
    # Legacy synthetic images were already cropped by 45 cells on every side.
    return np.arange(shape[1]) * 1.5, np.arange(shape[0]) * 1.5, (slice(None), slice(None))


def defaults(case_id, folder, x, z):
    base = {'case_id': int(case_id), 'source_folder': str(folder.relative_to(SIMULATIONS)),
            'review_status': 'proposed', 'frequency_hz': 5e6, 'api_metadata_status': 'saved_metadata',
            'notes': []}
    if 'td_rtm_full_fmc_20260917' in folder.parts or 'td_rtm_boundary_comparison_20260918' in folder.parts:
        model = next(part for part in folder.parts if part in MODELS)
        mask = SIMULATIONS / 'synthetic/td_rtm_full_fmc_20260917' / model / 'defect_mask.npy'
        base.update(sound_speed_m_s=6320., target_type='extended crack',
                    cnr_roi_mm=[-8, 8, 24, 40],
                    signal={'kind': 'distance_to_mask', 'mask_path': str(mask.relative_to(PROJECT)),
                            'max_distance_mm': .25},
                    noise={'kind': 'rectangles', 'bounds_mm': [[2.5, 7.8, 25, 28]]})
    elif 'real' in folder.parts:
        root = folder if (folder / 'run_config.json').exists() else folder.parent
        config_path = root / ('run_config.json' if (root / 'run_config.json').exists() else 'config.json')
        config = json.loads(config_path.read_text())
        if isinstance(config.get('acquisition'), dict):
            config = config['acquisition']
        base['sound_speed_m_s'] = config['specimen_speed_m_s']
        if config['inspection_type'] == 'contact':
            fit = json.loads((PROJECT / 'outputs/analysis/panther_diagnostics/diagnostics.json').read_text())['contact_arc_fit']
            base.update(target_type='extended curved echo', cnr_roi_mm=[-75, 75, 10, 82],
                        signal={'kind': 'arc_band', 'center_mm': [fit['center_x_mm'], fit['center_z_mm']],
                                'radius_mm': fit['radius_mm'], 'half_width_mm': 1.5},
                        noise={'kind': 'rectangles', 'bounds_mm': [[-5, 5, 59, 69]]})
            base['notes'].append('The signal band follows the visible curved echo; this is not a surveyed point defect.')
        else:
            diagnostic = json.loads((PROJECT / 'outputs/analysis/immersion_validation_20260916/diagnostics.json').read_text())['image_metrics']
            corrected = bool(config.get('timing_correction'))
            key = 'TFM, processed FMC (64 TX)' if corrected else 'RTM illumination normalized (64 TX)'
            centers = [[p['peak_x_mm'], p['peak_z_mm']] for p in diagnostic[key]['matched_peaks']]
            base.update(target_type='multiple indications', cnr_roi_mm=[-5, 25, 47, 72],
                        signal={'kind': 'circles', 'centers_mm': centers, 'radius_mm': 2.},
                        noise={'kind': 'rectangles',
                               'bounds_mm': [[-4, 3.5, 49, 54.5], [11.5, 19.5, 63, 68]]})
            base['notes'].append('CNR pools the indication disks. The close pair overlaps; the disks do not establish independent defect resolution.')
            base['notes'].append('Circle centers follow TFM indications after timing correction, or the previously measured RTM peaks before correction.')
        if int(case_id) == 13:
            base['notes'].append('Only TX32 was migrated in this case; the other real cases use full FMC.')
    else:
        source = np.load(PROJECT / 'assets/sources/source.npy')
        nfft = max(65536, 2 ** int(np.ceil(np.log2(len(source)))))
        frequency = np.fft.rfftfreq(nfft, 3e-7)[np.argmax(np.abs(np.fft.rfft(source, nfft)))]
        base.update(sound_speed_m_s=1500., frequency_hz=float(frequency),
                    api_metadata_status='inferred_legacy_metadata', target_type='extended interfaces and reflectors',
                    cnr_roi_mm=[float(x[0]), float(x[-1]), 150, 800],
                    signal={'kind': 'distance_to_mask', 'mask_path': 'assets/models/poynting_benchmark.png',
                            'max_distance_mm': 3.},
                    noise={'kind': 'rectangles', 'bounds_mm': [[360, 435, 180, 255]]})
        base['notes'] += [
            'No saved run configuration: dx=dz=1.5 mm and source dt=0.3 us are inferred from the current legacy workflow.',
            'Frequency is the spectral peak of the current assets/sources/source.npy. API is provisional until historical source/grid metadata is confirmed.',
            'Wavelength uses the 1500 m/s incident medium; the model also contains 3200 m/s material.',
            'Coordinates start at the cropped image origin; model masks are cropped by 45 cells.',
            'Cases 1 and 2 contain identical final Standard and Poynting arrays.',
        ]
    base['api_roi_mm'] = list(base['cnr_roi_mm'])
    return base


def region_mask(spec, x, z):
    xx, zz = np.meshgrid(x, z)
    kind = spec['kind']
    mask = np.zeros(xx.shape, bool)
    if kind in ('circles', 'outside_circles'):
        for cx, cz in spec['centers_mm']:
            mask |= (xx - cx)**2 + (zz - cz)**2 <= spec['radius_mm']**2
        return ~mask if kind == 'outside_circles' else mask
    if kind == 'rectangles':
        for xmin, xmax, zmin, zmax in spec['bounds_mm']:
            mask |= (xx >= xmin) & (xx <= xmax) & (zz >= zmin) & (zz <= zmax)
        return mask
    if kind == 'arc_band':
        cx, cz = spec['center_mm']
        return abs(np.hypot(xx - cx, zz - cz) - spec['radius_mm']) <= spec['half_width_mm']
    if kind == 'distance_to_mask':
        path = PROJECT / spec['mask_path']
        if path.suffix == '.npy':
            target = np.load(path).astype(bool)
        else:
            rgb = np.asarray(Image.open(path).convert('RGB'))[45:-45, 45:-45]
            material = np.all(rgb == (0, 255, 0), axis=-1)
            target = (material & ~binary_erosion(material)) | np.all(rgb == (0, 0, 0), axis=-1)
        if target.shape != xx.shape or not target.any():
            raise ValueError(f'Invalid target mask: {path}, shape={target.shape}, expected={xx.shape}')
        distance = distance_transform_edt(~target, sampling=(float(np.diff(z)[0]), float(np.diff(x)[0])))
        if 'max_distance_mm' in spec:
            return distance <= spec['max_distance_mm']
        return distance >= spec['min_distance_mm']
    raise ValueError(f'Unknown region kind: {kind}')


def cnr_masks(region, x, z):
    roi = region_mask({'kind': 'rectangles', 'bounds_mm': [region['cnr_roi_mm']]}, x, z)
    signal = region_mask(region['signal'], x, z) & roi
    noise = region_mask(region['noise'], x, z) & roi
    if not signal.any() or not noise.any() or np.any(signal & noise):
        raise ValueError('Signal/noise regions must be nonempty and disjoint.')
    return signal, noise


def api_measure(amplitude, dx_mm, dz_mm, wavelength_mm, domain=None):
    """Equation 27, with the peak and every counted pixel in the same domain."""
    if domain is not None and (domain.shape != amplitude.shape or not domain.any()):
        raise ValueError('The API domain must be nonempty and have the image shape.')
    peak = float(amplitude.max() if domain is None else amplitude[domain].max())
    if peak == 0:
        raise ValueError('A zero image has no peak-relative -6 dB area.')
    mask = amplitude >= peak * THRESHOLD
    if domain is not None:
        mask &= domain
    area = float(mask.sum() * dx_mm * dz_mm)
    return {'a_minus_6db_pixels': int(mask.sum()), 'a_minus_6db_mm2': area,
            'api': area / wavelength_mm**2, 'image_peak_amplitude': peak,
            'threshold_amplitude': peak * THRESHOLD}, mask


def measure(amplitude, signal, noise, dx_mm, dz_mm, wavelength_mm):
    """Equations 23–27 on power samples; return the full-image API reference."""
    amplitude = np.asarray(amplitude, dtype=np.float64)
    if amplitude.ndim != 2 or not np.isfinite(amplitude).all() or np.any(amplitude < 0):
        raise ValueError('Expected a finite nonnegative two-dimensional amplitude image.')
    if signal.shape != amplitude.shape or noise.shape != amplitude.shape:
        raise ValueError('Region masks must have the image shape.')
    if not signal.any() or not noise.any() or np.any(signal & noise):
        raise ValueError('Signal/noise regions must be nonempty and disjoint.')
    if min(dx_mm, dz_mm, wavelength_mm) <= 0:
        raise ValueError('Pixel spacing and wavelength must be positive.')
    api_metrics, api_mask = api_measure(amplitude, dx_mm, dz_mm, wavelength_mm)
    signal_power = amplitude[signal]**2
    noise_power = amplitude[noise]**2
    ui, uo = float(signal_power.mean()), float(noise_power.mean())
    sigma = float(noise_power.std(ddof=0))
    contrast = ui / uo if uo else None
    cnr = abs(ui - uo) / sigma if sigma else None
    notes = []
    if contrast is None:
        notes.append('Contrast undefined: background mean power is zero.')
    elif contrast == 0:
        notes.append('Contrast dB is -infinity: signal mean power is zero; stored as null.')
    if cnr is None:
        notes.append('CNR undefined: background power standard deviation is zero.')
    elif cnr == 0:
        notes.append('CNR dB is -infinity: signal and background mean powers are equal; stored as null.')
    return {
        'signal_mean_power': ui, 'noise_mean_power': uo, 'noise_std_power': sigma,
        'contrast': contrast, 'contrast_db': float(10 * np.log10(contrast)) if contrast and contrast > 0 else None,
        'cnr': cnr, 'cnr_db': float(20 * np.log10(cnr)) if cnr and cnr > 0 else None,
        **api_metrics,
        'wavelength_mm': wavelength_mm, 'dx_mm': dx_mm, 'dz_mm': dz_mm,
        'signal_pixels': int(signal.sum()), 'noise_pixels': int(noise.sum()),
        'metric_notes': notes,
    }, api_mask


def load_image(path, folder):
    field = np.load(path)
    if field.ndim != 2 or not np.isfinite(field).all():
        raise ValueError(f'Invalid final reconstruction: {path}')
    x, z, crop = grid(folder, field.shape)
    amplitude = np.abs(hilbert(field[crop].astype(np.float64), axis=0))
    return amplitude, x, z, crop


def display_limits(regions):
    """Share ROI-based signed-amplitude limits across methods and boundary variants."""
    groups, samples = {}, {}
    for case_id, source_folder in cases().items():
        folder = SIMULATIONS / source_folder
        group = folder.parent if folder.name in BOUNDARIES else folder
        groups[case_id] = group
        for path in result_paths(folder).values():
            field = np.load(path)
            x, z, crop = grid(folder, field.shape)
            region = regions.get(case_id) or defaults(case_id, folder, x, z)
            domain = region_mask({'kind': 'rectangles', 'bounds_mm': [region['cnr_roi_mm']]}, x, z)
            samples.setdefault(group, []).append(np.abs(field[crop][domain]))
    limits = {group: float(np.percentile(np.concatenate(values), 99.5)) or 1.
              for group, values in samples.items()}
    return {case_id: limits[group] for case_id, group in groups.items()}


def plot_panel(ax, field, x, z, api_mask, signal, noise, bounds=None, limit=None):
    if bounds is None:
        iz, ix = np.arange(len(z)), np.arange(len(x))
    else:
        xmin, xmax, zmin, zmax = bounds
        ix = np.flatnonzero((x >= xmin) & (x <= xmax))
        iz = np.flatnonzero((z >= zmin) & (z <= zmax))
    selected = np.ix_(iz, ix)
    dx, dz = float(np.diff(x)[0]), float(np.diff(z)[0])
    xs, zs = x[ix], z[iz]
    extent = (xs[0] - dx / 2, xs[-1] + dx / 2, zs[-1] + dz / 2, zs[0] - dz / 2)
    if limit is None:
        limit = float(np.percentile(np.abs(field[selected]), 99.5)) or 1.
    im = ax.imshow(field[selected], extent=extent, origin='upper', cmap='seismic', vmin=-limit, vmax=limit,
                   interpolation='none', aspect='equal')
    for name, mask, alpha in [('noise', noise, .12), ('signal', signal, .25), ('api', api_mask, .65)]:
        shown = mask[selected]
        if bounds is not None or name == 'api':
            ax.imshow(np.ma.masked_where(~shown, shown), extent=extent, origin='upper',
                      cmap=ListedColormap([COLORS[name]]), vmin=0, vmax=1,
                      alpha=alpha, interpolation='none', aspect='equal')
        if name != 'api' and shown.any() and not shown.all():
            ax.contour(xs, zs, shown, levels=[.5], colors=[COLORS[name]], linewidths=.7)
    if bounds is None:
        return im
    ax.set_xlim(bounds[:2])
    ax.set_ylim(bounds[3], bounds[2])
    return im


def legend(fig):
    fig.legend(handles=[Patch(facecolor=COLORS['api'], label='Reflector-window A−6 dB'),
                        Patch(facecolor=COLORS['signal'], label='Proposed signal'),
                        Patch(facecolor=COLORS['noise'], label='Proposed noise')],
               loc='lower center', ncols=3, frameon=False, fontsize=9)


def format_value(value):
    return 'undefined' if value is None else f'{value:.4g}'


def render_image(path, case_id, method, field, x, z, masks, region, metrics, limit):
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.1))
    fig.subplots_adjust(left=.07, right=.90, bottom=.27, top=.81, wspace=.25)
    for ax, bounds, title in zip(axes, (None, region['cnr_roi_mm']),
                                  ('Full image: shared amplitude scale', 'Reflector window: shared amplitude scale')):
        im = plot_panel(ax, field, x, z, *masks, bounds=bounds, limit=limit)
        ax.set(title=title, xlabel='x (mm)', ylabel='z (mm)')
    xmin, xmax, zmin, zmax = region['cnr_roi_mm']
    axes[0].add_patch(Rectangle((xmin, zmin), xmax-xmin, zmax-zmin, fill=False,
                                edgecolor='#ffc747', linewidth=1., linestyle='--'))
    axes[0].plot(metrics['full_image_peak_x_mm'], metrics['full_image_peak_z_mm'], '+', color='white', markersize=9)
    cax = fig.add_axes([.93, .30, .015, .45])
    fig.colorbar(im, cax=cax, label='Illumination-normalized signed amplitude')
    fig.suptitle(f'Case {case_id:02d} — {method.capitalize()} RTM\n{region["source_folder"]}', fontsize=11)
    fig.text(.5, .165,
             f'ROI A−6 dB = {metrics["a_minus_6db_mm2"]:.4g} mm²   |   API = {metrics["api"]:.4g}   |   '
             f'λ = {metrics["wavelength_mm"]:.4g} mm\n'
             f'Contrast = {format_value(metrics["contrast_db"])} dB   |   CNR = {format_value(metrics["cnr_db"])} dB'
             f'   |   Regions: {region["review_status"]}', ha='center', fontsize=10)
    fig.text(.5, .108,
             f'Full-image reference: A−6 dB = {metrics["full_image_a_minus_6db_mm2"]:.4g} mm²; '
             f'API = {metrics["full_image_api"]:.4g}. Magenta: envelope-based reflector-window A−6 dB.',
             ha='center', fontsize=9)
    if region['api_metadata_status'] == 'inferred_legacy_metadata':
        fig.text(.5, .068, 'API provisional: legacy wavelength inferred from current workflow/source.',
                 ha='center', fontsize=9, color='#a24213')
    legend(fig)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def process(case_id, folder, output, regions, limit):
    paths = result_paths(folder)
    destination = output / f'case_{int(case_id):02d}'
    destination.mkdir(parents=True, exist_ok=True)
    results = {}
    saved_masks = {}
    expected_x = expected_z = None
    for method, path in paths.items():
        amplitude, x, z, crop = load_image(path, folder)
        field = np.load(path)[crop]
        if case_id not in regions:
            regions[case_id] = defaults(case_id, folder, x, z)
        region = regions[case_id]
        region.setdefault('api_roi_mm', list(region['cnr_roi_mm']))
        if expected_x is not None:
            np.testing.assert_array_equal(x, expected_x)
            np.testing.assert_array_equal(z, expected_z)
        expected_x, expected_z = x, z
        signal, noise = cnr_masks(region, x, z)
        wavelength = region['sound_speed_m_s'] / region['frequency_hz'] * 1000
        dx, dz = float(np.diff(x)[0]), float(np.diff(z)[0])
        metrics, full_mask = measure(amplitude, signal, noise, dx, dz, wavelength)
        full_peak_z, full_peak_x = np.unravel_index(np.argmax(amplitude), amplitude.shape)
        domain = region_mask({'kind': 'rectangles', 'bounds_mm': [region['api_roi_mm']]}, x, z)
        api_metrics, api_mask = api_measure(amplitude, dx, dz, wavelength, domain)
        for key in api_metrics:
            metrics[f'full_image_{key}'] = metrics[key]
        metrics.update(api_metrics)
        peak_z, peak_x = np.unravel_index(np.argmax(np.where(domain, amplitude, -1)), amplitude.shape)
        metrics.update(case_id=int(case_id), method=method, source_array=str(path.relative_to(PROJECT)),
                       source_folder=str(folder.relative_to(SIMULATIONS)),
                       png=str((destination / f'{method}.png').relative_to(output)),
                       peak_x_mm=float(x[peak_x]), peak_z_mm=float(z[peak_z]),
                       peak_inside_signal=bool(signal[peak_z, peak_x]),
                       full_image_peak_x_mm=float(x[full_peak_x]), full_image_peak_z_mm=float(z[full_peak_z]),
                       full_image_peak_inside_signal=bool(signal[full_peak_z, full_peak_x]),
                       api_roi_mm=region['api_roi_mm'], api_domain='reflector_window',
                       sound_speed_m_s=region['sound_speed_m_s'], frequency_hz=region['frequency_hz'],
                       review_status=region['review_status'], api_metadata_status=region['api_metadata_status'],
                       source_grid_crop=[[s.start, s.stop] for s in crop], image_shape=list(amplitude.shape))
        render_image(destination / f'{method}.png', int(case_id), method, field, x, z,
                     (api_mask, signal, noise), region, metrics, limit)
        results[method] = metrics
        saved_masks[f'{method}_a_minus_6db'] = api_mask
        saved_masks[f'{method}_full_image_a_minus_6db'] = full_mask
    np.savez_compressed(destination / 'masks.npz', x_mm=expected_x, z_mm=expected_z,
                        signal=signal, noise=noise, **saved_masks)
    write_json(destination / 'metrics.json', results)
    print(f'Case {int(case_id):02d}: {", ".join(paths)} saved', flush=True)


def consolidate(output, regions):
    rows = []
    for path in sorted(output.glob('case_*/metrics.json')):
        rows += list(json.loads(path.read_text()).values())
    fields = [key for key in rows[0] if key not in ('metric_notes', 'source_grid_crop', 'image_shape', 'api_roi_mm')]
    with (output / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'metrics.json', {'conventions': CONVENTIONS, 'results': rows})
    with (output / 'cases.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['case_id', 'source_folder', 'methods', 'review_status', 'notes'])
        for case_id, folder in cases().items():
            available = [row['method'] for row in rows if row['case_id'] == int(case_id)]
            if available:
                writer.writerow([case_id, folder, '/'.join(available), regions[case_id]['review_status'],
                                 ' '.join(regions[case_id]['notes'])])
    return rows


def render_reviews(output, regions, rows, limits):
    reviews = output / 'review'
    reviews.mkdir(exist_ok=True)
    groups = [(1, 2), (3, 7), (8, 13), (14, 17), (18, 21), (22, 25), (26, 29), (30, 33), (34, 37)]
    lookup = {(r['case_id'], r['method']): r for r in rows}
    for start, stop in groups:
        ids = [i for i in range(start, stop+1) if (i, 'standard') in lookup]
        if not ids:
            continue
        methods = ('standard',) if (start, stop) == (3, 7) else ('standard', 'poynting')
        fig, axes = plt.subplots(len(ids), len(methods), figsize=(6.3*len(methods), 5.2*len(ids)), squeeze=False)
        fig.subplots_adjust(left=.09, right=.98, bottom=.05,
                            top=.88 if len(ids) <= 2 else .93, hspace=.52, wspace=.25)
        for row, case_id in enumerate(ids):
            folder = SIMULATIONS / cases()[str(case_id)]
            region = regions[str(case_id)]
            for col, method in enumerate(methods):
                if (case_id, method) not in lookup:
                    axes[row, col].axis('off')
                    continue
                metrics = lookup[case_id, method]
                path = result_paths(folder)[method]
                amplitude, x, z, crop = load_image(path, folder)
                field = np.load(path)[crop]
                signal, noise = cnr_masks(region, x, z)
                domain = region_mask({'kind': 'rectangles', 'bounds_mm': [region['api_roi_mm']]}, x, z)
                api_mask = (amplitude >= metrics['threshold_amplitude']) & domain
                ax = axes[row, col]
                plot_panel(ax, field, x, z, api_mask, signal, noise, bounds=region['cnr_roi_mm'],
                           limit=limits[str(case_id)])
                boundary_titles = {'reflect_sides_bottom': 'Reflect sides + bottom',
                                   'reflect_bottom': 'Reflect bottom', 'reflect_sides': 'Reflect sides',
                                   'absorb_all': 'Absorb all'}
                label = boundary_titles.get(folder.name, folder.name.replace('_', ' '))
                if 'td_rtm_boundary_comparison_20260918' in folder.parts:
                    label = f'{folder.parent.name.upper().replace("_", "-")} / {label}'
                if 'immersion_fmc_rot_Xdeg.m2k' in folder.parts:
                    label = 'Immersion'
                elif 'meia_lua_fmc.m2k' in folder.parts:
                    label = 'Contact'
                ax.set_title(f'Case {case_id:02d} — {method.capitalize()} RTM\n{label}\n'
                             f'CNR {format_value(metrics["cnr_db"])} dB | API {metrics["api"]:.4g} | {region["review_status"]}',
                             fontsize=10)
                ax.set(xlabel='x (mm)', ylabel='z (mm)')
        fig.suptitle(f'Region review (linear seismic): cases {start:02d}–{stop:02d}\n'
                     'Green: signal | Cyan: noise | Magenta: envelope A−6 dB', fontsize=13)
        legend(fig)
        fig.savefig(reviews / f'cases_{start:02d}_{stop:02d}.png', dpi=130)
        plt.close(fig)
        print(f'Review cases {start:02d}–{stop:02d} saved', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, nargs='+', choices=range(1, 38), default=list(range(1, 38)))
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--reviews-only', action='store_true', help='Regenerate the review sheets from existing measurements.')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    region_file = output / 'regions.json'
    regions = json.loads(region_file.read_text())['cases'] if region_file.exists() else {}
    limits = display_limits(regions)
    if not args.reviews_only:
        for case_id in dict.fromkeys(args.cases):
            key = str(case_id)
            process(key, SIMULATIONS / cases()[key], output, regions, limits[key])
            write_json(region_file, {'conventions': CONVENTIONS, 'cases': regions})
    rows = consolidate(output, regions)
    render_reviews(output, regions, rows, limits)
    print(f'{len(rows)} final results across {len(regions)} cases: {output}', flush=True)


if __name__ == '__main__':
    main()
