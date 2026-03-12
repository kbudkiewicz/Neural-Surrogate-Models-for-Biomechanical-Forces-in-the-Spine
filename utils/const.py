import pandas as pd
from typing import Iterable, Callable
from itertools import cycle
from functools import wraps


def remove_trailing_chars(func: Callable) -> Callable:
    """Decorator that replaces '__Y_' with a space in the input iterable."""
    @wraps(func)
    def wrapper(x: Iterable[str], *args, **kwargs) -> list:
        cleaned_x = list(map(lambda s: s.replace('__Y_', ' '), x))
        return func(cleaned_x, *args, **kwargs)
    return wrapper


@remove_trailing_chars
def compr_map(x: Iterable[str]) -> list:
    f = lambda s: s.replace('compr', r'$F_{C}$')
    return list(map(f, x))


@remove_trailing_chars
def shear_map(x: Iterable[str]) -> list:
    f = lambda s: s.replace('shear_AP', r'$F_{S,AP}$').replace('shear_ML', r'$F_{S,ML}$')
    return list(map(f, x))


def shearcompr_map(x: Iterable[str]) -> list:
    return compr_map(shear_map(x))


@remove_trailing_chars
def muscles_map(x: Iterable[str], use_abbrev: bool = True) -> list:
    rem_g = lambda s: s.replace('__G_', ' ').replace('G_', '').replace('__F_', ' ').replace('F_', '')
    rem__ = lambda s: s.replace('_', ' ')
    rem_ribcage = lambda s: s.replace('Ribcage_musc', '')
    capitalize_first = lambda s: s.title()
    x = list(map(rem_g, x))
    x = list(map(rem_ribcage, x))
    x = list(map(capitalize_first, x))

    if use_abbrev:
        for key, value in _MUSCLES_ABBREV.items():
            x = [s.replace(key, value) for s in x]

    return list(map(rem__, x))


# PRELIMINARIES: Use compr__L, etc. for the newest models and evals. NAKO: shear__Y_L, etc.
_COMPRESSION: list = [f'compr__Y_L{i}' for i in range(1, 6)]
_COMPRESSION.append('compr__Y_T12')
_COMPRESSION_D = dict(zip(_COMPRESSION, compr_map(_COMPRESSION)))

_SHEAR_AP: list = [f'shear_AP__Y_L{i}' for i in range(1, 6)]
_SHEAR_ML: list = [item.replace('AP', 'ML') for item in _SHEAR_AP]
_SHEAR_AP.append(f'shear_AP__Y_T12')
_SHEAR_ML.append('shear_ML__Y_T12')
_SHEAR = [*_SHEAR_AP, *_SHEAR_ML]
_SHEAR_D = dict(zip(_SHEAR, shear_map(_SHEAR)))

_SHEARCOMPR: list = _COMPRESSION.copy()
_SHEARCOMPR.extend(_SHEAR)
_SHEARCOMPR_D = dict(zip(_SHEARCOMPR, shearcompr_map(_SHEARCOMPR)))

_MUSCLES: list = ['F_Ribcage_musc_rect_abdom_l',
                  'F_Ribcage_musc_rect_abdom_r',
                  'G_lateral__G_external_oblique__G_left__F_exte',
                  'G_lateral__G_external_oblique__G_right__F_ext',
                  'G_lateral__G_iliocostalis_lumborum__G_left__F',
                  'G_lateral__G_iliocostalis_lumborum__G_right',
                  'G_lateral__G_internal_oblique__G_left__F_inte',
                  'G_lateral__G_internal_oblique__G_right__F_int',
                  'G_lateral__G_longissimus_thoracis__G_left__F',
                  'G_lateral__G_longissimus_thoracis__G_right__F',
                  'G_lateral__G_psoas_major__G_left__F_Pelvis_L1',
                  'G_lateral__G_psoas_major__G_left__F_Pelvis_L2',
                  'G_lateral__G_psoas_major__G_left__F_Pelvis_L3',
                  'G_lateral__G_psoas_major__G_left__F_Pelvis_L4',
                  'G_lateral__G_psoas_major__G_left__F_Pelvis_L5',
                  'G_lateral__G_psoas_major__G_right__F_Pelvis_L',
                  'G_lateral__G_quadratus_lumborum__G_left__F_L2',
                  'G_lateral__G_quadratus_lumborum__G_left__F_L3',
                  'G_lateral__G_quadratus_lumborum__G_left__F_L4',
                  'G_lateral__G_quadratus_lumborum__G_left__F_Pe',
                  'G_lateral__G_quadratus_lumborum__G_right__F_L',
                  'G_lateral__G_quadratus_lumborum__G_right__F_P',
                  'G_medial__G_interspinales__F_L1_L2',
                  'G_medial__G_interspinales__F_L2_L3',
                  'G_medial__G_interspinales__F_L3_L4',
                  'G_medial__G_interspinales__F_L4_L5',
                  'G_medial__G_interspinales__F_T12_L1',
                  'G_medial__G_multifidus__G_lumbar__G_left__F_L',
                  'G_medial__G_multifidus__G_lumbar__G_left__F_S',
                  'G_medial__G_multifidus__G_lumbar__G_right__F']
_MUSCLES_ABBREV: dict = {
    'Rect_Abdom': 'RA',
    'Internal_Oblique': 'IO',
    'External_Oblique': 'EO',
    'Psoas_Major': 'PM',
    'Quadratus_Lumborum': 'QL',
    'Multifidus': 'MF',
    'Longissimus_Thoracis': 'LTL',
    'Iliocostalis_Lumborum': 'IL',
    'Interspinales': 'IS'
}
_MUSCLES_D = dict(zip(_MUSCLES, muscles_map(_MUSCLES, use_abbrev=True)))
_PRELIMINARIES_D = dict(**_COMPRESSION_D, **_SHEAR_D, **_MUSCLES_D)


# NAKO
def coord_to_math(x: str) -> str:
    vertebra, _, idx, type = x.split('_')
    axes = {str(n): axis for n, axis in zip(range(6), cycle(['x', 'y', 'z']))}
    if type == 'moment':
        type = 'M'
    elif type == 'force':
        type = 'F'
    else:
        raise ValueError('Unexpected input. Expected force or moment')
    axis = axes[idx]
    return f'${type}_{axis}$ {vertebra}'


df = pd.read_csv('../data/csv/nako_zeroed.csv')
NAKO_FORCES = sorted(df.columns[df.columns.str.contains('force')])
NAKO_MOMENTS = sorted(df.columns[df.columns.str.contains('moment')])
_FORCES_D = dict(zip(NAKO_FORCES, map(coord_to_math, NAKO_FORCES)))
_MOMENTS_D = dict(zip(NAKO_MOMENTS, map(coord_to_math, NAKO_MOMENTS)))
NAKO_FORCES.extend(NAKO_MOMENTS)
_COORDS_ALL_D = dict(zip(NAKO_FORCES, map(coord_to_math, NAKO_FORCES)))
_COORDS_D: dict = {'coord_3': r'$F_{x}$',
                   'coord_4': r'$F_{y}$',
                   'coord_5': r'$F_{z}$',
                   'coord_0': r'$M_{x}$',
                   'coord_1': r'$M_{y}$',
                   'coord_2': r'$M_{z}$'}
