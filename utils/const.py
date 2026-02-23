from typing import Iterable


def remove_trailing_chars(x: Iterable[str]) -> list:
    return list(map(lambda s: s.replace('__Y_', ' '), x))


def compr_map(x: Iterable[str]) -> list:
    f = lambda s: s.replace('compr', r'$F_{C}$')
    x = remove_trailing_chars(x)
    return list(map(f, x))


def shear_map(x: Iterable[str]) -> list:
    f = lambda s: s.replace('shear_AP', r'$F_{S,AP}$').replace('shear_ML', r'$F_{S,ML}$')
    x = remove_trailing_chars(x)
    return list(map(f, x))


def shearcompr_map(x: Iterable[str]) -> list:
    x = remove_trailing_chars(x)
    return compr_map(shear_map(x))


# Preliminaries
_COMPRESSION: list = [f'compr__Y_L{i}' for i in range(1, 6)]
_COMPRESSION.append('compr__Y_T12')
_COMPRESSION_D = dict(zip(_COMPRESSION, compr_map(_COMPRESSION)))

_SHEAR: list = [f'shear_{name}__Y_L{i}' for i in range(1, 6) for name in ('AP', 'ML')]
_SHEAR.extend([f'shear_{name}__Y_T12' for name in ('AP', 'ML')])
_SHEAR_D = dict(zip(_SHEAR, shear_map(_SHEAR)))

_SHEARCOMPR: list = _COMPRESSION.copy()
_SHEARCOMPR.extend(_SHEAR)
_SHEARCOMPR_D = dict(zip(_SHEARCOMPR, shearcompr_map(_SHEARCOMPR)))

_MUSCLES: list = ['F_Ribcage_musc_rect_abdom_l',
                  'F_Ribcage_musc_rect_abdom_r',
                  'G_lateral__G_external_oblique__G_left',
                  'G_lateral__G_external_oblique__G_right',
                  'G_lateral__G_iliocostalis_lumborum__G_left',
                  'G_lateral__G_iliocostalis_lumborum__G_right',
                  'G_lateral__G_internal_oblique__G_lefte',
                  'G_lateral__G_internal_oblique__G_right',
                  'G_lateral__G_longissimus_thoracis__G_left',
                  'G_lateral__G_longissimus_thoracis__G_right',
                  'G_lateral__G_psoas_major__G_left__F_Pelvis',
                  'G_lateral__G_psoas_major__G_right__F_Pelvis_L',
                  'G_lateral__G_quadratus_lumborum__G_left',
                  'G_lateral__G_quadratus_lumborum__G_left',
                  'G_lateral__G_quadratus_lumborum__G_right',
                  'G_lateral__G_quadratus_lumborum__G_right',
                  'G_medial__G_interspinales__F',
                  'G_medial__G_interspinales__F_T12',
                  'G_medial__G_multifidus__G_lumbar__G_left',
                  'G_medial__G_multifidus__G_lumbar__G_left',
                  'G_medial__G_multifidus__G_lumbar__G_right']
_PRELIMINARIES: list = [*_COMPRESSION, *_SHEAR, *_MUSCLES]

# NAKO
_COORDS = [f'coord_{i}' for i in range(6)]
_COORDS_D: dict = {'coord_0': r'$M_{x}$',
                   'coord_1': r'$M_{y}$',
                   'coord_2': r'$M_{z}$',
                   'coord_3': r'$F_{x}$',
                   'coord_4': r'$F_{y}$',
                   'coord_5': r'$F_{z}$'}
