"""
The Periodic Table of Language — Visualization
10 elements across 28 dimensions, 5 feature groups.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Feature dimension names ──
DIM_NAMES = {
    0: 'animate', 1: 'inanimate', 2: 'edible', 3: 'physical',
    4: 'needs_subj', 5: 'needs_obj', 6: 'can_mod_verb', 7: 'can_mod_noun',
    8: 'finiteness',
    9: 'scope_barrier', 10: 'givenness', 11: 'prep_head_pref',
    12: 'directionality', 13: 'arg_reach', 14: 'bond_absorber',
    15: 'structural_role',
    16: 'NP_orbital', 17: 'VP_orbital', 18: 'HEAD_orbital', 19: 'ARG_orbital',
    20: 'query_det', 21: 'query_adv', 22: 'query_2', 23: 'query_3',
    24: 'sel_pref_0', 25: 'sel_pref_1', 26: 'sel_pref_2', 27: 'sel_pref_3',
}

# ── Feature groups with colors ──
GROUPS = {
    'Semantic\nBase': {'dims': [0,1,2,3,4,5,6,7,8], 'color': '#4ECDC4'},
    'Structural\nProperties': {'dims': [9,10,11,12,13,14,15], 'color': '#FF6B6B'},
    'Bonding\nOrbitals': {'dims': [16,17,18,19], 'color': '#45B7D1'},
    'Query\nSignals': {'dims': [20,21,22,23], 'color': '#FFA07A'},
    'Selectional\nPreference': {'dims': [24,25,26,27], 'color': '#98D8C8'},
}

# ── POS element data ──
ELEMENTS = {
    'Noun': {
        'symbol': 'N', 'number': 1,
        'role': 'Filler — encodes what it IS',
        'key_props': 'can_mod_noun, structural_role=1.0, NP=1.5, HEAD=1.5',
        'color': '#2ECC71',
        'features': {7: 1.0, 12: 0.0, 13: 0.0, 15: 1.0, 16: 1.5, 18: 1.5},
    },
    'Verb': {
        'symbol': 'V', 'number': 2,
        'role': 'Seeker — encodes what it NEEDS',
        'key_props': 'dir=-0.3, VP=0.5, ARG=1.0, HEAD=0.5',
        'color': '#E74C3C',
        'features': {12: -0.3, 13: 0.2, 15: 0.5, 17: 0.5, 18: 0.5, 19: 1.0},
    },
    'Det': {
        'symbol': 'D', 'number': 3,
        'role': 'Query — seeks noun to specify',
        'key_props': 'needs_obj=1.0, dir=1.0, query=1.0, NP=0.5',
        'color': '#F39C12',
        'features': {5: 1.0, 12: 1.0, 13: 0.25, 15: 0.1, 16: 0.5, 20: 1.0},
    },
    'Adj': {
        'symbol': 'A', 'number': 4,
        'role': 'Modifier — bonds to noun head',
        'key_props': 'can_mod_noun=1.0, dir=0.8, query=1.0, NP=0.5',
        'color': '#9B59B6',
        'features': {7: 1.0, 12: 0.8, 13: 0.15, 15: 0.3, 16: 0.5, 20: 1.0},
    },
    'Adv': {
        'symbol': 'Av', 'number': 5,
        'role': 'Modifier — bonds to verb head',
        'key_props': 'can_mod_verb=1.0, VP=0.5, query=1.0',
        'color': '#1ABC9C',
        'features': {6: 1.0, 12: 0.0, 13: 0.2, 15: 0.3, 17: 0.5, 21: 1.0},
    },
    'Prep': {
        'symbol': 'P', 'number': 6,
        'role': 'Bridge — absorbs and redirects bonds',
        'key_props': 'absorber=0.85, NP=0.8, HEAD+ARG=0.5',
        'color': '#3498DB',
        'features': {12: 0.5, 13: 0.3, 14: 0.85, 15: 0.1, 16: 0.8, 18: 0.5, 19: 0.5},
    },
    'Pron': {
        'symbol': 'Pn', 'number': 7,
        'role': 'Filler — lightweight noun substitute',
        'key_props': 'animate=1.0, physical=1.0, NP=0.5, HEAD=1.5',
        'color': '#27AE60',
        'features': {0: 1.0, 3: 1.0, 15: 0.9, 16: 0.5, 18: 1.5},
    },
    'Aux': {
        'symbol': 'Ax', 'number': 8,
        'role': 'Bridge — absorbs verb, redirects arguments',
        'key_props': 'absorber=0.7, dir=0.8, VP=0.5, ARG=0.5',
        'color': '#E67E22',
        'features': {12: 0.8, 13: 0.15, 14: 0.7, 15: 0.1, 17: 0.5, 19: 0.5, 21: 1.0},
    },
    'Coord': {
        'symbol': 'Cc', 'number': 9,
        'role': 'Connector — links parallel structures',
        'key_props': 'structural_role=0.0 (no inherent direction)',
        'color': '#95A5A6',
        'features': {15: 0.0},
    },
    'Num': {
        'symbol': 'Nu', 'number': 10,
        'role': 'Modifier — numeric specification',
        'key_props': 'can_mod_noun=1.0, dir=0.8, NP=0.5',
        'color': '#8E44AD',
        'features': {7: 1.0, 12: 0.8, 13: 0.15, 15: 0.3, 16: 0.5},
    },
}


def draw_element_card(ax, x, y, elem_name, elem_data, w=2.5, h=3.0):
    """Draw a single element card like in the real periodic table."""
    color = elem_data['color']

    # Card background with subtle gradient effect via layered patches
    card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                           facecolor=color, edgecolor='white', linewidth=2.5, alpha=0.88)
    ax.add_patch(card)
    # Inner highlight
    highlight = FancyBboxPatch((x + 0.06, y + h*0.45), w - 0.12, h*0.52,
                                boxstyle="round,pad=0.04",
                                facecolor='white', edgecolor='none', alpha=0.07)
    ax.add_patch(highlight)

    # Atomic number
    ax.text(x + 0.18, y + h - 0.30, str(elem_data['number']),
            fontsize=11, color='white', fontweight='bold', alpha=0.85)

    # Symbol (large)
    ax.text(x + w/2, y + h*0.55, elem_data['symbol'],
            fontsize=42, fontweight='bold', color='white',
            ha='center', va='center')

    # Name
    ax.text(x + w/2, y + h*0.22, elem_name,
            fontsize=14, color='white', ha='center', va='center', fontweight='bold')

    # Role (small but readable)
    role_short = elem_data['role'].split('—')[0].strip()
    ax.text(x + w/2, y + h*0.08, role_short,
            fontsize=9, color='white', ha='center', va='center', alpha=0.9)


def draw_feature_heatmap(ax, elements):
    """Draw the feature heatmap showing all 28 dims for each POS."""
    pos_names = list(elements.keys())
    n_pos = len(pos_names)
    n_dims = 28

    # Build matrix
    matrix = np.zeros((n_pos, n_dims))
    for i, name in enumerate(pos_names):
        for dim, val in elements[name]['features'].items():
            matrix[i, dim] = val

    # Custom dark-friendly colormap: dark background to bright orange/yellow
    from matplotlib.colors import LinearSegmentedColormap
    dark_cmap = LinearSegmentedColormap.from_list('dark_heat',
        ['#1a1a2e', '#2d1b3d', '#5c2d6e', '#c0392b', '#e67e22', '#f1c40f', '#ffffcc'], N=256)

    im = ax.imshow(matrix, cmap=dark_cmap, aspect='auto', vmin=0, vmax=1.5)

    ax.set_yticks(range(n_pos))
    ax.set_yticklabels(pos_names, fontsize=11, fontweight='bold', color='white')
    ax.set_xticks(range(n_dims))
    ax.set_xticklabels([DIM_NAMES.get(i, str(i)) for i in range(n_dims)],
                       fontsize=8, rotation=45, ha='right', color='#ccc')

    # Group brackets
    group_positions = {
        'Semantic Base': (0, 8),
        'Structural': (9, 15),
        'Orbitals': (16, 19),
        'Query': (20, 23),
        'Sel. Pref': (24, 27),
    }
    group_colors_list = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#FFA07A', '#98D8C8']

    for idx, (gname, (start, end)) in enumerate(group_positions.items()):
        ax.axvline(x=start - 0.5, color=group_colors_list[idx], linewidth=2, alpha=0.6)
        ax.text((start + end) / 2, -1.8, gname, fontsize=9, ha='center',
                color=group_colors_list[idx], fontweight='bold')

    # Add values in cells
    for i in range(n_pos):
        for j in range(n_dims):
            val = matrix[i, j]
            if val > 0.01:
                text_color = 'white' if val > 0.5 else '#aaa'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                       fontsize=7, color=text_color, fontweight='bold')

    ax.set_title('Feature Values by POS Category (28 dimensions)',
                 fontsize=13, pad=20, color='white', fontweight='bold')
    return im


def draw_prep_spectrum(ax):
    """Draw preposition head preference spectrum."""
    preps = {
        'of': 0.03, 'about': 0.07, 'between': 0.3, 'on': 0.44, 'for': 0.47,
        'with': 0.5, 'under': 0.5, 'as': 0.53, 'from': 0.55, 'through': 0.6,
        'in': 0.61, 'at': 0.68, 'by': 0.69, 'after': 0.7, 'before': 0.7,
        'into': 0.8, 'since': 0.8, 'to': 0.88,
    }
    names = list(preps.keys())
    vals = list(preps.values())

    # Gradient from warm red (noun-attach) to cool blue (verb-attach)
    from matplotlib.colors import LinearSegmentedColormap
    spec_cmap = LinearSegmentedColormap.from_list('prep_spec',
        ['#E74C3C', '#F39C12', '#45B7D1', '#3498DB'], N=256)
    colors = [spec_cmap(v) for v in vals]
    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor='#2a2a4a', linewidth=0.8, height=0.75)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9, color='#ddd', family='monospace')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Verb \u2190 preference \u2192 Noun', fontsize=9, color='#aaa')
    ax.set_title('Prep Head Preference\n(dim 11)', fontsize=12, color='white', fontweight='bold')
    ax.axvline(x=0.5, color='#555', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(0.05, -1.2, '\u2190 noun-attach', fontsize=9, color='#E74C3C', fontweight='bold')
    ax.text(0.72, -1.2, 'verb-attach \u2192', fontsize=9, color='#3498DB', fontweight='bold')
    ax.invert_yaxis()


def draw_scope_barriers(ax):
    """Draw scope barrier values."""
    barriers = {
        '.': 1.0, ',': 0.9, ';': 0.9, ':': 0.8,
        'because': 0.8, 'although': 0.8, 'if': 0.8, 'unless': 0.8,
        'that': 0.7, 'which': 0.7, 'who': 0.7, 'while': 0.7,
        'before': 0.6, 'after': 0.6,
        'but': 0.4, 'and': 0.2, 'or': 0.2,
    }
    names = list(barriers.keys())
    vals = list(barriers.values())

    # Gradient from soft to intense red
    from matplotlib.colors import LinearSegmentedColormap
    barrier_cmap = LinearSegmentedColormap.from_list('barriers',
        ['#4a2040', '#8e2c5e', '#c0392b', '#e74c3c', '#ff6b6b'], N=256)
    colors = [barrier_cmap(v) for v in vals]
    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor='#2a2a4a', linewidth=0.8, height=0.75)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9, family='monospace', color='#ddd')
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Barrier strength', fontsize=9, color='#aaa')
    ax.set_title('Scope Barriers\n(dim 9)', fontsize=12, color='white', fontweight='bold')
    ax.invert_yaxis()

    # Value labels on bars
    for i, (bar, val) in enumerate(zip(bars, vals)):
        ax.text(val + 0.03, i, f'{val:.1f}', va='center', fontsize=8, color='#ccc')


def main():
    fig = plt.figure(figsize=(26, 22), facecolor='#1a1a2e')

    # Title
    fig.text(0.5, 0.975, 'THE PERIODIC TABLE OF LANGUAGE',
             fontsize=36, fontweight='bold', color='white', ha='center',
             family='monospace')
    fig.text(0.5, 0.955, 'Atomic Semantic Architecture \u2014 10 Elements, 28 Dimensions, 5 Feature Groups',
             fontsize=15, color='#999', ha='center')

    # ── Section 1: Element Cards (periodic-table style) ──
    ax_cards = fig.add_axes([0.02, 0.64, 0.96, 0.28])
    ax_cards.set_facecolor('#1a1a2e')
    ax_cards.set_xlim(-1.5, 26)
    ax_cards.set_ylim(-0.5, 8.5)
    ax_cards.axis('off')

    # Layout: Row 1 = content words, Row 2 = function words
    layout = {
        # Row 1: Content words
        'Noun': (0, 4.5),
        'Verb': (2.9, 4.5),
        'Adj':  (5.8, 4.5),
        'Adv':  (8.7, 4.5),
        'Num':  (11.6, 4.5),
        # Row 2: Function words
        'Det':   (0, 1.0),
        'Pron':  (2.9, 1.0),
        'Prep':  (5.8, 1.0),
        'Aux':   (8.7, 1.0),
        'Coord': (11.6, 1.0),
    }

    # Row labels
    ax_cards.text(-0.3, 6.0, 'Content\nWords', fontsize=11, color='#888',
                  ha='right', va='center', fontweight='bold')
    ax_cards.text(-0.3, 2.5, 'Function\nWords', fontsize=11, color='#888',
                  ha='right', va='center', fontweight='bold')

    for name, (x, y) in layout.items():
        draw_element_card(ax_cards, x, y, name, ELEMENTS[name])

    # ── Legend: Seeker/Filler explanation ──
    legend_x = 15.5
    legend_y = 1.0

    ax_cards.text(legend_x, 7.5, 'BONDING ROLES', fontsize=16, color='white',
                  fontweight='bold')

    role_items = [
        ('#2ECC71', 'Filler', 'Encodes what the word IS (semantic properties)'),
        ('#E74C3C', 'Seeker', 'Encodes what the word NEEDS (argument slots)'),
        ('#F39C12', 'Query', 'Announces what it seeks (det \u2192 noun)'),
        ('#3498DB', 'Bridge', 'Absorbs bonds and redirects them'),
        ('#95A5A6', 'Connector', 'Links parallel structures'),
    ]

    for i, (color, role, desc) in enumerate(role_items):
        y = 6.3 - i * 1.1
        dot = FancyBboxPatch((legend_x, y - 0.15), 0.5, 0.5,
                              boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='none', alpha=0.8)
        ax_cards.add_patch(dot)
        ax_cards.text(legend_x + 0.8, y + 0.15, role, fontsize=13, color='white',
                      fontweight='bold')
        ax_cards.text(legend_x + 0.8, y - 0.35, desc, fontsize=10, color='#aaa')

    # Section divider
    divider1 = fig.add_axes([0.04, 0.635, 0.92, 0.001])
    divider1.set_facecolor('#1a1a2e')
    divider1.axhline(y=0.5, color='#333', linewidth=1.5)
    divider1.axis('off')

    # ── Section 2: Feature Heatmap ──
    ax_heat = fig.add_axes([0.06, 0.30, 0.55, 0.30])
    ax_heat.set_facecolor('#1a1a2e')
    im = draw_feature_heatmap(ax_heat, ELEMENTS)
    ax_heat.tick_params(colors='white')
    ax_heat.title.set_color('white')
    ax_heat.xaxis.label.set_color('white')
    for spine in ax_heat.spines.values():
        spine.set_color('#333')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(colors='white', labelsize=9)
    cbar.set_label('Feature value', color='white', fontsize=10)

    # ── Section 3: Prep spectrum ──
    ax_prep = fig.add_axes([0.70, 0.30, 0.27, 0.30])
    ax_prep.set_facecolor('#1a1a2e')
    draw_prep_spectrum(ax_prep)
    ax_prep.tick_params(colors='white')
    ax_prep.title.set_color('white')
    ax_prep.xaxis.label.set_color('white')
    for spine in ax_prep.spines.values():
        spine.set_color('#333')

    # ── Section 4: Element descriptions ──
    ax_desc = fig.add_axes([0.04, 0.02, 0.44, 0.24])
    ax_desc.set_facecolor('#1a1a2e')
    ax_desc.axis('off')

    ax_desc.text(0, 1.0, 'FEATURE GROUPS', fontsize=16,
                 color='white', fontweight='bold', transform=ax_desc.transAxes)

    elements_desc = [
        ('Semantic Base (dims 0\u20138)', 'What a word IS: animate, edible, physical, needs_subj/obj, modifier type, finiteness'),
        ('Scope Barrier (dim 9)', 'Bond-blocking strength of punctuation & subordinators'),
        ('Givenness (dim 10)', 'Information status: new vs given referent'),
        ('Prep Head Pref (dim 11)', 'Each preposition\'s intrinsic verb-vs-noun attachment preference'),
        ('Directionality (dim 12)', 'Left-bonding (+1) vs right-bonding (-1) tendency'),
        ('Argument Reach (dim 13)', 'How far bonds can extend from this word'),
        ('Bond Absorber (dim 14)', 'Preps & aux absorb bonds and redirect them'),
        ('Structural Role (dim 15)', 'Head (1.0) vs dependent (0.1) in phrases'),
        ('Bonding Orbitals (dims 16\u201319)', 'NP, VP, HEAD, ARG \u2014 constituency membership signals'),
        ('Query Signals (dims 20\u201323)', 'Determiners & adverbs announce what they seek'),
        ('Selectional Pref (dims 24\u201327)', 'SVD of verb-noun co-occurrence patterns'),
    ]

    for i, (name, desc) in enumerate(elements_desc):
        y = 0.92 - i * 0.082
        ax_desc.text(0.02, y, name, fontsize=10, color='#4ECDC4',
                     fontweight='bold', transform=ax_desc.transAxes)
        ax_desc.text(0.02, y - 0.038, desc, fontsize=8.5, color='#aaa',
                     transform=ax_desc.transAxes)

    # ── Section 5: Scope barriers ──
    ax_scope = fig.add_axes([0.52, 0.02, 0.20, 0.24])
    ax_scope.set_facecolor('#1a1a2e')
    draw_scope_barriers(ax_scope)
    ax_scope.tick_params(colors='white')
    ax_scope.title.set_color('white')
    ax_scope.xaxis.label.set_color('white')
    for spine in ax_scope.spines.values():
        spine.set_color('#333')

    plt.savefig('periodic_table.png', dpi=200, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    print("Saved periodic_table.png")


if __name__ == '__main__':
    main()
