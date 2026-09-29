"""
feature_labels.py
=================
Single source of human-readable names for the model features, used by the
SHAP figures (tabpfn_xai.py) and by Section 9 of docs/DATA_DICTIONARY.md
(generated from this module by ``python feature_labels.py --markdown``), so
that figure labels and the data dictionary cannot drift apart.

A feature column name is decomposed as

    <base>[_log][_std]                     household-level features
    <prefix>_<code>_<indicator>            code-derived features

and ``label(col)`` rebuilds a short display name from the parts, e.g.

    mean_DispInkFam04_log_std -> 'Family income, mean (log, z)'
    ov_H65-H75_exceeds_p75    -> 'OP H65-H75 middle ear: >P75'

'z' marks a standardised (z-scored) value; 'log' a log1p transform.
OP / IP = outpatient / inpatient ICD-10 diagnosis, Rx = dispensed ATC
prescription; any / >med / >P75 = household record count > 0 / > the
training-fold median / > the training-fold 75th percentile. Labels are kept
short (<= ~32 characters) so the plot area is not squeezed; full names are
in the descriptions (DATA_DICTIONARY.md Section 9).
"""
import re
import sys

# ---------------------------------------------------------------------------
# Household-level features: base name -> (short label, description)
# Descriptions follow DATA_DICTIONARY.md Section 3.
# ---------------------------------------------------------------------------
HOUSEHOLD = {
    'household_size':        ('Household size', 'Number of household members'),
    'index_cases_count':     ('No. of index cases', 'Number of index cases in the household'),
    'mean_age_2020':         ('Age, mean', 'Mean age of household members (2020)'),
    'max_age_2020':          ('Age, max', 'Age of the oldest member'),
    'min_age_2020':          ('Age, min', 'Age of the youngest member'),
    'age_variance':          ('Age, variance', 'Variance of member ages'),
    'age_IQR':               ('Age, IQR', 'Interquartile range of member ages'),
    'age_range':             ('Age, range', 'Oldest minus youngest member age'),
    'age_0_17_count':        ('Members aged 0-17', 'Number of members aged 0-17'),
    'age_18_64_count':       ('Members aged 18-64', 'Number of members aged 18-64'),
    'age_65plus_count':      ('Members aged 65+', 'Number of members aged 65 or older'),
    'has_member_75plus':     ('Any member 75+', 'Indicator: any member aged 75 or older'),
    'proportion_children':   ('Share of children', 'Share of members aged under 18'),
    'proportion_elderly':    ('Share aged 65+', 'Share of members aged 65 or older'),
    'prop_foreign_background':    ('Share foreign background', 'Share of members with foreign background'),
    'has_any_foreign_background': ('Any foreign background', 'Indicator: any member with foreign background'),
    'all_foreign_background':     ('All foreign background', 'Indicator: all members with foreign background'),
    'prop_born_sweden':      ('Share born in Sweden', 'Share of members born in Sweden'),
    'Fodelseland_mode_freq': ('Birth country (frequency)', 'Most common birth country in household, frequency-encoded'),
    'Fodelseland_diversity': ('Birth-country diversity', 'Number of distinct birth countries in household'),
    'male_count':            ('No. of males', 'Number of male members'),
    'female_count':          ('No. of females', 'Number of female members'),
    'proportion_male':       ('Share male', 'Share of male members'),
    'proportion_female':     ('Share female', 'Share of female members'),
    'gender_diversity':      ('Both sexes present', 'Indicator: household has both male and female members'),
    'has_child_under_6':     ('Child under 6', 'Indicator: any member aged under 6'),
    'has_child_6_17':        ('Child aged 6-17', 'Indicator: any member aged 6-17'),
    'has_elderly_65plus':    ('Any member 65+', 'Indicator: any member aged 65 or older'),
    'multigenerational':     ('Multigenerational', 'Indicator: oldest minus youngest member age > 40 years'),
    'three_generation':      ('Three generations', 'Indicator: members in all of <18, 18-64 and 65+'),
    'AntalBarnUnder18':      ('Children <18 (register)', 'Number of children under 18 registered to the household'),
    'Boarea_Person':         ('Living area per person', 'Living area per person (m2)'),
    'total_Boarea':          ('Total living area', 'Household total living area (area per person x household size)'),
    'crowding_index':        ('Crowding index', 'Household size / living area per person'),
    'is_overcrowded':        ('Overcrowded', 'Indicator: crowding index > 1.5'),
    'is_spacious':           ('Spacious', 'Indicator: crowding index < 0.5'),
    'TRYGG_1_sum':           ('Safety-alarm records', 'Sum of security-alarm (type 1) elderly-care records'),
    'TRYGG_total_sum':       ('Elderly-care records', 'Sum of all elderly-care service records'),
    'any_TRYGG_1':           ('Any safety-alarm record', 'Indicator: any security-alarm (type 1) record'),
    'any_TRYGG':             ('Any elderly-care record', 'Indicator: any elderly-care service record'),
    'proportion_with_TRYGG': ('Share with elderly care', 'Share of members with any elderly-care service record'),
    'TRYGG_1_per_elderly':   ('Safety-alarm records/65+', 'Security-alarm records per member aged 65+'),
    'TRYGG_total_per_capita': ('Elderly-care records/member', 'Elderly-care service records per household member'),
    'UtlSvBakg_mode_enc':    ('Background (mode)', 'Most common foreign/Swedish-background code, label-encoded'),
}

# Income statistics: <stat>_<DispInk04|DispInkFam04>
_INCOME = {'DispInk04': 'Individual disposable income',
           'DispInkFam04': 'Family disposable income'}
_INCOME_SHORT = {'DispInk04': 'Individual income',
                 'DispInkFam04': 'Family income'}
_STAT = {'mean': 'mean', 'median': 'median', 'max': 'max', 'min': 'min',
         'sd': 'SD', 'range': 'range'}
for _v, _vname in _INCOME.items():
    for _s, _sname in _STAT.items():
        HOUSEHOLD[f'{_s}_{_v}'] = (f'{_INCOME_SHORT[_v]}, {_sname}',
                                  f'{_sname} of members\' {_vname.lower()} (SEK)')

# Housing/tenure type (SCB 'Boendeform'), one-hot of the household's modal
# category. Category names follow SCB's standard Boendeform classification.
BOENDEFORM_SHORT = {
    'SmahusAg': 'house, owned', 'SmahusBo': 'house, tenant-owned',
    'SmahusHy': 'house, rented', 'FlerAg': 'flat, owned',
    'FlerBo': 'flat, tenant-owned', 'FlerHy': 'flat, rented',
    'SpecAF': 'special, elderly/disabled', 'SpecStu': 'student housing',
    'SpecOvr': 'special, other', 'Ovr': 'other', 'unknown': 'unknown',
}
BOENDEFORM = {
    'SmahusAg':  'small house, owner-occupied',
    'SmahusBo':  'small house, tenant-owned',
    'SmahusHy':  'small house, rented',
    'FlerAg':    'multi-dwelling, owner-occupied',
    'FlerBo':    'multi-dwelling, tenant-owned',
    'FlerHy':    'multi-dwelling, rented',
    'SpecAF':    'special housing, elderly/disabled',
    'SpecStu':   'special housing, students',
    'SpecOvr':   'special housing, other',
    'Ovr':       'other housing',
    'unknown':   'unknown',
}
for _k, _name in BOENDEFORM.items():
    HOUSEHOLD[f'Boendeform_mode_{_k}'] = (f'Housing: {BOENDEFORM_SHORT[_k]}',
                                         f'Indicator: modal housing/tenure type is "{_name}"')

# ---------------------------------------------------------------------------
# Code-derived features (WHO ICD-10 blocks / ATC groups retained in the model)
# ---------------------------------------------------------------------------
ICD10 = {
    'D22':     'melanocytic naevi',
    'F41':     'other anxiety disorders',
    'G40-G47': 'episodic & paroxysmal disorders',
    'H49-H52': 'ocular muscle & refraction disorders',
    'H65-H75': 'middle ear & mastoid',
    'H90-H95': 'other disorders of ear',
    'J30':     'vasomotor & allergic rhinitis',
    'J35':     'chronic tonsil & adenoid disease',
    'J45':     'asthma',
    'K20-K31': 'oesophagus, stomach & duodenum',
    'K40-K46': 'hernia',
    'K55-K64': 'other diseases of intestines',
    'L60-L75': 'disorders of skin appendages',
    'L80-L99': 'other skin disorders',
    'M15-M19': 'arthrosis',
    'M20-M25': 'other joint disorders',
    'M50-M54': 'other dorsopathies (incl. back pain)',
    'M70-M79': 'other soft-tissue disorders',
    'N40-N51': 'male genital organs',
    'R00-R09': 'symptoms: circulatory & respiratory',
    'R10-R19': 'symptoms: digestive & abdomen',
    'R20-R23': 'symptoms: skin',
    'R30-R39': 'symptoms: urinary',
    'R50-R69': 'general symptoms & signs',
    'S40-S49': 'injury: shoulder & upper arm',
    'S50-S59': 'injury: elbow & forearm',
    'S60-S69': 'injury: wrist & hand',
    'S80-S89': 'injury: knee & lower leg',
    'S90-S99': 'injury: ankle & foot',
    'Z00-Z13': 'examination & investigation',
    'Z40-Z54': 'specific procedures & aftercare',
    'Z70-Z76': 'health services, other circumstances',
}

ATC = {
    'A01A':    'stomatological preparations',
    'A02B':    'drugs for peptic ulcer & GORD',
    'A06A':    'drugs for constipation',
    'A10B':    'blood-glucose-lowering drugs (non-insulin)',
    'A11C':    'vitamin A and D',
    'A12A':    'calcium',
    'B01A':    'antithrombotic agents',
    'B03B':    'vitamin B12 and folic acid',
    'C07A':    'beta blockers',
    'C07AB02': 'metoprolol',
    'C08CA':   'dihydropyridine calcium-channel blockers',
    'C09A':    'ACE inhibitors',
    'C09C':    'angiotensin II receptor blockers',
    'C10A':    'lipid-modifying agents',
    'D01A':    'topical antifungals',
    'D02A':    'emollients & protectives',
    'D07A':    'topical corticosteroids',
    'D10A':    'topical anti-acne preparations',
    'G03A':    'hormonal contraceptives',
    'G03C':    'oestrogens',
    'G04B':    'urologicals',
    'H02A':    'systemic corticosteroids',
    'H03A':    'thyroid preparations',
    'J01A':    'tetracyclines',
    'J01C':    'penicillins',
    'J01X':    'other antibacterials',
    'M01A':    'NSAIDs',
    'M03B':    'centrally acting muscle relaxants',
    'N02A':    'opioids',
    'N02B':    'other analgesics & antipyretics',
    'N05B':    'anxiolytics',
    'R01A':    'nasal decongestants (topical)',
    'R03A':    'inhaled adrenergics',
    'R03AK':   'adrenergic + corticosteroid inhalers',
    'R03BA':   'inhaled glucocorticoids',
    'R05C':    'expectorants',
    'R05F':    'cough suppressant/expectorant combinations',
    'R06A':    'systemic antihistamines',
    'S01G':    'ophthalmic decongestants & antiallergics',
    'S03C':    'eye/ear corticosteroid + anti-infective',
}

# Short names used in figure labels (full names above go into descriptions)
ICD10_SHORT = {
    'D22': 'naevi', 'F41': 'anxiety', 'G40-G47': 'epilepsy/migraine',
    'H49-H52': 'refraction/eye muscle', 'H65-H75': 'middle ear',
    'H90-H95': 'other ear', 'J30': 'allergic rhinitis', 'J35': 'tonsils/adenoids',
    'J45': 'asthma', 'K20-K31': 'oesophagus/stomach', 'K40-K46': 'hernia',
    'K55-K64': 'other intestinal', 'L60-L75': 'skin appendages',
    'L80-L99': 'other skin', 'M15-M19': 'arthrosis', 'M20-M25': 'other joint',
    'M50-M54': 'back/neck', 'M70-M79': 'soft tissue', 'N40-N51': 'male genital',
    'R00-R09': 'circ./resp. symptoms', 'R10-R19': 'abdominal symptoms',
    'R20-R23': 'skin symptoms', 'R30-R39': 'urinary symptoms',
    'R50-R69': 'general symptoms', 'S40-S49': 'shoulder injury',
    'S50-S59': 'forearm injury', 'S60-S69': 'hand injury',
    'S80-S89': 'knee/leg injury', 'S90-S99': 'ankle/foot injury',
    'Z00-Z13': 'examination', 'Z40-Z54': 'procedures/aftercare',
    'Z70-Z76': 'other contact',
}
ATC_SHORT = {
    'A01A': 'stomatologicals', 'A02B': 'ulcer/reflux drugs', 'A06A': 'laxatives',
    'A10B': 'oral antidiabetics', 'A11C': 'vitamin A/D', 'A12A': 'calcium',
    'B01A': 'antithrombotics', 'B03B': 'B12/folic acid', 'C07A': 'beta blockers',
    'C07AB02': 'metoprolol', 'C08CA': 'dihydropyridines', 'C09A': 'ACE inhibitors',
    'C09C': 'ARBs', 'C10A': 'lipid modifiers', 'D01A': 'topical antifungals',
    'D02A': 'emollients', 'D07A': 'topical steroids', 'D10A': 'anti-acne',
    'G03A': 'contraceptives', 'G03C': 'oestrogens', 'G04B': 'urologicals',
    'H02A': 'systemic steroids', 'H03A': 'thyroid', 'J01A': 'tetracyclines',
    'J01C': 'penicillins', 'J01X': 'other antibiotics', 'M01A': 'NSAIDs',
    'M03B': 'muscle relaxants', 'N02A': 'opioids', 'N02B': 'other analgesics',
    'N05B': 'anxiolytics', 'R01A': 'nasal decongestants',
    'R03A': 'inhaled adrenergics', 'R03AK': 'combination inhalers',
    'R03BA': 'inhaled steroids', 'R05C': 'expectorants',
    'R05F': 'cough combinations', 'R06A': 'antihistamines',
    'S01G': 'eye antiallergics', 'S03C': 'eye/ear steroid+AB',
}

PREFIX = {'ov':      ('OP',   'Outpatient diagnoses (National Patient Register), ICD-10', ICD10, ICD10_SHORT),
          'sv':      ('IP',   'Inpatient diagnoses (National Patient Register), ICD-10',  ICD10, ICD10_SHORT),
          'lmed':    ('Rx',   'Dispensed prescriptions (Prescribed Drug Register), ATC',  ATC,   ATC_SHORT),
          'contact': ('1177', '1177 healthcare-advice contact reason', {}, {})}

INDICATOR = {'occurred':        ('any',     'at least one record'),
             'exceeds_median':  ('>med', 'household count above the training-fold median'),
             'exceeds_p75':     ('>P75',    'household count above the training-fold 75th percentile')}

_CODE_RE = re.compile(r'^(ov|sv|lmed|contact)_(.+)_(occurred|exceeds_median|exceeds_p75)$')


def _split_transform(col):
    """Strip the _log / _std suffixes; return (base, transform tags)."""
    tags = []
    if col.endswith('_std'):
        col, tags = col[:-4], ['z']
    if col.endswith('_log'):
        col, tags = col[:-4], ['log'] + tags
    return col, tags


def describe(col):
    """Return (short label, long description, group) for a feature column."""
    m = _CODE_RE.match(col)
    if m:
        prefix, code, ind = m.groups()
        short_src, long_src, vocab, vocab_short = PREFIX[prefix]
        name = vocab.get(code)
        ind_short, ind_long = INDICATOR[ind]
        if name is None:            # not in the curated list: show the code only
            short = f'{short_src} {code}: {ind_short}'
            desc = f'{long_src} {code}: {ind_long}'
        else:
            short = f'{short_src} {code} {vocab_short[code]}: {ind_short}'
            desc = f'{long_src} {code} ({name}): {ind_long}'
        return short, desc, long_src.split(' (')[0]

    base, tags = _split_transform(col)
    if base in HOUSEHOLD:
        short, desc = HOUSEHOLD[base]
        if tags:
            short = f'{short} ({", ".join(tags)})'
            desc = f'{desc}; ' + ', '.join(
                {'log': 'log1p-transformed', 'z': 'standardised (z-score)'}[t] for t in tags)
        return short, desc, 'Household-level'
    return col, '(no description)', 'Unknown'


def label(col):
    return describe(col)[0]


def markdown_table(columns):
    """Markdown table for DATA_DICTIONARY.md Section 9."""
    lines = ['| Column | Figure label | Description | Group |',
             '|---|---|---|---|']
    for c in columns:
        short, desc, group = describe(c)
        lines.append(f'| `{c}` | {short} | {desc} | {group} |')
    return '\n'.join(lines)


if __name__ == '__main__':
    # python feature_labels.py --markdown <explain_sample.csv>
    import csv
    path = sys.argv[2] if len(sys.argv) > 2 else \
        'experiment_results/TabPFN_XAI_Results_Full/global_importance/global_shap_explain_sample.csv'
    cols = next(csv.reader(open(path)))
    unknown = [c for c in cols if describe(c)[2] == 'Unknown']
    if unknown:
        sys.exit(f'No label for: {unknown}')
    print(markdown_table(cols))
