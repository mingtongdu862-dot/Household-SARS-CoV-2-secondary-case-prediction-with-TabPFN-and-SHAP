"""
Per-code data-dictionary appendix generator.

Run this ON THE SECURE ANALYSIS ENVIRONMENT (Bianca), where the raw
Config.FEATURES_DIR pickles are available -- it cannot run on a machine
without access to the raw register extracts. It does not train any model or
touch the label; it only enumerates the code vocabulary that
feature_extraction.py turns into ov_*/sv_*/lmed_*/contact_* columns.

Output: code_dictionary_appendix.csv, one row per (group, code), which can
be cross-referenced against the public WHO ICD-10 (ov_*, sv_*) and WHO
ATC/DDD (lmed_*) classifications to fill in a clinical-interpretation column,
and uploaded alongside DATA_DICTIONARY.md.

Usage:
    python code_dictionary_appendix.py [--out code_dictionary_appendix.csv]
"""

import argparse

import pandas as pd

from feature_extraction import load_unique_codes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=str, default='code_dictionary_appendix.csv')
    args = parser.parse_args()

    print("Loading unique code vocabulary from the raw register extracts...")
    codes = load_unique_codes()  # {'contact_reasons', 'atc_codes', 'ov_codes', 'sv_codes'}

    group_to_prefix = {
        'contact_reasons': 'contact_',
        'atc_codes': 'lmed_',
        'ov_codes': 'ov_',
        'sv_codes': 'sv_',
    }

    rows = []
    for group, code_set in codes.items():
        prefix = group_to_prefix.get(group, group + '_')
        for code in sorted(code_set):
            rows.append({
                'group': group,
                'feature_column': f'{prefix}{code}',
                'code': code,
                'clinical_interpretation': '',  # fill in from ICD-10 / ATC lookup
            })
        print(f"  {group}: {len(code_set):,} distinct codes")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df):,} rows to {args.out}")
    print("Next step: cross-reference 'code' against the WHO ICD-10 classification "
          "(ov_/sv_ groups) or WHO ATC/DDD classification (lmed_ group) to fill in "
          "'clinical_interpretation'. Note this lists the full pre-selection "
          "vocabulary; only columns retained after the per-fold chi-square feature "
          "selection (see DATA_DICTIONARY.md, Section 5) appear in the final model.")


if __name__ == '__main__':
    main()
