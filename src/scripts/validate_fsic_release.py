"""Validate a classifier-preserving FSIC repair before release.

Run after rebuilding from the checksum-pinned previous candidate. Produces
machine-readable evidence; it does not merge or deploy anything.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
import train_model  # trusted, checksum-verified model class compatibility
from src.classifier_integrity import verify_classifier_file
from src.feature_engineering import CrisisFeatureEngineer
from src.scripts.audit_model_policy import build_policy_audit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(root):
    return pickle.loads((root / 'cache/risk_model.pkl').read_bytes())


def replay(root, model):
    pipe = pickle.loads((root / 'cache/inference_pipeline.pkl').read_bytes())['pillar_pipeline']
    features = model['feature_values']
    out = pipe.transform(features).set_index('country_code')
    probability = features.set_index('country_code')['crisis_prob'].reindex(out.index)
    out['risk_score'] = (out.risk_score + 0.1 * ((1 + 9 * probability) - out.risk_score).clip(lower=0)).clip(1, 10)
    stored = model['country_scores'].set_index('country_code')
    difference = out.risk_score.reindex(stored.index) - stored.risk_score
    assert difference.abs().max() < 1e-10, difference.abs().nlargest(10)
    return float(difference.abs().max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    root = Path('.')
    manifest = json.loads(Path('artifacts/data_manifest.json').read_text())
    for name, meta in manifest['artifacts'].items():
        p = root / name
        assert p.stat().st_size == meta['bytes'], name
        assert sha(p) == meta['sha256'], name
    classifier = verify_classifier_file()
    assert sha('cache/crisis_classifier.pkl') == sha(args.previous / 'cache/crisis_classifier.pkl') == sha(args.baseline / 'cache/crisis_classifier.pkl')
    source_checks = {}
    for name in ['FSIC', 'MFS', 'WEO', 'WGI', 'FSIBSIS']:
        p = f'cache/{name}_cache.parquet'
        source_checks[name] = sha(p) == sha(args.previous / p)
        assert source_checks[name], name
    model = load_model(root)
    previous = load_model(args.previous)
    baseline = load_model(args.baseline)
    scores = model['country_scores'].set_index('country_code').sort_index()
    old_scores = baseline['country_scores'].set_index('country_code').sort_index()
    previous_scores = previous['country_scores'].set_index('country_code').sort_index()
    assert scores.index.is_unique and len(scores) == 201
    assert set(scores.index) == set(old_scores.index) == set(previous_scores.index)
    assert np.isfinite(scores.risk_score).all() and scores.risk_score.between(1, 10).all()
    reproduction = {k: replay(r, m) for k, r, m in [('baseline', args.baseline, baseline), ('previous', args.previous, previous), ('release', root, model)]}
    assert build_policy_audit(model['feature_values']) == json.loads(Path('artifacts/model_policy_audit.json').read_text())
    raw = pd.read_parquet('cache/crisis_features.parquet').set_index('country_code').sort_index()
    actual = model['feature_values'].set_index('country_code').reindex(raw.index)
    for col in raw.columns:
        equal = raw[col].eq(actual[col]) | (raw[col].isna() & actual[col].isna())
        assert equal.all(), f'Raw/model feature mismatch: {col}'
    fsic = pd.read_parquet('cache/FSIC_cache.parquet')
    eng = CrisisFeatureEngineer()
    def extract(frame):
        with contextlib.redirect_stdout(io.StringIO()):
            result = eng.extract_fsic_features(frame, as_of_date='2026-09-15')
        return result.set_index('country_code').sort_index().sort_index(axis=1)
    selected = extract(fsic)
    permutations = [('reversed', fsic.iloc[::-1])]
    permutations += [(f'shuffled_{seed}', fsic.sample(frac=1, random_state=seed)) for seed in (7, 42, 2026)]
    for label, frame in permutations:
        pd.testing.assert_frame_equal(selected, extract(frame), check_exact=True)
    extract(fsic)
    pd.DataFrame(eng.fsic_selection_audit).to_csv(out / 'fsic-selected-observations.csv', index=False)
    selected.to_csv(out / 'fsic-selected-features.csv')
    assert selected.loc['VNM', 'tier1_capital'] == 10.5859364868
    # Record all changed feature cells, not only the originally reported country.
    prior_raw = pd.read_parquet(args.previous / 'cache/crisis_features.parquet').set_index('country_code').reindex(raw.index)
    changes = []
    for col in raw.columns.intersection(prior_raw.columns):
        equal = raw[col].eq(prior_raw[col]) | (raw[col].isna() & prior_raw[col].isna())
        for cc in raw.index[~equal]:
            changes.append({'country_code': cc, 'field': col, 'previous': prior_raw.loc[cc, col], 'repaired': raw.loc[cc, col]})
    pd.DataFrame(changes, columns=['country_code', 'field', 'previous', 'repaired']).to_csv(out / 'repair-feature-changes.csv', index=False)
    # No unrelated feature engineering changes are permitted by this repair.
    allowed = {'tier1_capital', 'tier1_capital_year', 'capital_quality'}
    assert {x['field'] for x in changes} <= allowed, changes
    review = pd.DataFrame(index=scores.index)
    for label, frame in [('production', old_scores), ('previous_candidate', previous_scores), ('repaired_candidate', scores)]:
        review[label] = frame.risk_score
    review['vs_production'] = review.repaired_candidate - review.production
    review['repair_effect'] = review.repaired_candidate - review.previous_candidate
    for col in ('risk_category', 'risk_tier', 'crisis_prob', 'country_name'):
        if col in scores:
            review[col] = scores[col]
    review = review.sort_values('vs_production', key=lambda x: x.abs(), ascending=False)
    review.to_csv(out / 'country-review.csv')
    summaries = ET.parse(out / 'tests.xml').getroot().findall('testsuite')
    tests = {k: sum(int(s.attrib.get(k, 0)) for s in summaries) for k in ['tests', 'failures', 'errors', 'skipped']}
    assert tests['tests'] > 270 and tests['failures'] == tests['errors'] == 0
    tests['passed'] = tests['tests'] - tests['skipped']
    result = {'decision': 'VALIDATED_FOR_REVIEWED_PROMOTION', 'cutoff': '2026-09-15',
              'classifier': classifier, 'source_caches_unchanged': source_checks,
              'manifest_files_verified': len(manifest['artifacts']), 'countries': len(scores),
              'exact_replay_max_errors': reproduction, 'policy_audit_exact': True,
              'raw_features_match_model': True, 'fsic_permutations_passed': [x[0] for x in permutations],
              'fsic_countries': len(selected), 'fsic_selected_cells': len(eng.fsic_selection_audit),
              'vnm_tier1': float(selected.loc['VNM', 'tier1_capital']),
              'vnm_capital_quality': float(selected.loc['VNM', 'capital_quality']),
              'repair_feature_changes': changes, 'tests': tests,
              'max_score_change_vs_production': float(review.vs_production.abs().max()),
              'max_score_change_due_to_repair': float(review.repair_effect.abs().max()),
              'score_countries_changed_by_repair': int(review.repair_effect.abs().gt(1e-10).sum()),
              'top_score_changes': json.loads(review.head(15).reset_index().to_json(orient='records')),
              'limitations': ['Existing legacy classifier preserved, not recertified.', 'PCA/imputation refit remains part of the existing refresh methodology.', 'This validates selected model inputs, not every unused raw series.']}
    (out / 'validation.json').write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
