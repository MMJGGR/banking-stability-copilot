"""Prepare a tested atomic release branch, never merge or modify master."""
import datetime
import json
import os
import shutil
import subprocess
from pathlib import Path

BASELINE = '860cfd64d795458a871f1a654e7497bf3a360012'
SOURCES = ['.github/workflows/refresh-data.yml', 'src/classifier_integrity.py',
           'src/scripts/refresh_data.py', 'src/snapshot_reuse.py',
           'tests/test_classifier_integrity.py', 'train_model.py',
           'src/feature_engineering.py', 'src/fsic_selection.py',
           'tests/test_fsic_selection.py', 'src/scripts/validate_fsic_release.py',
           'src/pillar_pipeline.py', 'src/stable_distance.py', 'tests/test_stable_distance.py']


def run(*args, cwd=None, env=None):
    subprocess.run(args, cwd=cwd, env=env, check=True)


def main():
    workspace = Path.cwd()
    output = workspace / 'release-output'
    report = json.loads((output / 'validation.json').read_text())
    assert report['decision'] == 'VALIDATED_FOR_REVIEWED_PROMOTION'
    assert all(x['max_abs_error'] < 1e-10 for x in report['release_raw_replay_by_threads'])
    build_source = (output / 'source-commit.txt').read_text().strip()
    run('git', 'diff', '--exit-code', build_source, '--', *SOURCES)
    run('git', 'fetch', 'origin', 'master')
    assert subprocess.check_output(['git', 'rev-parse', 'origin/master'], text=True).strip() == BASELINE
    target = Path('/tmp/release')
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
    run('git', 'worktree', 'add', '-b', 'promote/fsic-2026-09-15', str(target), BASELINE, env=env)
    manifest = json.loads(Path('artifacts/data_manifest.json').read_text())
    paths = sorted(set(SOURCES + list(manifest['artifacts']) +
                       ['artifacts/data_manifest.json', 'artifacts/validated_classifier.json',
                        'artifacts/feature_heuristics.json']))
    for name in paths:
        source, destination = workspace / name, target / name
        assert source.is_file(), name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    docs = target / 'docs/releases/2026-09-15'
    docs.mkdir(parents=True, exist_ok=True)
    for name in ['validation.json', 'complete-replay.json', 'country-review.csv',
                 'repair-feature-changes.csv', 'fsic-selected-observations.csv', 'source-commit.txt']:
        shutil.copy2(output / name, docs / name)
    shutil.copy2(workspace / 'review/2026-09-15/fsic-source-diagnostic.json', docs / 'fsic-source-diagnostic.json')
    approval = {'status': 'validated_pending_final_PR_review',
                'user_authorization': 'Fix and promote after successful independent validation.',
                'validation_run': os.environ['GITHUB_RUN_ID'], 'build_source_commit': build_source,
                'baseline_commit': BASELINE,
                'scope': 'Same September 15 source vintage; canonical FSIC selection; mathematically equivalent stable neighbor-distance calculation; existing classifier preserved.',
                'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (docs / 'approval.json').write_text(json.dumps(approval, indent=2) + '\n')
    lines = ['# September 15, 2026 corrected banking-data release', '',
             'User authorized correction and promotion after independent validation. This workflow prepares a branch; the final PR review and merge are separate.', '',
             '## Corrections',
             '- FSIC Tier 1 uses canonical FSI626_CFSI_PT, never Common Equity Tier 1 FSI15_CFSI_PT. Conflicting observations and economic dimensions fail closed.',
             '- The pillar KNN metric now subtracts observed coordinates before taking their norm. This implements the same missing-aware Euclidean metric without cancellation between large nominal values. Neighbor count, distance weighting, feature units, pillar policy and existing classifier weights are unchanged.',
             '- Historical classifiers/pipelines retain their existing metric. Only the rebuilt pillar pipeline adopts the numeric fix; no classifier retraining occurs.', '',
             '## Validation',
             '- All five source caches match the verified original September 15 artifact 10402061259 byte-for-byte.',
             f"- {report['tests']['passed']} tests passed; {report['tests']['skipped']} skipped; zero failures.",
             f"- {report['countries']} scored countries retained; {report['manifest_files_verified']} manifest artifacts verified.",
             '- Complete FSIC extraction agrees exactly for original, reversed and three shuffled source orders.',
             '- New scores reproduce exactly from full raw model inputs at one, two and four computational threads. Historical exact replay uses saved imputed economic values; legacy raw-KNN discrepancies remain explicitly disclosed.',
             f"- Vietnam Tier 1: {report['vnm_tier1']}; capital quality: {report['vnm_capital_quality']}.",
             f"- Maximum score movement from both repairs/rebuild: {report['max_score_change_due_to_repair']}; versus production: {report['max_score_change_vs_production']}.",
             f'- Source commit: {build_source}',
             f"- Evidence: https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}", '',
             '## Rollback',
             'Revert the complete atomic release through a reviewed PR and rerun serving checks. Baseline: ' + BASELINE + ', preserved at rollback/pre-september-2026-09-15.', '',
             '## Scope limits',
             'This does not recertify the predictive classifier. PCA/imputation refits remain part of refresh methodology. Historical crisis-reference corrections are not new economic shocks. Selected-input validation does not certify every unused raw series.']
    (docs / 'README.md').write_text('\n'.join(lines) + '\n')
    run('git', 'lfs', 'checkout', cwd=target)
    with (output / 'clean-tree-smoke.txt').open('w') as log:
        subprocess.run(['python', '-m', 'src.scripts.smoke_test_artifacts'], cwd=target, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (output / 'clean-tree-tests.txt').open('w') as log:
        subprocess.run(['python', '-m', 'pytest', '-q', '--junitxml=' + str(output / 'clean-tree-tests.xml')], cwd=target, stdout=log, stderr=subprocess.STDOUT, check=True)
    run('git', 'add', '--', *paths, 'docs/releases/2026-09-15', cwd=target)
    run('git', '-c', 'user.name=github-actions[bot]', '-c', 'user.email=41898282+github-actions[bot]@users.noreply.github.com', 'commit', '-m', 'fix: canonical FSIC selection, stable imputation and September 15 serving snapshot', cwd=target)
    run('git', 'push', 'origin', 'HEAD:promote/fsic-2026-09-15', cwd=target)
    (output / 'release-commit.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=target, text=True))
    (output / 'release-diff-stat.txt').write_text(subprocess.check_output(['git', 'diff', '--stat', BASELINE], cwd=target, text=True))


if __name__ == '__main__':
    main()
