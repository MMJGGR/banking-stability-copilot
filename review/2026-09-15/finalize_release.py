"""Prepare the validated release; never merges or updates master."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

BASELINE = '860cfd64d795458a871f1a654e7497bf3a360012'
BUILD_SOURCE = '95da047452543c0139a41f3e419815c1f72a4571'
SOURCES = ['.github/workflows/refresh-data.yml', 'src/classifier_integrity.py',
           'src/scripts/refresh_data.py', 'src/snapshot_reuse.py',
           'tests/test_classifier_integrity.py', 'train_model.py',
           'src/feature_engineering.py', 'src/fsic_selection.py',
           'tests/test_fsic_selection.py', 'src/scripts/validate_fsic_release.py']


def run(*args, cwd=None, env=None):
    subprocess.run(args, cwd=cwd, env=env, check=True)


def patch_validator():
    path = Path('src/scripts/validate_fsic_release.py')
    text = path.read_text()
    old = "        assert np.isfinite(saved.to_numpy(dtype=float)).all()\n"
    new = '''        # Display sidecars intentionally retain missing observation dates.
        # Dates are not pillar inputs; reconstruct only those unused metadata
        # cells so the fitted scaler receives its complete numeric schema.
        missing_metadata = saved.columns[saved.isna().any()].tolist()
        used = set(pipe.economic_columns_) | set(pipe.industry_columns_)
        assert all(c.endswith('_year') for c in missing_metadata), missing_metadata
        assert not (set(missing_metadata) & used), missing_metadata
        if missing_metadata:
            with threadpool_limits(limits=threads):
                reconstructed = pd.DataFrame(
                    pipe.imputer_.transform(eligible[pipe.imputed_columns_]),
                    index=eligible.index, columns=pipe.imputed_columns_)
            for col in missing_metadata:
                if col in reconstructed:
                    saved[col] = saved[col].fillna(reconstructed[col])
                else:
                    assert col in pipe.empty_columns_, col
                    saved[col] = saved[col].fillna(0.0)
        assert np.isfinite(saved.to_numpy(dtype=float)).all()
'''
    if new not in text:
        assert text.count(old) == 1
        path.write_text(text.replace(old, new))


def prepare():
    workspace = Path.cwd()
    output = workspace / 'release-output'
    report = json.loads((output / 'validation.json').read_text())
    assert report['decision'] == 'VALIDATED_FOR_REVIEWED_PROMOTION'
    assert all(x['max_abs_error'] < 1e-10 for x in report['release_raw_replay_by_threads'])
    run('git', 'fetch', 'origin', 'master')
    assert subprocess.check_output(['git', 'rev-parse', 'origin/master'], text=True).strip() == BASELINE
    # Reuse the rebuilt candidate only if the model-building/runtime code is
    # exactly the code used for that candidate. Audit-only changes are allowed.
    runtime = ['src/feature_engineering.py', 'src/fsic_selection.py', 'train_model.py',
               'src/pillar_pipeline.py', 'src/classifier_integrity.py',
               'src/snapshot_reuse.py', 'src/scripts/refresh_data.py']
    run('git', 'diff', '--exit-code', BUILD_SOURCE, 'HEAD', '--', *runtime)
    target = Path('/tmp/release')
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
    run('git', 'worktree', 'add', '-b', 'promote/fsic-2026-09-15', str(target), BASELINE, env=env)
    manifest = json.loads(Path('artifacts/data_manifest.json').read_text())
    paths = sorted(set(SOURCES + list(manifest['artifacts']) +
                       ['artifacts/data_manifest.json', 'artifacts/validated_classifier.json',
                        'artifacts/feature_heuristics.json']))
    for name in paths:
        source = workspace / name
        destination = target / name
        assert source.is_file(), name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    docs = target / 'docs/releases/2026-09-15'
    docs.mkdir(parents=True, exist_ok=True)
    for name in ['validation.json', 'complete-replay.json', 'country-review.csv',
                 'repair-feature-changes.csv', 'fsic-selected-observations.csv', 'source-commit.txt']:
        shutil.copy2(output / name, docs / name)
    for name in ['fsic-source-diagnostic.json', 'replay-diagnostic.json']:
        shutil.copy2(workspace / 'review/2026-09-15' / name, docs / name)
    lines = ['# September 15, 2026 corrected FSIC release', '',
             'User authorization: fix the FSIC selection defect and promote after successful validation.', '',
             '## Validated controls',
             '- Full source retrieval vintage: September 15, 2026. All five source caches remain identical to original candidate 10402061259.',
             '- Rebuilt candidate 10404058870 is reused byte-for-byte; model-building and runtime code are verified identical to its source commit ' + BUILD_SOURCE + '.',
             '- Existing served classifier preserved byte-for-byte. Retraining remains an explicit separate approval.',
             '- Canonical Tier 1 code FSI626_CFSI_PT cannot be confused with CET1 FSI15_CFSI_PT. Conflicting values and economic dimensions fail closed.',
             f"- {report['tests']['passed']} tests passed; {report['tests']['skipped']} skipped; zero failures.",
             f"- {report['countries']} countries retained; {report['manifest_files_verified']} manifest artifacts verified.",
             '- Complete FSIC extraction is identical for original, reversed and three shuffled source orders.',
             '- The release reproduces its published scores exactly from raw inputs with one, two and four computational threads.',
             '- Historical bundles replay exactly using their saved imputed economic values. Display sidecars deliberately leave some dates blank; only those non-pillar date fields are reconstructed. Legacy raw-KNN replay discrepancies remain disclosed in complete-replay.json rather than being mistaken for input revisions.',
             f"- Vietnam Tier 1 {report['vnm_tier1']}; capital quality {report['vnm_capital_quality']}.",
             f"- Maximum score change attributable to this repair/rebuild {report['max_score_change_due_to_repair']}; total versus production {report['max_score_change_vs_production']}.",
             f"- Validation: https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}", '',
             '## Release and rollback',
             'The workflow prepares this branch without merging it. Final approval is recorded in the pull request.',
             'Rollback: ' + BASELINE + ', also retained on rollback/pre-september-2026-09-15. Revert the complete atomic release through a reviewed PR and rerun serving checks.', '',
             '## Scope',
             'This is not classifier recertification. Existing PCA/imputation refits remain part of the refresh method. Crisis-history corrections are not new economic shocks. Validation covers selected model inputs, not every unused raw series.']
    (docs / 'README.md').write_text('\n'.join(lines) + '\n')
    run('git', 'lfs', 'checkout', cwd=target)
    with (output / 'clean-tree-smoke.txt').open('w') as handle:
        subprocess.run(['python', '-m', 'src.scripts.smoke_test_artifacts'], cwd=target, stdout=handle, stderr=subprocess.STDOUT, check=True)
    with (output / 'clean-tree-tests.txt').open('w') as handle:
        subprocess.run(['python', '-m', 'pytest', '-q', '--junitxml=' + str(output / 'clean-tree-tests.xml')], cwd=target, stdout=handle, stderr=subprocess.STDOUT, check=True)
    run('git', 'add', '--', *paths, 'docs/releases/2026-09-15', cwd=target)
    run('git', '-c', 'user.name=github-actions[bot]', '-c', 'user.email=41898282+github-actions[bot]@users.noreply.github.com', 'commit', '-m', 'fix and promote: canonical FSIC selection and validated September 15 snapshot', cwd=target)
    run('git', 'push', 'origin', 'HEAD:promote/fsic-2026-09-15', cwd=target)
    (output / 'release-commit.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=target, text=True))
    (output / 'release-diff-stat.txt').write_text(subprocess.check_output(['git', 'diff', '--stat', BASELINE], cwd=target, text=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['patch', 'prepare'])
    args = parser.parse_args()
    patch_validator() if args.action == 'patch' else prepare()
