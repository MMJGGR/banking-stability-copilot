import pandas as pd

from src.crisis_labels import CrisisLabels


def test_2026_systemic_crisis_update_is_used():
    labels = CrisisLabels()

    assert labels.SOURCE_COVERAGE_END_YEAR == 2025
    assert labels.is_crisis_year("AGO", 2016)
    assert labels.is_crisis_year("AZE", 2015)
    assert labels.is_crisis_year("KAZ", 2014)
    assert labels.is_crisis_year("LBN", 2019)
    assert labels.is_crisis_year("TJK", 2016)

    # Currency or sovereign stress not classified as systemic banking crises
    # in the official 2026 appendix must not become positive targets.
    assert not labels.is_crisis_year("ARG", 2018)
    assert not labels.is_crisis_year("TUR", 2018)
    assert not labels.is_crisis_year("GHA", 2022)


def test_borderline_crises_are_excluded_by_default():
    core = CrisisLabels()
    inclusive = CrisisLabels(include_borderline=True)

    for country, year in (("NIC", 2018), ("LKA", 2023), ("VNM", 2022)):
        assert not core.is_crisis_year(country, year)
        assert inclusive.is_crisis_year(country, year)


def test_official_episode_artifact_has_exact_published_counts_and_provenance():
    labels = CrisisLabels()
    episodes = pd.read_csv(labels.EPISODE_DATA_PATH)

    assert len(episodes) == 164
    assert episodes["classification"].value_counts().to_dict() == {
        "systemic": 161,
        "borderline": 3,
    }
    assert episodes["source_pdf_sha256"].nunique() == 1
    assert episodes["source_pdf_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert episodes["source_table"].eq("Appendix I, Table A1").all()


def test_known_table_a1_dates_are_exact():
    labels = CrisisLabels()
    assert labels.SYSTEMIC_CRISES["MOZ"] == [(1987, 1991)]
    assert labels.SYSTEMIC_CRISES["KEN"] == [(1985, 1985), (1992, 1994)]
    assert labels.SYSTEMIC_CRISES["USA"] == [(1988, 1988), (2007, 2011)]
