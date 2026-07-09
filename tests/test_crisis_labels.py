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
