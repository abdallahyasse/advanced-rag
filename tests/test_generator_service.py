import pytest

from app.generation.generator_service import GeneratorService


def test_extract_degree_from_context():
    context = (
        "Education: Bachelor of Science in "
        "Computer Science & Artificial Intelligence."
    )

    answer = GeneratorService._extract_degree(
        context=context,
        question="What degree did Abdalla earn?",
    )

    assert answer == (
        "Bachelor of Science in "
        "Computer Science & Artificial Intelligence"
    )


def test_extract_degree_returns_none_without_degree_in_question():
    context = "Bachelor of Science in Computer Science."

    answer = GeneratorService._extract_degree(
        context=context,
        question="What programming languages does he know?",
    )

    assert answer is None


def test_extract_degree_returns_none_without_degree_in_context():
    answer = GeneratorService._extract_degree(
        context="He graduated from Helwan University.",
        question="What degree did he earn?",
    )

    assert answer is None


def test_extract_sat_annotator_techniques():
    context = (
        "The SAT-Annotator project supports automatic and manual "
        "image segmentation. It uses Computer Vision for geospatial "
        "image analysis."
    )

    answer = GeneratorService._extract_sat_annotator_techniques(
        context=context,
        question="What techniques were used in SAT-Annotator?",
    )

    assert answer == (
        "SAT-Annotator used automatic and manual image segmentation "
        "and Computer Vision for geospatial image analysis."
    )


def test_sat_annotator_extraction_requires_project_evidence():
    answer = GeneratorService._extract_sat_annotator_techniques(
        context="The project uses Computer Vision.",
        question="What techniques were used in SAT-Annotator?",
    )

    assert answer is None


@pytest.mark.parametrize(
    "question",
    [
        "What programming language was used to develop the mobile app?",
        "Which technology was used for Android application development?",
        "What language was used to build the iOS app?",
    ],
)
def test_mobile_question_detection(question):
    assert GeneratorService._is_mobile_development_question(question)


def test_mobile_question_without_explicit_evidence_returns_insufficient_info(
    monkeypatch,
):
    def unexpected_get_model(cls):
        raise AssertionError(
            "The model should not load when mobile-development evidence is missing."
        )

    monkeypatch.setattr(
        GeneratorService,
        "get_model",
        classmethod(unexpected_get_model),
    )

    answer = GeneratorService.generate(
        context=(
            "Skills: Python, Java, JavaScript, and Flutter."
        ),
        question=(
            "What programming language was used to develop "
            "the mobile app?"
        ),
    )

    assert answer == GeneratorService.INSUFFICIENT_INFORMATION


def test_mobile_question_with_explicit_evidence_passes_guard():
    context = (
        "Developed a mobile app using Flutter."
    )

    assert GeneratorService._has_mobile_development_evidence(context)


def test_empty_context_returns_insufficient_info_without_loading_model(
    monkeypatch,
):
    def unexpected_get_model(cls):
        raise AssertionError(
            "The model should not load when context is empty."
        )

    monkeypatch.setattr(
        GeneratorService,
        "get_model",
        classmethod(unexpected_get_model),
    )

    answer = GeneratorService.generate(
        context="   ",
        question="What is Machine Learning?",
    )

    assert answer == GeneratorService.INSUFFICIENT_INFORMATION
