from sentence_transformers import CrossEncoder

from app.config.settings import settings


class RerankerService:
    """
    Production Reranker Service.

    Responsible for:
    - Cross-Encoder reranking
    - RRF + reranker score fusion
    - Query-aware relevance boosting
    """

    _model: CrossEncoder | None = None

    @classmethod
    def get_model(cls):
        """
        Load the Cross-Encoder model once and reuse it.
        """

        if cls._model is None:
            cls._model = CrossEncoder(
                settings.reranker_model
            )

        return cls._model

    @staticmethod
    def _rank_documents(
        documents: list[dict],
        score_key: str,
        rank_key: str,
    ) -> None:
        """
        Assign ranking positions based on a score.
        """

        ranked = sorted(
            documents,
            key=lambda x: x.get(
                score_key,
                0.0,
            ),
            reverse=True,
        )

        for rank, doc in enumerate(
            ranked,
            start=1,
        ):
            doc[rank_key] = rank

    @staticmethod
    def _detect_query_type(
        query: str,
    ) -> str:
        """
        Detect the general intent of the query.

        This is intentionally lightweight.
        It is used only for relevance boosting,
        not as the main ranking mechanism.
        """

        query_lower = query.lower().strip()

        # ----------------------------------------
        # Project queries
        # ----------------------------------------

        project_terms = [
            "graduation project",
            "project",
            "sat-annotator",
            "sat annotator",
            "technologies used",
            "technology used",
            "what did he build",
            "what did he develop",
            "developed",
            "built",
        ]

        if any(
            term in query_lower
            for term in project_terms
        ):
            return "project"

        # ----------------------------------------
        # Education queries
        # ----------------------------------------

        education_terms = [
            "degree",
            "education",
            "university",
            "college",
            "bachelor",
            "master",
            "phd",
            "major",
            "study",
            "studied",
            "where did he study",
            "where did abdalla study",
            "graduated from",
            "university grade",
            "grade",
        ]

        if any(
            term in query_lower
            for term in education_terms
        ):
            return "education"

        # ----------------------------------------
        # Training / certification queries
        # ----------------------------------------

        training_terms = [
            "diploma",
            "certificate",
            "certification",
            "training",
            "nti",
            "amit",
            "score",
        ]

        if any(
            term in query_lower
            for term in training_terms
        ):
            return "training"

        # ----------------------------------------
        # Skills queries
        # ----------------------------------------

        skills_terms = [
            "programming language",
            "programming languages",
            "languages does he know",
            "skills",
            "technologies does he know",
            "frameworks",
            "libraries",
            "tools",
        ]

        if any(
            term in query_lower
            for term in skills_terms
        ):
            return "skills"

        return "general"

    @staticmethod
    def _query_section_boost(
        query: str,
        text: str,
    ) -> float:
        """
        Apply a small query-aware relevance boost.

        The Cross-Encoder remains the main ranking signal.
        """

        text_lower = text.lower()

        query_type = (
            RerankerService._detect_query_type(
                query
            )
        )

        # ========================================
        # EDUCATION
        # ========================================

        if query_type == "education":

            # Strong evidence
            if (
                "bachelor of science" in text_lower
                or "master of science" in text_lower
                or "ph.d" in text_lower
                or "phd" in text_lower
            ):
                return 4.0

            if "major:" in text_lower:
                return 3.5

            if "grade:" in text_lower:
                return 3.5

            if "education" in text_lower:
                return 3.0

            if "helwan university" in text_lower:
                return 2.5

            if "university" in text_lower:
                return 1.5

            return 0.0

        # ========================================
        # PROJECT
        # ========================================

        if query_type == "project":

            if "sat-annotator" in text_lower:
                return 4.0

            if "graduation project" in text_lower:
                return 3.5

            if (
                "developed" in text_lower
                or "built" in text_lower
            ):
                return 2.0

            if (
                "computer vision" in text_lower
                or "fastapi" in text_lower
                or "sam" in text_lower
            ):
                return 2.0

            return 0.0

        # ========================================
        # TRAINING
        # ========================================

        if query_type == "training":

            if "nti" in text_lower:
                return 4.0

            if "amit" in text_lower:
                return 4.0

            if "diploma" in text_lower:
                return 3.0

            if "certification" in text_lower:
                return 2.0

            if "score" in text_lower:
                return 1.5

            return 0.0

        # ========================================
        # SKILLS
        # ========================================

        if query_type == "skills":

            if "programming" in text_lower:
                return 4.0

            if "python" in text_lower:
                return 3.0

            if (
                "tensorflow" in text_lower
                or "pytorch" in text_lower
                or "scikit-learn" in text_lower
            ):
                return 2.5

            if (
                "docker" in text_lower
                or "git" in text_lower
            ):
                return 2.0

            return 0.0

        return 0.0

    @classmethod
    def rerank(
        cls,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank retrieved documents.

        Ranking priority:

        1. Cross-Encoder relevance
        2. RRF retrieval signal
        3. Query-aware section relevance

        The Cross-Encoder has the strongest influence.
        """

        if not documents:
            return []

        model = cls.get_model()

        # ========================================
        # Cross-Encoder Scoring
        # ========================================

        pairs = [
            (
                query,
                doc["text"],
            )
            for doc in documents
        ]

        scores = model.predict(pairs)

        for doc, score in zip(
            documents,
            scores,
        ):
            doc["rerank_score"] = float(score)

            doc["section_boost"] = (
                cls._query_section_boost(
                    query=query,
                    text=doc["text"],
                )
            )

        # ========================================
        # Reranker Rank
        # ========================================

        cls._rank_documents(
            documents=documents,
            score_key="rerank_score",
            rank_key="rerank_rank",
        )

        # ========================================
        # RRF Rank
        # ========================================

        cls._rank_documents(
            documents=documents,
            score_key="rrf_score",
            rank_key="rrf_rank",
        )

        # ========================================
        # Final Score
        #
        # Cross-Encoder = 70%
        # RRF          = 20%
        # Section      = 10%
        # ========================================

        for doc in documents:

            rerank_rank = doc["rerank_rank"]
            rrf_rank = doc["rrf_rank"]
            section_boost = doc.get(
                "section_boost",
                0.0,
            )

            doc["final_score"] = (
                (0.70 / rerank_rank)
                + (0.20 / rrf_rank)
                + (0.10 * section_boost)
            )

        # ========================================
        # Final Sorting
        # ========================================

        documents.sort(
            key=lambda x: x["final_score"],
            reverse=True,
        )

        return documents[:top_k]