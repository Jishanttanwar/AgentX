"""
Fact Checker Agent Module.
Extracts atomic factual claims from research findings and rigorously
verifies them against source evidence. Enforces strict evidence standards.
"""

import time
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.llm import get_llm
from app.core.logging import get_logger
from app.schemas.intelligence import (
    AdditionalResearchRequest,
    FactualClaim,
    VerificationResult,
    VerificationStatus,
    VerificationSummary,
)
from app.schemas.research import ResearchFinding, SourceMetadata

logger = get_logger("agents.fact_checker")

CLAIM_EXTRACTION_PROMPT = """You are a Claim Extraction Specialist for AgentX.

Your task is to extract discrete, atomic, falsifiable factual claims from the provided research findings.

RULES:
1. Each claim must be a single verifiable proposition — one fact, one statistic, one regulation.
2. Do not combine multiple facts into one claim.
3. Extract only claims present in the text — never infer or assume.
4. Focus on high-value claims: statistics, dates, percentages, named entities, regulatory facts, market figures.
5. Mark claims as is_statistical=true if they contain numbers, percentages, dates, or measurable values.
"""


class ClaimsExtractionPayload(BaseModel):
    """Structured output for extracted atomic claims."""

    claims: List[FactualClaim] = Field(
        default_factory=list,
        description="List of atomic, falsifiable factual claims extracted from research findings",
    )


VERIFICATION_PROMPT = """You are the Chief Fact Verification Officer for AgentX.

Your responsibility is to rigorously verify each provided factual claim against the provided source documents.

VERIFICATION STANDARDS — FOLLOW EXACTLY:

VERIFIED: The claim is directly, word-for-word or near-equivalently confirmed by one or more provided sources.
PARTIALLY_SUPPORTED: Part of the claim is supported but other elements are absent or only implied.
CONFLICTING_EVIDENCE: Different sources provide contradictory data about the claim.
UNSUPPORTED: No provided source confirms or implies the claim. May be true but lacks evidence.

ABSOLUTE RULES:
1. A claim must NEVER be marked VERIFIED based on LLM prior knowledge alone.
2. You MUST cite the exact source URL and supporting excerpt for VERIFIED or PARTIALLY_SUPPORTED verdicts.
3. If CONFLICTING_EVIDENCE or UNSUPPORTED, flag for additional research with suggested search queries.
4. Be conservative — when in doubt, downgrade to PARTIALLY_SUPPORTED.
"""


class VerificationResultsPayload(BaseModel):
    """Structured output for verification verdicts."""

    verification_results: List[VerificationResult] = Field(
        default_factory=list,
        description="List of structured verification verdicts for each claim",
    )


class FactCheckerAgent:
    """
    Specialized agent that extracts atomic factual claims from research findings
    and verifies each claim against the collected source evidence.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        model_name: Optional[str] = None,
    ):
        self.model_name = model_name or settings.OPENAI_MODEL or "gpt-4o"
        self._llm = llm or get_llm(model=self.model_name, temperature=0.1)
        self._claim_extractor = self._llm.with_structured_output(ClaimsExtractionPayload)
        self._claim_verifier = self._llm.with_structured_output(VerificationResultsPayload)

    async def extract_claims(
        self,
        findings: List[ResearchFinding],
        max_claims: int = 20,
    ) -> List[FactualClaim]:
        """
        Phase 1: Extract discrete, falsifiable atomic claims from research findings.
        """
        if not findings:
            return []

        logger.info(f"Fact Checker extracting claims from {len(findings)} findings")

        findings_text = "\n\n".join(
            f"[FINDING_{i}] (Task: {f.task_id}) {f.finding}\n"
            f"Source: {f.source_title} | URL: {f.source_url}\n"
            f"Excerpt: \"{f.supporting_excerpt}\"\n"
            for i, f in enumerate(findings[:40], start=1)
        )

        try:
            result: ClaimsExtractionPayload = await self._claim_extractor.ainvoke(
                [
                    SystemMessage(content=CLAIM_EXTRACTION_PROMPT),
                    HumanMessage(
                        content=f"Extract atomic factual claims from these research findings:\n\n{findings_text}"
                    ),
                ]
            )

            claims = result.claims[:max_claims]
            logger.info(f"Fact Checker extracted {len(claims)} atomic claims")
            return claims

        except Exception as e:
            logger.error(f"Claim extraction failed: {str(e)}", exc_info=True)
            return []

    async def verify_claims(
        self,
        claims: List[FactualClaim],
        sources: List[SourceMetadata],
        findings: List[ResearchFinding],
    ) -> List[VerificationResult]:
        """
        Phase 2: Verify each extracted claim against the provided source corpus.
        """
        if not claims:
            return []

        logger.info(f"Fact Checker verifying {len(claims)} claims against {len(sources)} sources")

        # Build source corpus for verification context
        source_corpus = []
        for src in sources[:15]:
            content = src.raw_content or src.snippet or "No content available."
            source_corpus.append(
                f"SOURCE URL: {src.url}\n"
                f"TITLE: {src.title}\n"
                f"RELIABILITY: {src.reliability_score:.2f}\n"
                f"CONTENT EXCERPT:\n{content[:2000]}\n---"
            )

        corpus_text = "\n\n".join(source_corpus)
        claims_text = "\n".join(
            f"[{c.claim_id}] {c.claim_text} (statistical: {c.is_statistical})"
            for c in claims
        )

        try:
            result: VerificationResultsPayload = await self._claim_verifier.ainvoke(
                [
                    SystemMessage(content=VERIFICATION_PROMPT),
                    HumanMessage(
                        content=(
                            f"CLAIMS TO VERIFY:\n{claims_text}\n\n"
                            f"SOURCE EVIDENCE CORPUS:\n{corpus_text}\n\n"
                            "Verify each claim against the evidence above."
                        )
                    ),
                ]
            )

            logger.info(
                f"Fact Checker completed verification: {len(result.verification_results)} verdicts issued"
            )
            return result.verification_results

        except Exception as e:
            logger.error(f"Claim verification failed: {str(e)}", exc_info=True)
            return []

    async def run(
        self,
        findings: List[ResearchFinding],
        sources: List[SourceMetadata],
        max_claims: int = 20,
    ) -> "FactCheckerResult":
        """
        Full fact-checking pipeline: extract claims then verify them.
        Returns structured result with verdicts and additional research requests.
        """
        start_time = time.time()

        # Step 1: Extract claims
        claims = await self.extract_claims(findings, max_claims=max_claims)

        # Step 2: Verify claims against sources
        verification_results = await self.verify_claims(
            claims=claims,
            sources=sources,
            findings=findings,
        )

        # Step 3: Build verification summary and additional research requests
        additional_requests: List[AdditionalResearchRequest] = []
        verified_count = 0
        partial_count = 0
        conflict_count = 0
        unsupported_count = 0

        for vr in verification_results:
            if vr.status == VerificationStatus.VERIFIED:
                verified_count += 1
            elif vr.status == VerificationStatus.PARTIALLY_SUPPORTED:
                partial_count += 1
            elif vr.status == VerificationStatus.CONFLICTING_EVIDENCE:
                conflict_count += 1
            elif vr.status == VerificationStatus.UNSUPPORTED:
                unsupported_count += 1

            if vr.requires_additional_research and vr.suggested_followup_queries:
                additional_requests.append(
                    AdditionalResearchRequest(
                        target_claim_id=vr.claim_id,
                        reason=f"Claim unsupported: {vr.claim_text[:120]}",
                        queries=vr.suggested_followup_queries,
                        priority=1 if vr.status == VerificationStatus.UNSUPPORTED else 2,
                    )
                )

        total = len(verification_results)
        verification_rate = (verified_count + partial_count) / total if total > 0 else 0.0

        summary = VerificationSummary(
            total_claims_checked=total,
            verified_count=verified_count,
            partially_supported_count=partial_count,
            conflicting_count=conflict_count,
            unsupported_count=unsupported_count,
            verification_rate=round(verification_rate, 3),
            needs_additional_research=len(additional_requests) > 0,
            additional_requests=additional_requests,
        )

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Fact Checker completed in {duration_ms:.2f}ms | "
            f"verified={verified_count} | partial={partial_count} | "
            f"conflicting={conflict_count} | unsupported={unsupported_count} | "
            f"needs_more_research={summary.needs_additional_research}"
        )

        return FactCheckerResult(
            claims=claims,
            verification_results=verification_results,
            summary=summary,
            additional_requests=additional_requests,
        )


class FactCheckerResult(BaseModel):
    """Complete structured output from the Fact Checker Agent."""

    claims: List[FactualClaim] = Field(default_factory=list)
    verification_results: List[VerificationResult] = Field(default_factory=list)
    summary: VerificationSummary = Field(default_factory=VerificationSummary)
    additional_requests: List[AdditionalResearchRequest] = Field(default_factory=list)


# Global fact checker agent instance
fact_checker_agent = FactCheckerAgent()

