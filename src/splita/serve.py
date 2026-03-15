"""REST API wrapper for splita.

Starts a FastAPI server exposing core splita functionality as HTTP
endpoints::

    from splita.serve import serve
    serve(host="0.0.0.0", port=8080)

Requires the ``api`` optional dependency group::

    pip install splita[api]
"""

# NOTE: do NOT use `from __future__ import annotations` here.
# FastAPI needs runtime access to type annotations for request body parsing.

from typing import Any


def _create_app() -> Any:  # pragma: no cover
    """Create and configure the FastAPI application.

    Returns
    -------
    FastAPI
        Configured FastAPI app instance.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise ImportError(
            "FastAPI and Pydantic are required for the REST API.\n"
            "  Hint: install them with `pip install splita[api]`."
        ) from exc

    app = FastAPI(
        title="splita API",
        description="A/B test analysis REST API powered by splita.",
        version="0.1.0",
    )

    # ── Request/Response models ─────────────────────────────────────

    class ExperimentRequest(BaseModel):
        control: list[float] = Field(..., min_length=2, description="Control group observations.")
        treatment: list[float] = Field(
            ..., min_length=2, description="Treatment group observations."
        )
        metric: str = Field(default="auto", description="Metric type.")
        method: str = Field(default="auto", description="Statistical method.")
        alpha: float = Field(default=0.05, gt=0, lt=1, description="Significance level.")
        alternative: str = Field(default="two-sided", description="Test direction.")

    class SampleSizeRequest(BaseModel):
        test_type: str = Field(
            default="proportion",
            description="Type of test: 'proportion' or 'mean'.",
        )
        baseline: float = Field(..., description="Baseline rate or mean.")
        mde: float = Field(..., description="Minimum detectable effect.")
        baseline_std: float | None = Field(
            default=None,
            description="Baseline standard deviation (required for 'mean').",
        )
        alpha: float = Field(default=0.05, gt=0, lt=1)
        power: float = Field(default=0.80, gt=0, lt=1)

    class SRMCheckRequest(BaseModel):
        observed: list[int] = Field(
            ...,
            min_length=2,
            description="Observed counts per variant.",
        )
        expected_fractions: list[float] | None = Field(
            default=None,
            description="Expected fractions per variant.",
        )
        alpha: float = Field(default=0.01, gt=0, lt=1)

    # ── Endpoints ───────────────────────────────────────────────────

    @app.get("/health")
    def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok", "service": "splita"}

    @app.post("/experiment")
    def run_experiment(req: ExperimentRequest) -> dict[str, Any]:
        """Run an A/B test experiment."""
        from splita import Experiment

        try:
            result = Experiment(
                req.control,
                req.treatment,
                metric=req.metric,  # type: ignore[arg-type]
                method=req.method,  # type: ignore[arg-type]
                alpha=req.alpha,
                alternative=req.alternative,  # type: ignore[arg-type]
            ).run()
            return result.to_dict()
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/sample-size")
    def compute_sample_size(req: SampleSizeRequest) -> dict[str, Any]:
        """Compute required sample size."""
        from splita import SampleSize

        try:
            if req.test_type == "proportion":
                result = SampleSize.for_proportion(
                    baseline=req.baseline,
                    mde=req.mde,
                    alpha=req.alpha,
                    power=req.power,
                )
            elif req.test_type == "mean":
                if req.baseline_std is None:
                    raise HTTPException(
                        status_code=422,
                        detail="`baseline_std` is required for test_type='mean'.",
                    )
                result = SampleSize.for_mean(
                    baseline_mean=req.baseline,
                    baseline_std=req.baseline_std,
                    mde=req.mde,
                    alpha=req.alpha,
                    power=req.power,
                )
            else:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unsupported test_type {req.test_type!r}. Use 'proportion' or 'mean'.",
                )
            return result.to_dict()
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/srm-check")
    def run_srm_check(req: SRMCheckRequest) -> dict[str, Any]:
        """Run an SRM check."""
        from splita import SRMCheck

        try:
            result = SRMCheck(
                req.observed,
                expected_fractions=req.expected_fractions,
                alpha=req.alpha,
            ).run()
            return result.to_dict()
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    return app


def serve(host: str = "0.0.0.0", port: int = 8080) -> None:  # pragma: no cover
    """Start a FastAPI server exposing splita as a REST API.

    Parameters
    ----------
    host : str, default "0.0.0.0"
        Host to bind to.
    port : int, default 8080
        Port to listen on.

    Raises
    ------
    ImportError
        If FastAPI or uvicorn is not installed.

    Notes
    -----
    Requires the ``api`` optional dependency group::

        pip install splita[api]
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn is required for the REST API server.\n"
            "  Hint: install it with `pip install splita[api]`."
        ) from exc

    app = _create_app()
    uvicorn.run(app, host=host, port=port)
