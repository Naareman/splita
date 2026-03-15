"""Plain-English (and multilingual) interpretation of splita result objects.

This module provides the :func:`explain` function, which converts any
splita result dataclass into a human-readable paragraph with actionable
suggestions.  Supports English (``"en"``), Arabic (``"ar"``), Spanish
(``"es"``), and Chinese (``"zh"``).

Examples
--------
>>> from splita import Experiment, explain
>>> import numpy as np
>>> ctrl = np.random.binomial(1, 0.10, 5000)
>>> trt  = np.random.binomial(1, 0.12, 5000)
>>> result = Experiment(ctrl, trt).run()
>>> print(explain(result))  # doctest: +SKIP
>>> print(explain(result, lang="ar"))  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import fields as dc_fields
from typing import Any

# ─── Multilingual templates ──────────────────────────────────────────

_SUPPORTED_LANGS = {"en", "ar", "es", "zh"}

_TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "increased": "increased",
        "decreased": "decreased",
        "treatment_direction": "Your treatment {direction} the mean by {lift} (95% CI: {ci_lower} to {ci_upper}).",
        "significant": "This is statistically significant at alpha={alpha} (p={pvalue}).",
        "effect_size": "The effect size (Cohen's d = {es}) is {label}.",
        "power_adequate": "Post-hoc power is {power}, suggesting adequate sample size.",
        "power_low": "Post-hoc power is {power}, which is below 0.80 — the experiment may be underpowered.",
        "not_significant": "No statistically significant difference was detected (p={pvalue}).",
        "effect_too_small": "With n={n} per group, the observed effect of {lift} was not large enough to reach significance.",
        "wide_ci": "The confidence interval is wide relative to the observed effect — sample size may be insufficient.",
        "consider_longer": "Consider running longer or using variance reduction (CUPED).",
        "suggestions": "Suggestions:",
        "underpowered": "Experiment appears underpowered. Consider using CUPED for variance reduction or increasing sample size.",
        "wide_ci_suggestion": "Wide confidence interval relative to observed effect — consider increasing sample size.",
        "negligible": "negligible",
        "small": "small",
        "medium": "medium",
        "large": "large",
        # SRM
        "srm_warning": "WARNING: Sample Ratio Mismatch detected (p={pvalue}). The traffic split deviates significantly from expected (variant {variant} is off by {deviation}%). All experiment results should be considered invalid until the cause is identified. Common causes: randomization bugs, bot traffic, tracking errors.",
        "srm_suggestions": "Suggestions:\n  - Check randomization logic and hash function\n  - Filter bot traffic and verify tracking pipeline\n  - Compare pre-experiment vs. in-experiment traffic ratios",
        "srm_passed": "No Sample Ratio Mismatch detected (p={pvalue}). Traffic split is consistent with the expected allocation.",
        # Bayesian
        "prob_better": "There is a {prob} probability that the treatment is better than control.",
        "expected_loss": "The expected loss from choosing the {better} is {loss}.",
        "ship_treatment": "Recommendation: ship the treatment.",
        "keep_control": "Recommendation: keep the control.",
        "not_decisive": "The evidence is not yet decisive. Consider collecting more data.",
        "rope_info": "Probability that the effect is practically negligible (within ROPE [{lo}, {hi}]): {prob}.",
        # Sample size
        "need_users": "You need {n_per_variant} users per variant ({n_total} total) to detect a ",
        "relative_lift": "{rel_mde} relative lift (absolute MDE = {mde}) ",
        "absolute_lift": "{mde} absolute lift ",
        "from_baseline": "from a {baseline} baseline with {power} power at alpha={alpha}.",
        "days_needed": "At the given traffic rate, this will take approximately {days} days.",
    },
    "ar": {
        "increased": "زاد",
        "decreased": "انخفض",
        "treatment_direction": "العلاج {direction} المتوسط بمقدار {lift} (فاصل الثقة 95%: {ci_lower} إلى {ci_upper}).",
        "significant": "هذا ذو دلالة إحصائية عند alpha={alpha} (p={pvalue}).",
        "effect_size": "حجم التأثير (Cohen's d = {es}) هو {label}.",
        "power_adequate": "القوة الإحصائية البعدية هي {power}، مما يشير إلى حجم عينة كافٍ.",
        "power_low": "القوة الإحصائية البعدية هي {power}، وهي أقل من 0.80 — قد تكون التجربة ناقصة القوة.",
        "not_significant": "لم يتم اكتشاف فرق ذو دلالة إحصائية (p={pvalue}).",
        "effect_too_small": "مع n={n} لكل مجموعة، لم يكن التأثير المُلاحظ البالغ {lift} كبيرًا بما يكفي لتحقيق الدلالة.",
        "wide_ci": "فاصل الثقة واسع نسبةً إلى التأثير المُلاحظ — قد يكون حجم العينة غير كافٍ.",
        "consider_longer": "فكّر في تشغيل التجربة لفترة أطول أو استخدام تقليل التباين (CUPED).",
        "suggestions": "اقتراحات:",
        "underpowered": "التجربة تبدو ناقصة القوة. فكّر في استخدام CUPED لتقليل التباين أو زيادة حجم العينة.",
        "wide_ci_suggestion": "فاصل الثقة واسع نسبةً إلى التأثير المُلاحظ — فكّر في زيادة حجم العينة.",
        "negligible": "مهمَل",
        "small": "صغير",
        "medium": "متوسط",
        "large": "كبير",
        "srm_warning": "تحذير: تم اكتشاف عدم تطابق نسبة العينة (p={pvalue}). توزيع الزيارات ينحرف بشكل ملحوظ عن المتوقع (المتغير {variant} ينحرف بمقدار {deviation}%). يجب اعتبار جميع نتائج التجربة غير صالحة حتى يتم تحديد السبب.",
        "srm_suggestions": "اقتراحات:\n  - تحقق من منطق التوزيع العشوائي\n  - قم بتصفية حركة الروبوتات\n  - قارن نسب الزيارات قبل وأثناء التجربة",
        "srm_passed": "لم يتم اكتشاف عدم تطابق نسبة العينة (p={pvalue}). توزيع الزيارات متسق مع التوزيع المتوقع.",
        "prob_better": "هناك احتمال {prob} أن العلاج أفضل من المجموعة الضابطة.",
        "expected_loss": "الخسارة المتوقعة من اختيار {better} هي {loss}.",
        "ship_treatment": "التوصية: أطلق العلاج.",
        "keep_control": "التوصية: أبقِ على المجموعة الضابطة.",
        "not_decisive": "الأدلة ليست حاسمة بعد. فكّر في جمع المزيد من البيانات.",
        "rope_info": "احتمال أن التأثير مهمَل عمليًا (ضمن ROPE [{lo}, {hi}]): {prob}.",
        "need_users": "تحتاج إلى {n_per_variant} مستخدم لكل متغير ({n_total} إجمالًا) لاكتشاف ",
        "relative_lift": "تغير نسبي {rel_mde} (MDE مطلق = {mde}) ",
        "absolute_lift": "تغير مطلق {mde} ",
        "from_baseline": "من خط أساس {baseline} بقوة {power} عند alpha={alpha}.",
        "days_needed": "بمعدل الزيارات الحالي، سيستغرق ذلك حوالي {days} يومًا.",
    },
    "es": {
        "increased": "aumentó",
        "decreased": "disminuyó",
        "treatment_direction": "Su tratamiento {direction} la media en {lift} (IC 95%: {ci_lower} a {ci_upper}).",
        "significant": "Esto es estadísticamente significativo con alpha={alpha} (p={pvalue}).",
        "effect_size": "El tamaño del efecto (d de Cohen = {es}) es {label}.",
        "power_adequate": "La potencia post-hoc es {power}, lo que sugiere un tamaño de muestra adecuado.",
        "power_low": "La potencia post-hoc es {power}, que está por debajo de 0.80 — el experimento puede tener potencia insuficiente.",
        "not_significant": "No se detectó una diferencia estadísticamente significativa (p={pvalue}).",
        "effect_too_small": "Con n={n} por grupo, el efecto observado de {lift} no fue lo suficientemente grande para alcanzar significancia.",
        "wide_ci": "El intervalo de confianza es amplio en relación con el efecto observado — el tamaño de muestra puede ser insuficiente.",
        "consider_longer": "Considere ejecutar más tiempo o usar reducción de varianza (CUPED).",
        "suggestions": "Sugerencias:",
        "underpowered": "El experimento parece tener potencia insuficiente. Considere usar CUPED para reducción de varianza o aumentar el tamaño de muestra.",
        "wide_ci_suggestion": "Intervalo de confianza amplio en relación con el efecto observado — considere aumentar el tamaño de muestra.",
        "negligible": "insignificante",
        "small": "pequeño",
        "medium": "mediano",
        "large": "grande",
        "srm_warning": "ADVERTENCIA: Se detectó una discrepancia en la proporción de la muestra (p={pvalue}). La distribución del tráfico se desvía significativamente de lo esperado (variante {variant} se desvía en {deviation}%). Todos los resultados deben considerarse inválidos hasta identificar la causa.",
        "srm_suggestions": "Sugerencias:\n  - Verifique la lógica de aleatorización\n  - Filtre el tráfico de bots\n  - Compare las proporciones de tráfico antes y durante el experimento",
        "srm_passed": "No se detectó discrepancia en la proporción de la muestra (p={pvalue}). La distribución del tráfico es consistente con la asignación esperada.",
        "prob_better": "Hay una probabilidad de {prob} de que el tratamiento sea mejor que el control.",
        "expected_loss": "La pérdida esperada de elegir el {better} es {loss}.",
        "ship_treatment": "Recomendación: lanzar el tratamiento.",
        "keep_control": "Recomendación: mantener el control.",
        "not_decisive": "La evidencia aún no es decisiva. Considere recopilar más datos.",
        "rope_info": "Probabilidad de que el efecto sea prácticamente insignificante (dentro de ROPE [{lo}, {hi}]): {prob}.",
        "need_users": "Necesita {n_per_variant} usuarios por variante ({n_total} en total) para detectar ",
        "relative_lift": "un cambio relativo de {rel_mde} (MDE absoluto = {mde}) ",
        "absolute_lift": "un cambio absoluto de {mde} ",
        "from_baseline": "desde una línea base de {baseline} con potencia {power} a alpha={alpha}.",
        "days_needed": "A la tasa de tráfico actual, esto tomará aproximadamente {days} días.",
    },
    "zh": {
        "increased": "增加了",
        "decreased": "减少了",
        "treatment_direction": "您的实验组{direction}均值 {lift}（95% CI: {ci_lower} 到 {ci_upper}）。",
        "significant": "在 alpha={alpha} 水平下具有统计显著性（p={pvalue}）。",
        "effect_size": "效应量（Cohen's d = {es}）为{label}。",
        "power_adequate": "事后统计效力为 {power}，表明样本量充足。",
        "power_low": "事后统计效力为 {power}，低于 0.80 — 实验可能效力不足。",
        "not_significant": "未检测到统计显著差异（p={pvalue}）。",
        "effect_too_small": "每组 n={n} 的情况下，观察到的效应 {lift} 不足以达到显著性。",
        "wide_ci": "置信区间相对于观察效应过宽 — 样本量可能不足。",
        "consider_longer": "建议延长实验时间或使用方差缩减（CUPED）。",
        "suggestions": "建议：",
        "underpowered": "实验似乎效力不足。建议使用 CUPED 进行方差缩减或增加样本量。",
        "wide_ci_suggestion": "置信区间相对于观察效应过宽 — 建议增加样本量。",
        "negligible": "可忽略的",
        "small": "小的",
        "medium": "中等的",
        "large": "大的",
        "srm_warning": "警告：检测到样本比例不匹配（p={pvalue}）。流量分配与预期显著偏离（变体 {variant} 偏离 {deviation}%）。在确定原因之前，所有实验结果应视为无效。",
        "srm_suggestions": "建议：\n  - 检查随机化逻辑和哈希函数\n  - 过滤机器人流量并验证追踪管道\n  - 比较实验前后的流量比例",
        "srm_passed": "未检测到样本比例不匹配（p={pvalue}）。流量分配与预期一致。",
        "prob_better": "实验组优于对照组的概率为 {prob}。",
        "expected_loss": "选择{better}的预期损失为 {loss}。",
        "ship_treatment": "建议：发布实验组。",
        "keep_control": "建议：保留对照组。",
        "not_decisive": "证据尚不具有决定性。建议收集更多数据。",
        "rope_info": "效应在实际上可忽略范围内（ROPE [{lo}, {hi}]）的概率：{prob}。",
        "need_users": "您需要每组 {n_per_variant} 个用户（共 {n_total} 个）来检测",
        "relative_lift": "相对变化 {rel_mde}（绝对 MDE = {mde}）",
        "absolute_lift": "绝对变化 {mde}",
        "from_baseline": "（基准 {baseline}，效力 {power}，alpha={alpha}）。",
        "days_needed": "按当前流量速率，大约需要 {days} 天。",
    },
}


def _t(lang: str, key: str) -> str:
    """Look up a translated template string."""
    return _TRANSLATIONS[lang][key]


def _effect_label(effect_size: float, lang: str = "en") -> str:
    """Classify Cohen's d / h magnitude."""
    d = abs(effect_size)
    if d < 0.2:
        return _t(lang, "negligible")
    if d < 0.5:  # pragma: no cover
        return _t(lang, "small")
    if d < 0.8:  # pragma: no cover
        return _t(lang, "medium")
    return _t(lang, "large")  # pragma: no cover


def _fmt_pct(val: float) -> str:
    return f"{val * 100:.2f}%"


def _fmt_num(val: float) -> str:
    if abs(val) < 0.0001 and val != 0.0:
        return f"{val:.2e}"
    return f"{val:.2f}"


def _explain_experiment(result: Any, lang: str = "en") -> str:
    """Interpret an ExperimentResult."""
    lines: list[str] = []

    direction = _t(lang, "increased") if result.lift > 0 else _t(lang, "decreased")
    lines.append(
        _t(lang, "treatment_direction").format(
            direction=direction,
            lift=_fmt_num(result.lift),
            ci_lower=_fmt_num(result.ci_lower),
            ci_upper=_fmt_num(result.ci_upper),
        )
    )

    if result.significant:
        lines.append(
            _t(lang, "significant").format(alpha=result.alpha, pvalue=_fmt_num(result.pvalue))
        )
        label = _effect_label(result.effect_size, lang)
        lines.append(_t(lang, "effect_size").format(es=_fmt_num(result.effect_size), label=label))
        if result.power >= 0.8:
            lines.append(_t(lang, "power_adequate").format(power=_fmt_num(result.power)))
        else:
            lines.append(_t(lang, "power_low").format(power=_fmt_num(result.power)))
    else:
        lines.append(_t(lang, "not_significant").format(pvalue=_fmt_num(result.pvalue)))
        n = min(result.control_n, result.treatment_n)
        lines.append(_t(lang, "effect_too_small").format(n=n, lift=_fmt_num(result.lift)))
        ci_width = result.ci_upper - result.ci_lower
        if ci_width > abs(result.lift) * 4 and result.lift != 0:
            lines.append(_t(lang, "wide_ci"))
        lines.append(_t(lang, "consider_longer"))

    # Suggestions
    suggestions = _experiment_suggestions(result, lang)
    if suggestions:
        lines.append("")
        lines.append(_t(lang, "suggestions"))
        for s in suggestions:
            lines.append(f"  - {s}")

    return " ".join(lines[: _first_blank(lines)]) + (
        "\n" + "\n".join(lines[_first_blank(lines) :]) if _first_blank(lines) < len(lines) else ""
    )


def _first_blank(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if line == "":
            return i
    return len(lines)


def _experiment_suggestions(result: Any, lang: str = "en") -> list[str]:
    suggestions: list[str] = []
    if result.power < 0.8:
        suggestions.append(_t(lang, "underpowered"))
    if not result.significant:
        ci_width = result.ci_upper - result.ci_lower
        if ci_width > 0 and abs(result.lift) > 0 and ci_width > abs(result.lift) * 4:
            suggestions.append(_t(lang, "wide_ci_suggestion"))
    return suggestions


def _explain_srm(result: Any, lang: str = "en") -> str:
    """Interpret an SRMResult."""
    if not result.passed:
        worst_dev = result.deviations_pct[result.worst_variant]
        warning = _t(lang, "srm_warning").format(
            pvalue=_fmt_num(result.pvalue),
            variant=result.worst_variant,
            deviation=f"{worst_dev:+.1f}",
        )
        suggestions = _t(lang, "srm_suggestions")
        return f"{warning}\n\n{suggestions}"
    return _t(lang, "srm_passed").format(pvalue=_fmt_num(result.pvalue))


def _explain_bayesian(result: Any, lang: str = "en") -> str:
    """Interpret a BayesianResult."""
    prob_pct = _fmt_pct(result.prob_b_beats_a)

    if result.prob_b_beats_a > 0.5:
        better = "treatment"
        loss = result.expected_loss_b
    else:
        better = "control"
        loss = result.expected_loss_a

    loss_str = _fmt_pct(loss) if abs(loss) < 1 else _fmt_num(loss)

    lines: list[str] = [_t(lang, "prob_better").format(prob=prob_pct)]

    lines.append(_t(lang, "expected_loss").format(better=better, loss=loss_str))

    if result.prob_b_beats_a >= 0.95:
        lines.append(_t(lang, "ship_treatment"))
    elif result.prob_b_beats_a <= 0.05:
        lines.append(_t(lang, "keep_control"))
    else:
        lines.append(_t(lang, "not_decisive"))

    if result.prob_in_rope is not None:
        lines.append(
            _t(lang, "rope_info").format(
                lo=_fmt_num(result.rope[0]),
                hi=_fmt_num(result.rope[1]),
                prob=_fmt_pct(result.prob_in_rope),
            )
        )

    return " ".join(lines)


def _explain_sample_size(result: Any, lang: str = "en") -> str:
    """Interpret a SampleSizeResult."""
    lines: list[str] = [
        _t(lang, "need_users").format(
            n_per_variant=f"{result.n_per_variant:,}",
            n_total=f"{result.n_total:,}",
        )
    ]

    if result.relative_mde is not None:
        lines[0] += _t(lang, "relative_lift").format(
            rel_mde=_fmt_pct(result.relative_mde),
            mde=_fmt_num(result.mde),
        )
    else:  # pragma: no cover
        lines[0] += _t(lang, "absolute_lift").format(mde=_fmt_num(result.mde))

    lines[0] += _t(lang, "from_baseline").format(
        baseline=_fmt_num(result.baseline),
        power=_fmt_pct(result.power),
        alpha=result.alpha,
    )

    if result.days_needed is not None:
        lines.append(_t(lang, "days_needed").format(days=result.days_needed))

    return " ".join(lines)


# ─── New explainer functions for all result types ────────────────────


def _explain_bandit(result: Any, lang: str = "en") -> str:
    """Interpret a BanditResult."""
    total_pulls = sum(result.n_pulls_per_arm)
    best = result.current_best_arm
    prob = result.prob_best[best]
    loss = result.expected_loss[best]
    n_arms = len(result.n_pulls_per_arm)
    stop_text = (
        f"The stopping criterion has been met -- you can confidently pick arm {best}."
        if result.should_stop
        else "The stopping criterion has NOT been met -- continue collecting data before deciding."
    )
    # Arm summary
    arm_summaries = []
    for i in range(n_arms):
        mean_str = _fmt_num(result.arm_means[i]) if result.arm_means else "?"
        arm_summaries.append(f"arm {i}: mean={mean_str}, pulls={result.n_pulls_per_arm[i]}")
    arm_detail = "; ".join(arm_summaries)
    return (
        f"After {total_pulls} total pulls across {n_arms} arms, arm {best} is the current "
        f"best with {prob:.0%} probability of being optimal. "
        f"Expected loss from choosing arm {best}: {loss:.4f}. "
        f"{stop_text} "
        f"Arm summary: {arm_detail}."
    )


def _explain_correction(result: Any, lang: str = "en") -> str:
    """Interpret a CorrectionResult."""
    sig_labels = []
    nonsig_labels = []
    for i, rej in enumerate(result.rejected):
        label = result.labels[i] if result.labels else f"test_{i}"
        if rej:
            sig_labels.append(f"{label} (adj. p={_fmt_num(result.adjusted_pvalues[i])})")
        else:
            nonsig_labels.append(f"{label} (adj. p={_fmt_num(result.adjusted_pvalues[i])})")
    sig_str = ", ".join(sig_labels) if sig_labels else "none"
    nonsig_str = ", ".join(nonsig_labels) if nonsig_labels else "none"
    lost = result.n_tests - result.n_rejected
    return (
        f"Multiple testing correction ({result.method}) was applied to {result.n_tests} metrics "
        f"at alpha = {_fmt_num(result.alpha)}. "
        f"{result.n_rejected} of {result.n_tests} survived correction. "
        f"Significant after correction: {sig_str}. "
        f"Not significant after correction: {nonsig_str}. "
        + (
            f"{lost} metric(s) that might have appeared significant without correction "
            f"did not survive, protecting you from false discoveries. "
            if lost > 0
            else ""
        )
        + "Next step: focus on the metrics that survived correction for your ship decision."
    )


def _explain_msprt_state(result: Any, lang: str = "en") -> str:
    """Interpret an mSPRTState."""
    n = result.n_control + result.n_treatment
    effect = result.current_effect_estimate
    if result.should_stop:
        action = (
            f"STOP the experiment. A significant effect has been detected "
            f"(estimated effect = {_fmt_num(effect)}). You can make a decision now."
        )
    else:
        action = (
            f"CONTINUE the experiment. No significant effect detected yet "
            f"(current estimate = {_fmt_num(effect)}). Keep collecting data."
        )
    return (
        f"Sequential test (mSPRT) after {n} observations "
        f"({result.n_control} control, {result.n_treatment} treatment). "
        f"Always-valid p-value: {result.always_valid_pvalue:.4f}, "
        f"CI: [{_fmt_num(result.always_valid_ci_lower)}, {_fmt_num(result.always_valid_ci_upper)}]. "
        f"Recommendation: {action}"
    )


def _explain_msprt_result(result: Any, lang: str = "en") -> str:
    """Interpret an mSPRTResult."""
    effect = result.current_effect_estimate
    reason_map = {
        "boundary_crossed": "the effect crossed the significance boundary",
        "truncation": "the maximum sample size was reached",
        "not_stopped": "neither boundary has been crossed yet",
    }
    reason_text = reason_map.get(result.stopping_reason, result.stopping_reason)
    if result.should_stop:
        action = f"STOP the experiment ({reason_text}). Estimated effect: {_fmt_num(effect)}."
    else:  # pragma: no cover
        action = f"CONTINUE the experiment ({reason_text}). Current estimate: {_fmt_num(effect)}."
    return (
        f"Sequential test (mSPRT) final result after {result.total_observations} observations "
        f"({result.n_control} control, {result.n_treatment} treatment). "
        f"Always-valid p-value: {result.always_valid_pvalue:.4f}, "
        f"CI: [{_fmt_num(result.always_valid_ci_lower)}, {_fmt_num(result.always_valid_ci_upper)}]. "
        f"Recommendation: {action}"
    )


def _explain_quantile(result: Any, lang: str = "en") -> str:
    """Interpret a QuantileResult."""
    sig_quantiles = [f"{result.quantiles[i]:.0%}" for i, s in enumerate(result.significant) if s]
    n_tested = len(result.quantiles)
    if sig_quantiles:
        q_str = ", ".join(sig_quantiles)
        # Build details about each significant quantile
        details = []
        for i, s in enumerate(result.significant):
            if s:
                q = result.quantiles[i]
                diff = result.differences[i]
                direction = "higher" if diff > 0 else "lower"
                details.append(f"{q:.0%} quantile is {_fmt_num(abs(diff))} {direction}")
        details_str = "; ".join(details)
        median_idx = len(result.quantiles) // 2
        median_diff = result.differences[median_idx]
        return (
            f"Quantile test compared {n_tested} quantiles between treatment and control. "
            f"Treatment differs significantly at: {q_str}. "
            f"Specifically: {details_str}. "
            f"The median difference is {_fmt_num(median_diff)}. "
            f"This means the treatment effect is not uniform across the distribution "
            f"-- it affects some parts of the distribution more than others. "
            f"Next step: examine which quantiles changed to understand if the effect "
            f"is concentrated in the tails or the center of the distribution."
        )
    return (
        f"Quantile test compared {n_tested} quantiles between treatment and control. "
        f"No significant differences were found at any tested quantile. "
        f"This suggests the treatment had no meaningful effect on the distribution shape. "
        f"Next step: consider increasing sample size or checking if the effect is too small to detect."
    )


def _explain_cluster(result: Any, lang: str = "en") -> str:
    """Interpret a ClusterResult."""
    de = 1.0 / (1.0 - result.icc) if result.icc < 1.0 else float("inf")
    sig_text = (
        "statistically significant" if result.significant else "not statistically significant"
    )
    n_clusters = result.n_clusters_control + result.n_clusters_treatment
    return (
        f"Cluster-robust analysis tested the treatment effect while accounting for "
        f"within-cluster correlation (ICC = {_fmt_num(result.icc)}). "
        f"The treatment effect (lift = {_fmt_num(result.lift)}) is {sig_text} "
        f"(p = {_fmt_num(result.pvalue)}, CI: [{_fmt_num(result.ci_lower)}, {_fmt_num(result.ci_upper)}]). "
        f"The design effect is {de:.1f}, meaning standard errors are {de:.1f}x wider than a naive "
        f"analysis that ignores clustering would produce. "
        f"In practice, ignoring the clustering would have made the test {de:.1f}x too optimistic "
        f"about statistical significance. "
        f"With {n_clusters} total clusters ({result.n_clusters_control} control, "
        f"{result.n_clusters_treatment} treatment), power depends primarily on the number of clusters, "
        f"not the number of individual observations."
    )


def _explain_stratified(result: Any, lang: str = "en") -> str:
    """Interpret a StratifiedResult."""
    sig_text = (
        "statistically significant" if result.significant else "not statistically significant"
    )
    return (
        f"Stratified analysis partitioned the data into {result.n_strata} strata and estimated "
        f"a weighted average treatment effect across them. "
        f"The overall ATE is {_fmt_num(result.ate)} (p = {_fmt_num(result.pvalue)}, "
        f"CI: [{_fmt_num(result.ci_lower)}, {_fmt_num(result.ci_upper)}]), which is {sig_text}. "
        f"Stratification reduces variance by removing between-stratum variation, "
        f"giving you a more precise estimate than a pooled analysis. "
        f"This is equivalent to running a longer experiment without stratification. "
        f"Next step: check individual stratum effects to see if the treatment works "
        f"differently across segments."
    )


def _explain_hte(result: Any, lang: str = "en") -> str:
    """Interpret an HTEResult."""
    heterogeneity = "substantial" if result.cate_std > abs(result.mean_cate) * 0.5 else "modest"
    method_name = result.method.replace("_", "-")
    top_str = ""
    if result.top_features is not None and len(result.top_features) > 0:  # pragma: no cover
        top_str = (
            f" The most important features driving heterogeneity are at indices "
            f"{result.top_features[:5]}."
        )
    return (
        f"Heterogeneous treatment effect analysis ({method_name}) found that the treatment "
        f"effect varies across individuals. The average effect (CATE) is {_fmt_num(result.mean_cate)}, "
        f"but individual effects have a standard deviation of {_fmt_num(result.cate_std)}, "
        f"indicating {heterogeneity} heterogeneity. "
        f"This means the treatment does not affect everyone equally -- some users benefit "
        f"more than others, and some may even be harmed.{top_str} "
        f"Next step: segment users by their predicted CATE to identify who benefits most, "
        f"and consider targeting the treatment to high-CATE users."
    )


def _explain_triggered(result: Any, lang: str = "en") -> str:
    """Interpret a TriggeredResult."""
    rate = (result.trigger_rate_control + result.trigger_rate_treatment) / 2
    return (
        f"ITT effect: {_fmt_num(result.itt_result.lift)} "
        f"(p={_fmt_num(result.itt_result.pvalue)}). "
        f"Per-protocol effect: {_fmt_num(result.per_protocol_result.lift)} "
        f"(p={_fmt_num(result.per_protocol_result.pvalue)}). "
        f"Trigger rate: {rate:.0%}."
    )


def _explain_interaction(result: Any, lang: str = "en") -> str:
    """Interpret an InteractionResult."""
    n_segments = len(result.segment_results)
    if result.has_interaction:
        return (
            f"Interaction test across {n_segments} segments found that the treatment effect "
            f"significantly differs across segments (interaction p = {_fmt_num(result.interaction_pvalue)}). "
            f"The segment with the strongest effect is '{result.strongest_segment}'. "
            f"This means a one-size-fits-all launch may not be optimal -- some segments "
            f"respond much better to the treatment than others. "
            f"Next step: consider a targeted rollout to the highest-performing segments, "
            f"or investigate why some segments respond differently."
        )
    return (
        f"Interaction test across {n_segments} segments found that the treatment effect "
        f"does not significantly differ across segments (interaction p = {_fmt_num(result.interaction_pvalue)}). "
        f"The treatment appears to have a consistent effect regardless of segment. "
        f"This is good news for a broad rollout -- you can ship the treatment to all users "
        f"without worrying about negative effects in specific segments."
    )


def _explain_multi_objective(result: Any, lang: str = "en") -> str:
    """Interpret a MultiObjectiveResult."""
    n = len(result.metric_results)
    n_sig = sum(1 for p in result.corrected_pvalues if p < 0.05)
    return (
        f"Recommendation: {result.recommendation}. "
        f"{n_sig} of {n} metrics are significant after correction."
    )


def _explain_did(result: Any, lang: str = "en") -> str:
    """Interpret a DiDResult."""
    sig_text = (
        "statistically significant" if result.significant else "not statistically significant"
    )
    if result.parallel_trends_pvalue > 0.05:
        trend_text = (
            f"The parallel trends assumption holds (p = {_fmt_num(result.parallel_trends_pvalue)}), "
            f"meaning the control group is a valid counterfactual for what would have happened "
            f"to the treatment group without the intervention."
        )
    else:
        trend_text = (
            f"WARNING: The parallel trends assumption fails (p = {_fmt_num(result.parallel_trends_pvalue)}, "
            f"pre-trend difference = {_fmt_num(result.pre_trend_diff)}). "
            f"The control group may not be a valid counterfactual. "
            f"The DiD estimate may be biased. Consider synthetic control or matching methods."
        )
    return (
        f"Difference-in-differences estimated the average treatment effect on the treated (ATT) "
        f"as {_fmt_num(result.att)} (p = {_fmt_num(result.pvalue)}, "
        f"CI: [{_fmt_num(result.ci_lower)}, {_fmt_num(result.ci_upper)}]), which is {sig_text}. "
        f"{trend_text} "
        f"Next step: plot the pre- and post-treatment trends for both groups to visually "
        f"verify the parallel trends assumption."
    )


def _explain_synthetic_control(result: Any, lang: str = "en") -> str:
    """Interpret a SyntheticControlResult."""
    return (
        f"Synthetic control effect: {_fmt_num(result.effect)}. "
        f"Pre-treatment fit: RMSE = {_fmt_num(result.pre_treatment_rmse)}."
    )


def _explain_meta_analysis(result: Any, lang: str = "en") -> str:
    """Interpret a MetaAnalysisResult."""
    # Classify heterogeneity
    if result.i_squared < 0.25:  # pragma: no cover
        het_label = "low"
        het_advice = "The studies are measuring a consistent effect."
    elif result.i_squared < 0.75:
        het_label = "moderate"
        het_advice = (
            "There is moderate variability across studies, which may reflect "
            "genuine differences in populations, interventions, or contexts."
        )
    else:  # pragma: no cover
        het_label = "high"
        het_advice = (
            "The studies show substantially different effects. The combined estimate "
            "may mask important differences. Investigate study-level characteristics "
            "to understand what drives the variation."
        )
    sig_text = (
        "statistically significant" if result.pvalue < 0.05 else "not statistically significant"
    )
    return (
        f"Meta-analysis combined {len(result.study_weights)} studies using the "
        f"{result.method}-effects model. The pooled effect is {_fmt_num(result.combined_effect)} "
        f"(p = {_fmt_num(result.pvalue)}, CI: [{_fmt_num(result.ci_lower)}, {_fmt_num(result.ci_upper)}]), "
        f"which is {sig_text}. "
        f"Heterogeneity is {het_label} (I-squared = {result.i_squared:.0%}, "
        f"Cochran's Q p = {_fmt_num(result.heterogeneity_pvalue)}). {het_advice} "
        f"Next step: if heterogeneity is high, consider subgroup analysis or "
        f"meta-regression to identify moderating factors."
    )


def _explain_auto(result: Any, lang: str = "en") -> str:
    """Interpret an AutoResult."""
    return "\n".join(result.reasoning)


def _explain_check(result: Any, lang: str = "en") -> str:
    """Interpret a CheckResult."""
    status = "passed" if result.all_passed else "failed"
    n_passed = sum(1 for c in result.checks if c.get("passed", False))
    n_total = len(result.checks)
    recs = " ".join(result.recommendations) if result.recommendations else ""
    base = f"Pre-analysis check: {status}. {n_passed}/{n_total} checks passed."
    if recs:
        base += f" {recs}"
    return base


def _explain_monitor(result: Any, lang: str = "en") -> str:
    """Interpret a MonitorResult."""
    days_str = (
        f"{result.days_remaining} days remaining"
        if result.days_remaining is not None
        else "days remaining unknown"
    )
    return (
        f"Current status: {result.recommendation}. "
        f"Lift = {_fmt_num(result.current_lift)}, "
        f"p = {_fmt_num(result.current_pvalue)}. "
        f"{days_str}."
    )


def _explain_whatif(result: Any, lang: str = "en") -> str:
    """Interpret a WhatIfResult."""
    return result.message


def _explain_power_simulation(result: Any, lang: str = "en") -> str:
    """Interpret a PowerSimulationResult."""
    return f"Estimated power: {result.power:.0%} at n={result.n_per_variant} per variant."


def _explain_simulation(result: Any, lang: str = "en") -> str:
    """Interpret a SimulationResult."""
    return result.recommendation


def _explain_comparison(result: Any, lang: str = "en") -> str:
    """Interpret a ComparisonResult."""
    diff_word = "significantly different" if result.significant else "not significantly different"
    return (
        f"The two effects are {diff_word} "
        f"(p={_fmt_num(result.pvalue)}). "
        f"Direction: {result.direction}."
    )


def _explain_survival(result: Any, lang: str = "en") -> str:
    """Interpret a SurvivalResult."""
    ctrl_str = (
        _fmt_num(result.median_survival_ctrl)
        if result.median_survival_ctrl is not None
        else "not reached"
    )
    trt_str = (
        _fmt_num(result.median_survival_trt)
        if result.median_survival_trt is not None
        else "not reached"
    )
    hr = result.hazard_ratio
    if hr < 1:
        hr_meaning = (
            f"a hazard ratio of {_fmt_num(hr)} means the treatment group experiences events "
            f"at {(1 - hr) * 100:.0f}% lower rate than control -- the treatment is protective"
        )
    elif hr > 1:  # pragma: no cover
        hr_meaning = (
            f"a hazard ratio of {_fmt_num(hr)} means the treatment group experiences events "
            f"at {(hr - 1) * 100:.0f}% higher rate than control -- the treatment increases risk"
        )
    else:
        hr_meaning = "a hazard ratio of 1.00 means no difference in event rates"  # pragma: no cover
    sig_text = (
        "statistically significant" if result.significant else "not statistically significant"
    )
    return (
        f"Survival analysis compared time-to-event outcomes between groups. "
        f"The result is {sig_text} (log-rank p = {_fmt_num(result.logrank_pvalue)}, "
        f"CI: [{_fmt_num(result.ci_lower)}, {_fmt_num(result.ci_upper)}]). "
        f"In plain terms, {hr_meaning}. "
        f"Median survival time: control = {ctrl_str}, treatment = {trt_str}. "
        f"Next step: examine Kaplan-Meier curves to understand when the survival "
        f"curves diverge and whether the effect is early or late."
    )


def _explain_diagnosis(result: Any, lang: str = "en") -> str:  # pragma: no cover
    """Interpret a DiagnosisResult."""
    items = "; ".join(result.action_items) if result.action_items else "none"
    return (
        f"Diagnosis: status={result.status}, "
        f"confidence={result.confidence_level}. "
        f"Action items: {items}."
    )


def _explain_recommendation(result: Any, lang: str = "en") -> str:
    """Interpret a RecommendationResult."""
    return "\n".join(result.reasoning)  # pragma: no cover


def _explain_generic(result: Any, lang: str = "en") -> str:
    """Generate a generic explanation from dataclass fields.

    This is the fallback for any result type without a dedicated handler.
    """
    type_name = type(result).__name__
    parts = [f"{type_name}:"]
    try:
        for f in dc_fields(result):
            val = getattr(result, f.name)
            if isinstance(val, float):  # pragma: no cover
                parts.append(f"{f.name}={_fmt_num(val)}")
            elif isinstance(val, (bool, int, str)) or (isinstance(val, list) and len(val) <= 5):
                parts.append(f"{f.name}={val}")
            elif isinstance(val, list):  # pragma: no cover
                parts.append(f"{f.name}=[{len(val)} items]")
            # Skip complex nested objects
    except TypeError:
        parts.append("(could not introspect fields)")
    return " ".join(parts)


# ─── Registry of explainable types ──────────────────────────────────

_EXPLAINERS: dict[str, Any] = {
    "ExperimentResult": _explain_experiment,
    "SRMResult": _explain_srm,
    "BayesianResult": _explain_bayesian,
    "SampleSizeResult": _explain_sample_size,
    "BanditResult": _explain_bandit,
    "CorrectionResult": _explain_correction,
    "mSPRTState": _explain_msprt_state,
    "mSPRTResult": _explain_msprt_result,
    "QuantileResult": _explain_quantile,
    "ClusterResult": _explain_cluster,
    "StratifiedResult": _explain_stratified,
    "HTEResult": _explain_hte,
    "TriggeredResult": _explain_triggered,
    "InteractionResult": _explain_interaction,
    "MultiObjectiveResult": _explain_multi_objective,
    "DiDResult": _explain_did,
    "SyntheticControlResult": _explain_synthetic_control,
    "MetaAnalysisResult": _explain_meta_analysis,
    "AutoResult": _explain_auto,
    "CheckResult": _explain_check,
    "MonitorResult": _explain_monitor,
    "WhatIfResult": _explain_whatif,
    "PowerSimulationResult": _explain_power_simulation,
    "SimulationResult": _explain_simulation,
    "ComparisonResult": _explain_comparison,
    "SurvivalResult": _explain_survival,
    "DiagnosisResult": _explain_diagnosis,
    "RecommendationResult": _explain_recommendation,
}


def explain(result: Any, *, lang: str = "en") -> str:
    """Return an interpretation of any splita result.

    Parameters
    ----------
    result : dataclass
        A splita result object.  All result types defined in
        ``splita._types`` are supported.  Types with dedicated handlers
        produce rich, contextual explanations; all others produce a
        generic field-based summary.
    lang : str, default ``"en"``
        Language code.  Supported: ``"en"`` (English), ``"ar"`` (Arabic),
        ``"es"`` (Spanish), ``"zh"`` (Chinese).  Advanced result types
        (beyond ExperimentResult, SRMResult, BayesianResult,
        SampleSizeResult) use English regardless of the ``lang``
        parameter.

    Returns
    -------
    str
        Human-readable interpretation with actionable suggestions.

    Raises
    ------
    ValueError
        If ``lang`` is not one of the supported languages.

    Examples
    --------
    >>> from splita._types import ExperimentResult
    >>> r = ExperimentResult(
    ...     control_mean=0.10, treatment_mean=0.12,
    ...     lift=0.02, relative_lift=0.2, pvalue=0.003,
    ...     statistic=2.1, ci_lower=0.007, ci_upper=0.033,
    ...     significant=True, alpha=0.05, method="ztest",
    ...     metric="conversion", control_n=5000,
    ...     treatment_n=5000, power=0.82, effect_size=0.15,
    ... )
    >>> from splita.explain import explain
    >>> text = explain(r)
    >>> "significant" in text
    True
    >>> text_ar = explain(r, lang="ar")
    >>> "دلالة إحصائية" in text_ar
    True
    """
    if lang not in _SUPPORTED_LANGS:
        raise ValueError(
            f"`lang` must be one of {sorted(_SUPPORTED_LANGS)}, got {lang!r}.\n"
            f"  Hint: use 'en', 'ar', 'es', or 'zh'."
        )
    type_name = type(result).__name__
    explainer = _EXPLAINERS.get(type_name)
    if explainer is not None:
        return explainer(result, lang)
    # Fallback: generic explanation from dataclass fields
    return _explain_generic(result, lang)
