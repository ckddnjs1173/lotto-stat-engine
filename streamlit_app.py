import streamlit as st

from lotto_engine.config import DEFAULT_CANDIDATE_COUNT
from lotto_engine.loader import LottoDataError
from lotto_engine.v27_release import generate_release_recommendations, public_recommendation_payload

st.set_page_config(page_title="Lotto Stat Engine v2.7.1", page_icon="🎲", layout="wide")
st.title("LOTTO STAT ENGINE v2.7.1")
st.caption("Unified Bayesian number + pair evidence · all valid 6/45 combinations remain eligible")
st.warning(
    "표시되는 score는 실제 당첨확률이 아닙니다. 번호와 pair의 Bayesian evidence를 동일한 "
    "fair-null 기준에서 계산하고, strict walk-forward Brier skill로 각 evidence의 영향력을 "
    "자동 축소한 내부 순위 점수입니다."
)

with st.expander("개발용 평가 옵션", expanded=False):
    sampled = st.checkbox("표본 후보만 평가 (개발 smoke 전용)", value=False)
    candidate_count = st.number_input(
        "표본 후보 수",
        min_value=1000,
        max_value=300000,
        value=DEFAULT_CANDIDATE_COUNT,
        step=1000,
        disabled=not sampled,
    )
    seed_offset = st.number_input("Seed Offset", min_value=0, max_value=9999, value=0, step=1)

if st.button("추천번호 생성", type="primary"):
    with st.spinner("최신 데이터의 evidence reliability를 검증하고 후보를 평가하고 있습니다..."):
        try:
            payload = generate_release_recommendations(
                seed_offset=int(seed_offset),
                candidate_count=int(candidate_count),
                exhaustive=not sampled,
            )
        except (LottoDataError, ValueError) as exc:
            st.error(str(exc))
            st.stop()

    public = public_recommendation_payload(payload)
    st.info(public["score_disclaimer"])
    st.write({
        "모델": public["model_version"],
        "상태": public["release_status"],
        "최신 반영 회차": public["latest_draw"],
        "추천 대상 회차": public["target_draw"],
        "평가 방식": public["recommendation_mode"],
        "평가 조합 수": public["evaluated_count"],
        "선정 방식": public["selection_strategy"],
        "패턴 타입 용도": public["pattern_type_role"],
    })

    evidence = public.get("evidence_models", {})
    if evidence:
        st.subheader("Evidence reliability")
        rows = []
        for label in ("number", "pair"):
            detail = evidence.get(label, {})
            if not detail:
                continue
            skills = detail.get("brier_skill_vs_uniform", {})
            rows.append({
                "evidence": label,
                "model": detail.get("spec", {}).get("name"),
                "reliability": detail.get("reliability", 0.0),
                "Brier skill overall": skills.get("overall", 0.0),
                "Brier skill recent300": skills.get("recent300", 0.0),
                "Brier skill recent100": skills.get("recent100", 0.0),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            "reliability가 작을수록 해당 evidence가 최종 랭킹에 미치는 영향도 자동으로 작아집니다. "
            "음의 Brier skill은 반대 예측 신호로 뒤집어 사용하지 않습니다."
        )

    st.subheader("v2.7.1 통합 evidence TOP 10")
    for item in public["recommendations"]:
        with st.container(border=True):
            st.markdown(f"### #{item['rank']} · {item['pattern_type']}")
            st.markdown("  ".join(f"**{number}**" for number in item["numbers"]))
            st.write(f"score: {item['score']:.8f}")
            st.caption(
                f"ranking evidence: {item['ranking_score']:.12f} · "
                f"{item['score_origin']} · pattern type is metadata only"
            )
            st.dataframe(
                [{"component": key, "value": value} for key, value in item["components"].items()],
                use_container_width=True,
                hide_index=True,
            )
else:
    st.info(
        "추천번호 생성 버튼을 누르면 먼저 현재 data/lotto.xlsx로 number/pair walk-forward reliability를 "
        "계산한 뒤, 기본적으로 전체 8,145,060개 조합을 동일한 evidence 식으로 평가합니다."
    )
