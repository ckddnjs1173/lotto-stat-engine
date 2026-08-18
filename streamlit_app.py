import streamlit as st

from lotto_engine.config import DEFAULT_CANDIDATE_COUNT
from lotto_engine.loader import LottoDataError
from lotto_engine.v27_release import generate_release_recommendations, public_recommendation_payload

st.set_page_config(page_title="Lotto Stat Engine v2.7.1", page_icon="🎲", layout="wide")
st.title("LOTTO STAT ENGINE v2.7.1")
st.caption("Unified Bayesian evidence research candidate · all valid 6/45 combinations remain eligible")
st.warning(
    "표시되는 score는 실제 당첨확률이 아닙니다. 번호/페어 Bayesian evidence는 strict walk-forward "
    "검증으로 영향력이 자동 축소되며, Normal/Mixed/Outlier 분류는 순위 계산에 사용되지 않습니다."
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
    with st.spinner("Bayesian evidence로 조합을 평가하고 있습니다..."):
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
    diagnostics = public["evidence_diagnostics"]
    st.write({
        "모델": public["model_version"],
        "상태": public["release_status"],
        "최신 반영 회차": public["latest_draw"],
        "추천 대상 회차": public["target_draw"],
        "평가 방식": public["recommendation_mode"],
        "평가 조합 수": public["evaluated_count"],
        "선정 방식": public["selection_strategy"],
        "번호 reliability": diagnostics["number"]["reliability"],
        "pair reliability": diagnostics["pair"]["reliability"],
    })

    if diagnostics["total_active_reliability"] <= 0:
        st.error(
            "현재 번호와 pair reliability가 모두 0입니다. 아래 TOP-K는 예측 순위가 아니라 "
            "동점 처리 결과이므로 구매용 추천으로 해석하면 안 됩니다. 먼저 combination-ranking "
            "audit 결과를 확인해야 합니다."
        )

    st.subheader("v2.7.1 Evidence TOP 10")
    for item in public["recommendations"]:
        with st.container(border=True):
            st.markdown(f"### #{item['rank']} · {item['pattern_type']}")
            st.markdown("  ".join(f"**{number}**" for number in item["numbers"]))
            st.write(f"score: {item['score']:.8f}")
            st.write(f"ranking evidence: {item['ranking_evidence']:.12f}")
            st.caption(item["score_origin"])
            st.dataframe(
                [{"component": key, "value": value} for key, value in item["components"].items()],
                use_container_width=True,
                hide_index=True,
            )
else:
    st.info("기본 모드는 전체 8,145,060개 조합 평가입니다. 연구 검증 중에는 sampled smoke를 먼저 사용하세요.")
