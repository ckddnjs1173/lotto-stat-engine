import streamlit as st

from lotto_engine.config import DEFAULT_CANDIDATE_COUNT
from lotto_engine.loader import LottoDataError
from lotto_engine.recommender import generate_recommendations

st.set_page_config(
    page_title="Lotto Stat Engine Final",
    page_icon="🎲",
    layout="wide",
)
st.title("LOTTO STAT ENGINE FINAL")
st.caption("Type-Separated Dynamic Portfolio")
st.warning(
    "점수는 실제 당첨확률이 아닙니다. Normal/Mixed/Outlier는 각 타입 내부에서 "
    "순위를 매기고, 최신 실제 전이확률은 10장 타입 예산에 반영됩니다."
)

mode = st.radio("평가 방식", ["전체 8,145,060개 조합 전수평가", "후보 표본 평가"], index=0)
exhaustive = mode.startswith("전체")
seed_offset = st.number_input("Seed Offset", min_value=0, max_value=9999, value=0, step=1)
candidate_count = st.number_input(
    "표본 후보 수",
    min_value=1000,
    max_value=300000,
    value=DEFAULT_CANDIDATE_COUNT,
    step=1000,
    disabled=exhaustive,
)

if st.button("추천번호 생성", type="primary"):
    with st.spinner("조합을 평가하고 있습니다..."):
        try:
            payload = generate_recommendations(
                seed_offset=int(seed_offset),
                candidate_count=int(candidate_count),
                exhaustive=exhaustive,
            )
        except (LottoDataError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
            st.stop()

    meta = payload["meta"]
    st.info(meta["score_disclaimer"])
    st.write({
        "최신 반영 회차": meta["latest_draw"],
        "예측 대상 회차": meta["target_draw"],
        "평가 방식": meta["recommendation_mode"],
        "평가 조합 수": meta["evaluated_count"],
        "타입 배분 목표": meta["portfolio_allocation"],
        "타입 배분 결과": meta["portfolio_selected"],
    })

    diagnostics = meta["mixed_diagnostics"]
    st.subheader("Mixed family diagnostics")
    st.write({
        "family allocation target": diagnostics["family_allocation_target"],
        "family allocation selected": diagnostics["family_allocation_selected"],
        "unique subtype signatures": diagnostics["unique_subtype_signature_count"],
        "average pairwise number overlap": diagnostics["average_pairwise_number_overlap"],
    })

    st.subheader("최종 추천 10조합")
    for item in payload["recommendations"]:
        with st.container(border=True):
            st.markdown(
                f"### #{item['rank']} · {item['pattern_type']} · type rank {item['type_rank']}"
            )
            st.markdown("  ".join(f"**{number}**" for number in item["numbers"]))
            st.write(f"within_type_score: {item['within_type_score']}")
            st.write(f"portfolio_selection_score: {item.get('portfolio_selection_score')}")
            if item["pattern_type"] == "mixed":
                st.write({
                    "mixed base": item["base_score"],
                    "transition lift": item["transition_lift_score"],
                    "momentum lift": item["momentum_lift_score"],
                    "family": item["selected_family"],
                })
            st.dataframe(
                [
                    {"component": key, "score": value}
                    for key, value in item["score_breakdown"].items()
                ],
                use_container_width=True,
                hide_index=True,
            )
else:
    st.info("추천번호 생성 버튼을 누르면 평가를 시작합니다.")
