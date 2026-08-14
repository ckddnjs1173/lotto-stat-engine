import streamlit as st

from lotto_engine.config import DEFAULT_CANDIDATE_COUNT
from lotto_engine.loader import LottoDataError
from lotto_engine.v27_release import generate_release_recommendations, public_recommendation_payload

st.set_page_config(page_title="Lotto Stat Engine v2.7", page_icon="🎲", layout="wide")
st.title("LOTTO STAT ENGINE v2.7")
st.caption("Static release-candidate ranking · all valid 6/45 combinations remain eligible")
st.warning(
    "표시되는 score는 실제 당첨확률이 아닙니다. v2.7 검증 결과 Transition/Momentum은 "
    "production 점수에서 제외되었으며, 결과는 정적 통계 순위입니다."
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
    with st.spinner("정적 점수로 조합을 평가하고 있습니다..."):
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
        "최신 반영 회차": public["latest_draw"],
        "추천 대상 회차": public["target_draw"],
        "평가 방식": public["recommendation_mode"],
        "평가 조합 수": public["evaluated_count"],
        "선정 방식": public["selection_strategy"],
    })
    st.subheader("v2.7 정적 점수 TOP 10")
    for item in public["recommendations"]:
        with st.container(border=True):
            st.markdown(f"### #{item['rank']} · {item['pattern_type']}")
            st.markdown("  ".join(f"**{number}**" for number in item["numbers"]))
            st.write(f"score: {item['score']:.4f}")
            st.caption(item["score_origin"])
            st.dataframe(
                [{"component": key, "score": value} for key, value in item["components"].items()],
                use_container_width=True,
                hide_index=True,
            )
else:
    st.info("추천번호 생성 버튼을 누르면 기본적으로 전체 8,145,060개 조합을 평가합니다.")
