import streamlit as st

from lotto_engine.config import DEFAULT_CANDIDATE_COUNT
from lotto_engine.loader import LottoDataError
from lotto_engine.recommender import generate_recommendations

st.set_page_config(page_title="Lotto Stat Engine v2.2", page_icon="🎲", layout="wide")
st.title("LOTTO STAT ENGINE v2.2")
st.caption("Mixed-structure prediction engine")
st.warning(
    "prediction_score는 실제 당첨확률이 아니라 내부 예측확률점수입니다. "
    "보너스 번호는 평가에 사용하지 않으며, 모든 유효 조합은 동일하게 평가 대상입니다."
)

mode = st.radio("평가 방식", ["전체 8,145,060개 조합 전수평가", "후보 표본 평가"], index=0)
exhaustive = mode.startswith("전체")
seed_offset = st.number_input("Seed Offset", min_value=0, max_value=9999, value=0, step=1)
candidate_count = st.number_input(
    "표본 후보 수", min_value=1000, max_value=300000,
    value=DEFAULT_CANDIDATE_COUNT, step=1000, disabled=exhaustive,
)

if st.button("추천번호 생성", type="primary"):
    with st.spinner("조합을 평가하고 있습니다..."):
        try:
            payload = generate_recommendations(
                seed_offset=int(seed_offset),
                candidate_count=int(candidate_count),
                exhaustive=exhaustive,
            )
        except (LottoDataError, ValueError) as exc:
            st.error(str(exc))
            st.stop()

    meta = payload["meta"]
    st.info(meta["score_disclaimer"])
    st.write({
        "최신 반영 회차": meta["latest_round"],
        "예측 대상 회차": meta["target_round"],
        "평가 방식": meta["recommendation_mode"],
        "평가 조합 수": meta["evaluated_count"],
    })
    st.subheader("prediction_score TOP 10")
    for item in payload["recommendations"]:
        with st.container(border=True):
            st.markdown(f"### #{item['rank']} · {item['pattern_type']}")
            st.markdown("  ".join(f"**{number}**" for number in item["numbers"]))
            st.write(f"prediction_score: {item['prediction_score']}")
            st.dataframe(
                [{"component": key, "score": value} for key, value in item["score_breakdown"].items()],
                use_container_width=True,
                hide_index=True,
            )
else:
    st.info("추천번호 생성 버튼을 누르면 평가를 시작합니다.")
