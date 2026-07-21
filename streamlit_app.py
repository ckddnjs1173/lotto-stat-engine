import streamlit as st

from lotto_engine.config import DEFAULT_CANDIDATE_COUNT
from lotto_engine.loader import LottoDataError
from lotto_engine.recommender import generate_recommendations

st.set_page_config(page_title="Lotto Forced Prediction Engine", page_icon="🎱", layout="wide")

st.title("LOTTO FORCED PREDICTION ENGINE")
st.caption("과거 당첨번호 구조 + actual-vs-random 검증 기반 예측확률점수 랭킹")

st.warning(
    "로또 6/45의 모든 6개 조합은 공정한 독립 시행에서 1등 확률이 동일합니다. "
    "prediction_score는 실제 당첨확률이 아니라, 과거 당첨번호 구조와 랜덤 기준선 검증으로 만든 내부 예측확률점수입니다."
)

mode = st.radio("추천 평가 방식", ["전체 조합 전수 평가", "랜덤 후보 샘플 평가"], index=0)
exhaustive = mode == "전체 조합 전수 평가"
seed_offset = st.number_input("Seed Offset", min_value=0, max_value=9999, value=0, step=1)
candidate_count = st.number_input(
    "샘플 후보 수",
    min_value=1000,
    max_value=300000,
    value=DEFAULT_CANDIDATE_COUNT,
    step=1000,
    disabled=exhaustive,
)

if st.button("추천번호 생성", type="primary"):
    with st.spinner("추천번호를 계산 중입니다. 전수 평가는 시간이 걸릴 수 있습니다..."):
        try:
            payload = generate_recommendations(
                seed_offset=int(seed_offset),
                candidate_count=int(candidate_count),
                exhaustive=exhaustive,
            )
        except LottoDataError as exc:
            st.error(str(exc))
            st.stop()

    meta = payload["meta"]
    st.subheader("실행 정보")
    st.write(
        {
            "최신 반영 회차": meta["latest_round"],
            "예측 대상 회차": meta["target_round"],
            "평가 방식": meta["recommendation_mode"],
            "평가 조합 수": meta["evaluated_count"],
            "가중치 방식": meta["weight_mode"],
            "점수명": meta["score_name"],
        }
    )

    st.subheader("prediction_score TOP 10")
    for idx, item in enumerate(payload["recommendations"], 1):
        nums = "  ".join(f"**{n}**" for n in item["numbers"])
        f = item["features"]
        with st.container(border=True):
            st.markdown(f"### [{idx}] {item['strategy']}")
            st.markdown(nums)
            st.write(f"prediction_score: {item['prediction_score']}")
            st.caption(
                f"sum={f['sum']}, odd={f['odd_count']}, even={f['even_count']}, "
                f"gap_std={float(f['gap_std']):.4f}, entropy={float(f['ending_digit_entropy']):.4f}, "
                f"section_entropy={float(f['section_entropy']):.4f}, gap_entropy={float(f['gap_entropy']):.4f}"
            )

    st.subheader("Feature Validation")
    wp = payload.get("weight_payload") or {}
    if "percentile_scores" in wp:
        rows = []
        for key, weight in payload["weights"].items():
            rows.append(
                {
                    "feature": key,
                    "percentile": wp["percentile_scores"].get(key),
                    "stability": wp["stability_scores"].get(key),
                    "weight": weight,
                }
            )
        st.dataframe(rows, use_container_width=True)
else:
    st.info("data/lotto.xlsx 파일을 넣은 뒤 추천번호 생성 버튼을 누르세요.")
