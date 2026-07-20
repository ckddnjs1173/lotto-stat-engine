import streamlit as st

from lotto_engine.config import DEFAULT_CANDIDATE_COUNT
from lotto_engine.loader import LottoDataError
from lotto_engine.recommender import generate_recommendations

st.set_page_config(page_title="Lotto Stat Engine", page_icon="🎱", layout="wide")

st.title("LOTTO STAT ENGINE")
st.caption("워크포워드 구조 검증 기반 로또 6/45 추천 엔진")

st.warning(
    "로또 6/45의 모든 6개 조합은 공정한 독립 시행에서 1등 확률이 동일합니다. "
    "이 앱은 당첨을 보장하지 않으며, 과거 당첨번호의 구조적 특성을 참고한 추천 도구입니다."
)

seed_offset = st.number_input("Seed Offset", min_value=0, max_value=9999, value=0, step=1)
candidate_count = st.number_input("후보 생성 수", min_value=1000, max_value=300000, value=DEFAULT_CANDIDATE_COUNT, step=1000)

if st.button("추천번호 생성", type="primary"):
    with st.spinner("추천번호를 계산 중입니다..."):
        try:
            payload = generate_recommendations(seed_offset=int(seed_offset), candidate_count=int(candidate_count))
        except LottoDataError as exc:
            st.error(str(exc))
            st.stop()

    meta = payload["meta"]
    st.subheader("실행 정보")
    st.write(
        {
            "최신 회차": meta["latest_round"],
            "Seed": meta["seed"],
            "Seed Offset": meta["seed_offset"],
            "후보 생성 수": meta["candidate_count"],
            "필터 통과 수": meta["passed_count"],
            "가중치 방식": meta["weight_mode"],
        }
    )

    st.subheader("추천번호 10게임")
    for idx, item in enumerate(payload["recommendations"], 1):
        nums = "  ".join(f"**{n}**" for n in item["numbers"])
        f = item["features"]
        with st.container(border=True):
            st.markdown(f"### [{idx}] {item['strategy']}")
            st.markdown(nums)
            st.write(f"Score: {item['score']}")
            st.caption(
                f"sum={f['sum']}, odd={f['odd_count']}, even={f['even_count']}, "
                f"gap_std={float(f['gap_std']):.4f}, entropy={float(f['ending_digit_entropy']):.4f}"
            )

    st.subheader("Portfolio Report")
    st.json(payload["portfolio_report"])
else:
    st.info("data/lotto.xlsx 파일을 넣은 뒤 추천번호 생성 버튼을 누르세요.")
