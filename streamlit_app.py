import streamlit as st

st.set_page_config(page_title="Lotto Stat Engine", page_icon="🎲")
st.title("Lotto Stat Engine")
st.warning("이 저장소의 Streamlit UI는 현재 사용하지 않습니다.")
st.code(
    "python main.py --scenario full11\n"
    "# or\n"
    "python main.py --scenario clean3\n"
    "# stable alias\n"
    "python scripts\\run_recommend.py --scenario full11",
    language="powershell",
)
st.caption(
    "v3.1 FULL11과 CLEAN3는 현재 모두 실험 시나리오입니다. "
    "어느 것도 승격된 기본 모델이 아니며 실행 시 시나리오를 명시해야 합니다."
)
