import streamlit as st
import time, json, random, os
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# OpenAI 클라이언트 초기화
@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ .env 파일에 OPENAI_API_KEY를 설정해주세요!")
        st.stop()
    return OpenAI(api_key=api_key)

client = get_openai_client()

# ---------------------------
# 0) 유틸
# ---------------------------

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_event(kind, payload=None):
    st.session_state["events"].append({"t": now(), "kind": kind, "payload": payload or {}})


def init_state():
    for k, v in {
        "book_title": "",
        "book_title_input": "",
        "book_text": "",
        "outline": {"intro": "", "body": "", "concl": ""},
        "outline_intro": "",
        "outline_body": "",
        "outline_concl": "",
        "draft": "",
        "suggestions": [],
        "events": [],
        "n_sugs": 3,
        "enable_hints": True,
        "enable_questions": True,
        "role": "학생",
        "selected_book": "",
        "ai_suggestions_cache": {},
        "current_questions": [],
        "use_chat_mode": True,  # 초안 입력은 Enter 전송 채팅 방식
        # 각 영역 직접 편집 토글
        "edit_book_text": False,
        "edit_outline_intro": False,
        "edit_outline_body": False,
        "edit_outline_concl": False,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------
# OpenAI API 함수들 (temperature 제거)
# ---------------------------

def call_openai_api(messages, max_tokens=500):
    """OpenAI API 호출 함수 (gpt-5는 temperature 커스터마이즈 미지원)"""
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"AI API 오류: {str(e)}")
        return None


def generate_ai_suggestions(context, block, n=3):
    """AI를 활용한 작문 제안 생성 (Enter 기반 입력 흐름과 무관)"""

    key_fields = (
        context.get("book_title", ""),
        context.get("book_text", "")[:500],
        json.dumps(context.get("outline", {}), ensure_ascii=False),
        context.get("draft", "")[:300],
        block,
        n,
    )
    cache_key = f"{hash(key_fields)}"
    if cache_key in st.session_state["ai_suggestions_cache"]:
        return st.session_state["ai_suggestions_cache"][cache_key]

    block_prompts = {
        "intro": "독서감상문의 서론 부분으로, 책을 읽게 된 계기나 첫인상에 대한",
        "body": "독서감상문의 본론 부분으로, 인상 깊은 장면과 느낀 점에 대한",
        "concl": "독서감상문의 결론 부분으로, 배운 점과 추천 이유에 대한",
    }

    prompt = f"""
당신은 초등학교 4-6학년 학생들의 독서감상문 작성을 도와주는 AI 교사입니다.

학생 정보:
- 책 제목: {context.get('book_title', '미정')}
- 책 내용: {context.get('book_text', '내용 없음')[:500]}
- 현재 개요: {context.get('outline', {})}
- 현재까지 작성한 글: {context.get('draft', '없음')[:300]}

요청: {block_prompts[block]} 문장 제안 {n}개를 만들어주세요.

조건:
1. 초등학생 수준의 쉽고 자연스러운 표현 사용
2. 각 제안은 한 문장으로 완성된 형태
3. 구체적이고 실용적인 내용
4. 학생이 선택해서 바로 사용하거나 수정할 수 있는 형태
5. "나는", "내가" 등 1인칭 시점 사용
6. 번호나 특수문자 없이 문장만 제시

예시 형식:
이 책을 읽게 된 이유는 표지가 예뻐서 호기심이 생겼기 때문이다.
주인공이 어려움을 이겨내는 모습을 보며 나도 용기를 얻었다.

제안 {n}개:
"""

    messages = [{"role": "user", "content": prompt}]

    with st.spinner("🤖 AI가 제안을 만들고 있어요..."):
        response = call_openai_api(messages, max_tokens=400)

    if response:
        suggestions = []
        for line in response.strip().split("\n"):
            line = line.strip().lstrip("0123456789.- ")
            if line and len(line) > 10:
                suggestions.append(line)
        suggestions = suggestions[:n]
        st.session_state["ai_suggestions_cache"][cache_key] = suggestions
        log_event("ai_suggestions_generated", {"block": block, "count": len(suggestions)})
        return suggestions

    return get_fallback_suggestions(block, n)


def get_fallback_suggestions(block, n=3):
    fallback = {
        "intro": [
            "이 책을 읽게 된 이유는 친구가 재미있다고 추천해주었기 때문입니다.",
            "처음 이 책의 제목을 봤을 때 어떤 내용일지 궁금했습니다.",
            "도서관에서 우연히 발견한 이 책이 생각보다 흥미로워 보였습니다.",
        ],
        "body": [
            "가장 인상 깊었던 장면은 주인공이 어려움을 극복하는 부분이었습니다.",
            "등장인물들의 우정을 보며 진정한 친구의 의미를 생각해보게 되었습니다.",
            "만약 내가 주인공이라면 다른 선택을 했을 것 같다는 생각이 들었습니다.",
        ],
        "concl": [
            "이 책을 통해 포기하지 않는 것의 중요함을 배웠습니다.",
            "친구들에게도 꼭 추천하고 싶은 좋은 책이었습니다.",
            "앞으로는 이 책에서 배운 교훈을 실천해보고 싶습니다.",
        ],
    }
    return random.sample(fallback[block], min(n, len(fallback[block])))


def generate_guiding_questions(context):
    prompt = f"""
초등학생이 독서감상문을 쓸 때 도움이 되는 생각 유도 질문을 3개 만들어주세요.

현재 상황:
- 책: {context.get('book_title', '미정')}
- 작성중인 내용: {context.get('draft', '없음')[:200]}

조건:
1. 초등학생이 이해하기 쉬운 질문
2. 깊이 있는 사고를 유도하는 질문
3. 구체적이고 실용적인 질문
4. "왜", "어떻게", "만약" 등을 활용한 열린 질문

각 질문은 한 줄로 작성해주세요.
"""

    messages = [{"role": "user", "content": prompt}]
    response = call_openai_api(messages, max_tokens=300)

    if response:
        questions = [q.strip().lstrip("0123456789.- ") for q in response.split("\n") if q.strip()]
        return questions[:3]

    return [
        "주인공의 행동에 대해 어떻게 생각하나요?",
        "이 책에서 가장 중요한 메시지는 무엇인가요?",
        "친구에게 이 책을 어떻게 소개하고 싶나요?",
    ]


def check_spelling_and_grammar(text):
    if not text.strip():
        return []

    prompt = f"""
다음 초등학생이 쓴 독서감상문에서 맞춤법이나 어색한 표현을 찾아서 간단히 알려주세요.

글: {text[:1000]}

조건:
1. 초등학생 수준에서 쉽게 이해할 수 있는 설명
2. 너무 많은 지적보다는 주요한 2-3개만
3. 격려하는 톤으로 설명
4. 형식: "원래 표현 → 고친 표현 (이유)"

피드백:
"""

    messages = [{"role": "user", "content": prompt}]
    response = call_openai_api(messages, max_tokens=300)

    if response:
        return response.strip().split("\n")
    return []


# ---------------------------
# 추천 도서 데이터
# ---------------------------
RECOMMENDED_BOOKS = {
    "소리없는 아이들 - 황선미": {
        "summary": "특별한 아이들의 소통과 이해에 관한 이야기",
        "key_scenes": [
            "주인공이 처음 특별한 친구를 만나는 장면",
            "서로 다른 소통 방식을 이해하게 되는 순간",
            "편견을 극복하고 진정한 우정을 나누는 결말",
        ],
    },
    "나와 조금 다를 뿐이야 - 이금이": {
        "summary": "다름을 인정하고 받아들이는 성장 이야기",
        "key_scenes": [
            "주인공이 자신과 다른 친구를 처음 만나는 장면",
            "차이점 때문에 생기는 갈등과 오해",
            "서로의 다름을 이해하고 받아들이는 화해",
        ],
    },
    "여름과 가을 사이 - 박슬기": {
        "summary": "계절의 변화처럼 성장하는 아이의 마음",
        "key_scenes": [
            "여름 방학 동안 겪은 특별한 경험",
            "새 학기를 앞두고 느끼는 복잡한 감정",
            "성장을 받아들이며 새로운 시작을 준비하는 모습",
        ],
    },
    "인어 소녀 - 차율이": {
        "summary": "꿈과 현실 사이에서 고민하는 소녀의 이야기",
        "key_scenes": [
            "주인공이 자신만의 특별한 꿈을 갖게 되는 순간",
            "꿈을 이루기 위해 노력하면서 겪는 어려움",
            "꿈을 향한 의지를 다지며 성장하는 결말",
        ],
    },
}


# ---------------------------
# 공통 컴포넌트: Enter로 전송되는 입력 폼
# ---------------------------

def enter_append_form(label: str, value_key: str, preview_height: int = 120, edit_toggle_key: str | None = None):
    """
    - 현재 값 미리보기 (기본 읽기 전용)
    - 한 줄 입력 + Enter로 전송(Form): 입력 즉시 누적
    - 필요 시 '직접 편집' 토글로 text_area 활성화
    """
    st.markdown(f"**{label}**")

    # 직접 편집 토글
    editable = False
    if edit_toggle_key:
        editable = st.checkbox("✎ 직접 편집(텍스트 영역)", key=edit_toggle_key)

    if editable:
        st.text_area("", key=value_key, height=preview_height)
    else:
        st.text_area("", value=st.session_state.get(value_key, ""), height=preview_height, disabled=True)

        # Enter 전송 폼
        with st.form(key=f"{value_key}_form", clear_on_submit=True):
            new_line = st.text_input("Enter로 추가 (한 줄)", key=f"{value_key}_input", placeholder="여기에 입력하고 Enter")
            submitted = st.form_submit_button("추가")
            if submitted and new_line and new_line.strip():
                prev = st.session_state.get(value_key, "").strip()
                st.session_state[value_key] = (prev + ("\n" if prev else "") + new_line.strip())
                log_event("enter_append", {"field": value_key, "chars": len(new_line.strip())})
                st.rerun()


# ---------------------------
# 3) 사이드바 (책 제목도 Enter로 반영)
# ---------------------------

def render_sidebar():
    st.sidebar.header("⚙️ 설정")

    # API 상태 확인
    api_status = "🟢 연결됨" if os.getenv("OPENAI_API_KEY") else "🔴 미연결"
    st.sidebar.caption(f"AI 상태: {api_status}")

    st.session_state["role"] = st.sidebar.selectbox("사용자 모드", ["학생", "교사"], index=0)
    st.session_state["n_sugs"] = st.sidebar.slider("AI 제안 개수", 1, 5, 3)
    st.session_state["enable_hints"] = st.sidebar.checkbox("맞춤법 검사", True)
    st.session_state["enable_questions"] = st.sidebar.checkbox("AI 유도 질문", True)

    st.sidebar.divider()
    st.sidebar.subheader("📚 책 선택")

    # 추천 도서 선택 기능
    book_options = ["직접 입력"] + list(RECOMMENDED_BOOKS.keys())
    selected_book = st.sidebar.selectbox("책 선택", book_options)

    if selected_book != "직접 입력":
        st.session_state["book_title"] = selected_book
        book_info = RECOMMENDED_BOOKS[selected_book]
        st.session_state["book_text"] = (
            f"{book_info['summary']}\n\n주요 장면들:\n" + "\n".join([f"- {scene}" for scene in book_info["key_scenes"]])
        )
        st.sidebar.success(f"'{selected_book}' 정보가 로드되었습니다!")
    else:
        # 책 제목: Enter로 반영되는 Form
        with st.sidebar.form("book_title_form", clear_on_submit=False):
            st.session_state["book_title_input"] = st.text_input(
                "책 제목 (Enter로 반영)", st.session_state.get("book_title", "")
            )
            submitted = st.form_submit_button("적용")
            if submitted:
                st.session_state["book_title"] = st.session_state["book_title_input"].strip()

        uploaded = st.sidebar.file_uploader("책 요약/중요 부분(.txt)", type=["txt"])
        if uploaded:
            try:
                st.session_state["book_text"] = uploaded.read().decode("utf-8")
            except Exception:
                st.session_state["book_text"] = uploaded.read().decode("utf-8", errors="ignore")
            log_event("book_uploaded", {"name": uploaded.name, "chars": len(st.session_state["book_text"])})

    st.sidebar.divider()
    if st.sidebar.button("📊 활동 로그 준비"):
        data = {
            "book_title": st.session_state["book_title"],
            "outline": st.session_state["outline"],
            "draft": st.session_state["draft"],
            "events": st.session_state["events"],
        }
        st.sidebar.download_button(
            "events.json 저장",
            data=json.dumps(data, ensure_ascii=False, indent=2),
            file_name=f"독서감상문_로그_{now().replace(':', '-').replace(' ', '_')}.json",
        )


# ---------------------------
# 4) 본문 UI
# ---------------------------

def render_quality_panel():
    if st.session_state["enable_hints"] and st.session_state["draft"].strip():
        st.markdown("**✍️ AI 맞춤법 및 표현 도움**")
        if st.button("🔍 맞춤법 검사하기"):
            with st.spinner("AI가 글을 검토하고 있어요..."):
                feedback = check_spelling_and_grammar(st.session_state["draft"])
                if feedback:
                    for tip in feedback:
                        if tip.strip():
                            st.info(tip.strip())
                else:
                    st.success("훌륭해요! 특별한 문제점이 없어 보입니다.")

    if st.session_state["enable_questions"]:
        st.markdown("**🤔 AI가 제안하는 생각 질문**")
        if st.button("💡 새로운 질문 받기"):
            context = {"book_title": st.session_state["book_title"], "draft": st.session_state["draft"]}
            with st.spinner("AI가 질문을 만들고 있어요..."):
                questions = generate_guiding_questions(context)
                st.session_state["current_questions"] = questions
        if st.session_state.get("current_questions"):
            for i, q in enumerate(st.session_state["current_questions"], 1):
                st.write(f"{i}. {q}")


def render_outline():
    st.subheader("📖 Step 1. 책 내용 확인 및 정리 (Enter로 추가)")

    if st.session_state["book_title"]:
        st.success(f"선택된 책: **{st.session_state['book_title']}**")

    # 책 요약: Enter로 줄 단위 추가 + 필요 시 직접 편집
    enter_append_form("책 요약/중요 부분", "book_text", preview_height=140, edit_toggle_key="edit_book_text")

    st.subheader("📝 Step 2. 글의 뼈대 만들기 (Enter로 추가)")
    cols = st.columns(3)

    with cols[0]:
        enter_append_form("서론 (책 소개/읽게 된 이유)", "outline_intro", preview_height=140, edit_toggle_key="edit_outline_intro")

    with cols[1]:
        enter_append_form("본론 (인상 깊은 장면/느낀 점)", "outline_body", preview_height=140, edit_toggle_key="edit_outline_body")

    with cols[2]:
        enter_append_form("결론 (배운 점/추천 이유)", "outline_concl", preview_height=140, edit_toggle_key="edit_outline_concl")

    # outline dict 동기화
    st.session_state["outline"]["intro"] = st.session_state.get("outline_intro", "")
    st.session_state["outline"]["body"] = st.session_state.get("outline_body", "")
