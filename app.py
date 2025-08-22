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
        "book_text": "",
        "outline": {"intro": "", "body": "", "concl": ""},
        "draft": "",
        "suggestions": [],
        "events": [],
        "n_sugs": 3,
        "enable_hints": True,
        "enable_questions": True,
        "role": "학생",
        "selected_book": "",
        "ai_suggestions_cache": {},  # AI 제안 캐시
        "current_questions": [],
        "use_chat_mode": True,  # ← Enter로 전송되는 채팅 입력 모드 기본값
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
    """AI를 활용한 작문 제안 생성"""

    # 캐시 키 생성 (필요 필드만 묶어서 안정적인 키 생성)
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

    # 블록별 프롬프트 설정
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
        # 응답을 줄 단위로 분리하고 정리
        suggestions = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # 번호나 특수문자 제거
            line = line.lstrip("0123456789.- ")
            if line and len(line) > 10:  # 너무 짧은 건 제외
                suggestions.append(line)

        # 요청된 개수만큼만 반환
        suggestions = suggestions[:n]

        # 캐시에 저장
        st.session_state["ai_suggestions_cache"][cache_key] = suggestions

        log_event("ai_suggestions_generated", {"block": block, "count": len(suggestions)})
        return suggestions

    # API 실패 시 기본 제안 반환
    return get_fallback_suggestions(block, n)


def get_fallback_suggestions(block, n=3):
    """AI API 실패 시 기본 제안"""
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
    """상황에 맞는 유도 질문 생성"""
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

    # 기본 질문 반환
    return [
        "주인공의 행동에 대해 어떻게 생각하나요?",
        "이 책에서 가장 중요한 메시지는 무엇인가요?",
        "친구에게 이 책을 어떻게 소개하고 싶나요?",
    ]


def check_spelling_and_grammar(text):
    """맞춤법 및 문법 검사 (간단한 피드백)"""
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
# 워드 파일에서 언급된 추천 도서들
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
# 3) UI 개선
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
            f"{book_info['summary']}\n\n주요 장면들:\n"
            + "\n".join([f"- {scene}" for scene in book_info["key_scenes"]])
        )
        st.sidebar.success(f"'{selected_book}' 정보가 로드되었습니다!")
    else:
        st.session_state["book_title"] = st.sidebar.text_input("책 제목", st.session_state["book_title"])
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
            context = {
                "book_title": st.session_state["book_title"],
                "draft": st.session_state["draft"],
            }
            with st.spinner("AI가 질문을 만들고 있어요..."):
                questions = generate_guiding_questions(context)
                st.session_state["current_questions"] = questions

        # 현재 질문들 표시
        if st.session_state.get("current_questions"):
            for i, q in enumerate(st.session_state["current_questions"], 1):
                st.write(f"{i}. {q}")


def render_outline():
    st.subheader("📖 Step 1. 책 내용 확인 및 정리")

    if st.session_state["book_title"]:
        st.success(f"선택된 책: **{st.session_state['book_title']}**")

    st.text_area(
        "책 요약/중요 부분",
        key="book_text",
        height=120,
        placeholder="책에서 중요한 장면이나 문장을 입력해주세요. 위에서 추천 도서를 선택하면 자동으로 채워집니다.",
    )

    st.subheader("📝 Step 2. 글의 뼈대 만들기")
    cols = st.columns(3)

    with cols[0]:
        st.markdown("**서론 (책 소개/읽게 된 이유)**")
        st.text_area("", key="outline_intro", height=120, placeholder="• 책을 읽게 된 계기\n• 첫인상이나 기대\n• 간단한 책 소개")

    with cols[1]:
        st.markdown("**본론 (인상 깊은 장면/느낀 점)**")
        st.text_area("", key="outline_body", height=120, placeholder="• 가장 기억에 남는 장면\n• 그 이유와 느낀 점\n• 나의 경험과 연결")

    with cols[2]:
        st.markdown("**결론 (배운 점/추천 이유)**")
        st.text_area("", key="outline_concl", height=120, placeholder="• 책에서 배운 것\n• 추천하고 싶은 이유\n• 앞으로의 다짐")

    # session dict 동기화
    st.session_state["outline"]["intro"] = st.session_state.get("outline_intro", "")
    st.session_state["outline"]["body"] = st.session_state.get("outline_body", "")
    st.session_state["outline"]["concl"] = st.session_state.get("outline_concl", "")


def render_suggestion_block(label, key_block, icon):
    st.markdown(f"### {icon} {label} 작성 도움")

    if st.button(f"🤖 AI {label} 제안 받기", key=f"generate_{key_block}"):
        ctx = {
            "book_title": st.session_state["book_title"],
            "book_text": st.session_state["book_text"],  # ← 키 통일
            "outline": st.session_state["outline"],
            "draft": st.session_state["draft"],
        }
        suggestions = generate_ai_suggestions(ctx, key_block, st.session_state["n_sugs"])
        st.session_state[f"suggestions_{key_block}"] = suggestions

    # 각 블록별로 별도의 suggestions 저장
    current_suggestions = st.session_state.get(f"suggestions_{key_block}", [])

    if current_suggestions:
        st.markdown("**🎯 AI가 만든 작문 제안들**")

        for i, sug in enumerate(current_suggestions):
            with st.container(border=True):
                st.write(f"**제안 {i+1}:** {sug}")
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    if st.button(f"✅ 선택", key=f"accept_{key_block}_{i}"):
                        current_draft = st.session_state["draft"].strip()
                        addition = f"\n\n{sug}" if current_draft else sug
                        st.session_state["draft"] = (current_draft + addition) if current_draft else addition
                        log_event("ai_suggestion_accepted", {"block": key_block, "text": sug})
                        st.success("본문에 추가했어요!")
                        time.sleep(0.5)
                        st.rerun()

                with c2:
                    if st.button(f"❌ 패스", key=f"reject_{key_block}_{i}"):
                        log_event("ai_suggestion_rejected", {"block": key_block, "text": sug})
                        st.info("다른 제안을 확인해보세요!")


# 핵심 변경: Enter로 전송되는 채팅 입력 모드 추가

def render_editor():
    st.subheader("✏️ Step 3. 초안 작성 및 다듬기")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.toggle("Enter로 전송(채팅 입력 모드)", key="use_chat_mode", value=st.session_state.get("use_chat_mode", True), help="켜면 Enter로 전송되고, 줄바꿈은 Shift+Enter입니다. 끄면 기존 텍스트 영역에서 편집합니다.")

        if st.session_state["use_chat_mode"]:
            # 읽기 전용 초안 뷰 + 채팅 입력
            st.text_area("현재 초안 (읽기 전용)", value=st.session_state["draft"], key="draft_view", height=300, disabled=True)

            # chat_input은 Enter로 전송, Shift+Enter로 줄바꿈
            new_line = st.chat_input("여기에 입력하고 Enter를 눌러 추가 (Shift+Enter 줄바꿈)")
            if new_line is not None and new_line.strip() != "":
                if st.session_state["draft"].strip():
                    st.session_state["draft"] += "\n" + new_line
                else:
                    st.session_state["draft"] = new_line
                log_event("draft_appended", {"source": "chat_input", "chars": len(new_line)})
                st.rerun()
        else:
            # 기존 방식: 자유롭게 편집 가능한 텍스트 영역
            st.text_area(
                "내가 쓰고 있는 독서감상문",
                key="draft",
                height=300,
                placeholder="위에서 AI 제안을 선택하거나 직접 작성해보세요.\n\n💡 팁: AI 제안은 시작점일 뿐이에요. 여러분의 생각과 경험을 더해서 자신만의 독서감상문을 만들어보세요!",
            )

    with col2:
        render_quality_panel()

    st.subheader("💾 Step 4. 완성된 글 저장하기")

    if st.session_state["draft"].strip():
        word_count = len(st.session_state["draft"])  # 단순 글자 수
        st.info(f"현재 글자 수: {word_count}자")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📄 텍스트 파일로 저장",
                data=st.session_state["draft"],
                file_name=f"{st.session_state['book_title']}_독서감상문.txt" if st.session_state["book_title"] else "독서감상문.txt",
            )
        with c2:
            if st.button("🎉 작성 완료!"):
                log_event("writing_completed", {"word_count": word_count, "book": st.session_state["book_title"]})
                st.balloons()
                st.success("독서감상문 작성을 완료했습니다! 수고하셨어요!")
    else:
        st.warning("아직 작성된 내용이 없습니다. 위에서 AI 제안을 선택하거나 직접 작성해보세요.")


# ---------------------------
# 4) 메인 함수
# ---------------------------

def main():
    st.set_page_config(
        page_title="AI 독서감상문 작문 도우미",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    render_sidebar()

    st.title("📚 AI와 함께 쓰는 독서감상문")
    st.caption("🤖 AI 제안 → 🤔 선택/수정 → ✨ 완성! AI 선생님과 함께 특별한 독서감상문을 만들어보세요.")

    # 진행 상황 표시
    steps_completed = [
        bool(st.session_state["book_text"].strip()),
        any(st.session_state["outline"].values()),
        bool(st.session_state["draft"].strip()),
    ]
    progress = sum(steps_completed) / 3
    st.progress(progress)
    st.caption(f"진행 상황: {int(progress * 100)}% 완료")

    render_outline()

    st.divider()
    st.markdown("## 🎯 AI와 함께하는 단계별 작문")

    c1, c2, c3 = st.columns(3)
    with c1:
        render_suggestion_block("서론", "intro", "🌟")
    with c2:
        render_suggestion_block("본론", "body", "💭")
    with c3:
        render_suggestion_block("결론", "concl", "🎭")

    st.divider()
    render_editor()


if __name__ == "__main__":
    main()
