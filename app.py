# app.py
import streamlit as st
import time, json, random, os, hashlib
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# 0) 부팅 & 공용 유틸
# =========================
load_dotenv()

@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ .env 파일에 OPENAI_API_KEY를 설정해주세요!")
        st.stop()
    return OpenAI(api_key=api_key)

client = get_openai_client()

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_event(kind, payload=None):
    st.session_state["events"].append({
        "t": now(),
        "kind": kind,
        "payload": payload or {}
    })

def init_state():
    defaults = {
        "book_title": "",
        "book_text": "",
        "outline": {"intro": "", "body": "", "concl": ""},
        "draft": "",
        "events": [],
        "n_sugs": 3,
        "enable_hints": True,
        "enable_questions": True,
        "role": "학생",
        "ai_suggestions_cache": {},
        "current_questions": [],
        "use_chat_mode": False,          # ✅ 자유 편집 기본
        "selected_book_prev": "",
        "book_index": "",                # LLM 요약 원문(텍스트)
        "book_index_json": {},           # 파싱된 요약/장면/키워드
        "focus_kw": "",                  # 선택된 키워드
        "saved_versions": [],
        "model_name": "gpt-4o",           # 기본 모델명
        "spelling_feedback": [],         # 맞춤법/표현 피드백 보존
        "question_history": [],          # 중복 질문 방지 히스토리
        "question_nonce": 0              # 다양화 토큰
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# --- Draft 안전 추가용 헬퍼 (충돌 방지) ---
def _apply_draft_queue():
    """이번 런 시작 전에 큐에 쌓인 추가 문장을 draft에 합친다."""
    queued = st.session_state.pop("_draft_append_queue", [])
    if queued:
        cur = st.session_state.get("draft", "")
        for part in queued:
            if not part:
                continue
            cur = (cur + ("\n\n" if cur.strip() else "") + part).strip()
        st.session_state["draft"] = cur

def _queue_append(text: str):
    """바로 session_state['draft']를 건드리지 말고 큐에 넣은 뒤 rerun."""
    if not text:
        return
    q = st.session_state.get("_draft_append_queue", [])
    q.append(text)
    st.session_state["_draft_append_queue"] = q
    st.rerun()

# =========================
# 1) OpenAI 래퍼
# =========================
def call_openai_api(messages, max_tokens=500, model=None):
    """Chat Completions 호출 (모델 gpt-4o-mini로 고정, SDK 파라미터 호환)."""
    use_model = "gpt-4o-mini"  # ✅ 고정
    try:
        # 일반 SDK 파라미터
        resp = client.chat.completions.create(
            model=use_model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except TypeError:
        # 일부 환경은 max_completion_tokens만 허용
        try:
            resp = client.chat.completions.create(
                model=use_model,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e2:
            st.error(f"AI API 오류(compat): {e2}")
            return None
    except Exception as e:
        st.error(f"AI API 오류: {e}")
        return None


# =========================
# 2) 책 인덱싱(요약/장면/키워드)
# =========================
def index_book_text():
    txt = st.session_state.get("book_text", "").strip()
    if not txt:
        st.warning("인덱싱할 책 내용이 없습니다.")
        return

    prompt = f"""
너는 초등학생 독서감상문 도우미다. 아래 책 내용을 읽고 JSON으로만 답하라.
형식:
{{
  "summary": ["문장1","문장2","문장3"],     // 3문장 요약
  "key_scenes": ["장면1","장면2","장면3"],  // 3개
  "keywords": ["키워드1","키워드2","키워드3","키워드4","키워드5"] // 5개
}}
원문(일부 또는 전체):
{txt[:4000]}
"""
    res = call_openai_api([{"role": "user", "content": prompt}], max_tokens=600)
    if not res:
        st.warning("책 인덱싱에 실패했습니다.")
        return

    st.session_state["book_index"] = res

    # JSON 파싱 시도
    try:
        data = json.loads(res)
        st.session_state["book_index_json"] = {
            "summary": data.get("summary", []) if isinstance(data.get("summary", []), list) else [],
            "key_scenes": data.get("key_scenes", []) if isinstance(data.get("key_scenes", []), list) else [],
            "keywords": data.get("keywords", []) if isinstance(data.get("keywords", []), list) else [],
        }
    except Exception:
        # 파싱 실패 시 텍스트만 유지
        st.session_state["book_index_json"] = {}
    log_event("book_indexed", {
        "summary_len": len(st.session_state.get("book_index_json", {}).get("summary", [])),
        "keywords_len": len(st.session_state.get("book_index_json", {}).get("keywords", []))
    })
    st.success("📚 책 인덱싱(요약/장면/키워드)을 완료했습니다.")

# =========================
# 3) 제안/질문/맞춤법 등
# =========================
def _stable_cache_key(parts: list) -> str:
    s = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def generate_ai_suggestions(context, block, n=3):
    """AI를 활용한 작문 제안 생성 (서론/본론/결론)"""
    key_fields = [
        context.get("book_title", ""),
        context.get("book_text", "")[:800],
        json.dumps(context.get("outline", {}), ensure_ascii=False, sort_keys=True),
        context.get("draft", "")[-500:],   # 최근 문맥 반영
        block,
        n,
        st.session_state.get("focus_kw", ""),
        st.session_state.get("book_index", "")[:1200]
    ]
    cache_key = _stable_cache_key(key_fields)
    cache = st.session_state["ai_suggestions_cache"]
    if cache_key in cache:
        return cache[cache_key]

    block_prompts = {
        "intro": "독서감상문의 서론 부분으로, 책을 읽게 된 계기나 첫인상에 대한",
        "body":  "독서감상문의 본론 부분으로, 인상 깊은 장면과 느낀 점에 대한",
        "concl": "독서감상문의 결론 부분으로, 배운 점과 추천 이유에 대한",
    }
    focus_kw = st.session_state.get("focus_kw", "")
    book_index = st.session_state.get("book_index", "")

    prompt = f"""
당신은 초등학교 4-6학년 학생의 독서감상문 작성을 돕는 AI 교사입니다.
학생 정보:
- 책 제목: {context.get('book_title', '미정')}
- 책 내용(요약/장면/키워드): {book_index[:1200] or '(없음)'}
- 현재 개요: {json.dumps(context.get('outline', {}), ensure_ascii=False)}
- 현재 초안(마지막 500자): {context.get('draft', '')[-500:] or '(없음)'}
- 선택된 키워드: {focus_kw or '(없음)'}

요청: {block_prompts[block]} 문장 제안을 {n}개 만들어주세요.

조건:
1) 초등학생 수준의 쉽고 자연스러운 표현
2) 각 제안은 한 문장으로 완성
3) 구체적이고 실용적인 내용
4) 학생이 선택·수정하여 바로 사용할 수 있는 형태
5) 1인칭 시점("나는", "내가")
6) 번호나 특수문자 없이 문장만 제시

예시: 이 책을 읽게 된 이유는 표지가 예뻐서 호기심이 생겼기 때문이다.

제안 {n}개:
"""
    with st.spinner("🤖 AI가 제안을 만들고 있어요..."):
        response = call_openai_api([{"role": "user", "content": prompt}], max_tokens=400)
    if response:
        lines = []
        for line in response.strip().split("\n"):
            s = line.strip().lstrip("0123456789.- ").strip()
            if s and len(s) > 8:
                lines.append(s)
        lines = lines[:n]
        cache[cache_key] = lines
        log_event("ai_suggestions_generated", {"block": block, "count": len(lines)})
        return lines

    return get_fallback_suggestions(block, n)

def get_fallback_suggestions(block, n=3):
    fallback = {
        "intro": [
            "이 책을 읽게 된 이유는 친구가 재미있다고 추천했기 때문이다.",
            "처음 제목을 보았을 때 어떤 이야기일지 궁금했다.",
            "도서관에서 우연히 이 책을 발견하고 기대가 생겼다.",
        ],
        "body": [
            "가장 인상 깊었던 장면은 주인공이 어려움을 극복하는 부분이었다.",
            "등장인물들의 우정을 보며 진정한 친구의 의미를 생각하게 되었다.",
            "만약 내가 주인공이었다면 어떤 선택을 했을지 상상해 보았다.",
        ],
        "concl": [
            "이 책을 통해 포기하지 않는 태도의 중요함을 배웠다.",
            "친구들에게도 꼭 추천하고 싶은 책이라고 느꼈다.",
            "앞으로 이 책에서 배운 교훈을 생활에서 실천해 보고 싶다.",
        ],
    }
    return random.sample(fallback[block], min(n, len(fallback[block])))

def generate_guiding_questions(context):
    """
    상황에 맞는 유도 질문 3개 (중복 회피 + 매번 다르게)
    """
    hist = st.session_state.get("question_history", [])
    st.session_state["question_nonce"] = st.session_state.get("question_nonce", 0) + 1
    nonce = st.session_state["question_nonce"]

    focus_kw = st.session_state.get("focus_kw", "")
    idx_json = st.session_state.get("book_index_json", {})
    summary_txt = " / ".join(idx_json.get("summary", [])[:3])
    scenes_txt  = " / ".join(idx_json.get("key_scenes", [])[:3])

    draft_tail = (context.get("draft", "") or "")[-200:]

    system = (
        "너는 초등학생이 이해하기 쉬운 열린 질문을 만드는 한국어 교사야. "
        "반드시 서로 다른 질문 시작어를 사용하고(왜/어떻게/만약), 각 질문은 1줄, 물음표(?)로 끝나야 해. "
        "이미 했던 질문들과 표현/의미가 겹치지 않게 만들어."
        "아이들이 글을 쓰다 질문을 하면 너는 뒤를 이을 수 있는 책과 관련된 문장들을 세 가지 이상 추천해줘야 해"
    )

    user = f"""
[학생 정보]
- 책 제목: {context.get('book_title','미정')}
- 선택된 키워드(있으면 반영): {focus_kw or '(없음)'}
- 책 요약: {summary_txt or '(없음)'}
- 핵심 장면: {scenes_txt or '(없음)'}
- 현재 초안(마지막 200자): {draft_tail or '(없음)'}

[이미 했던 질문들(중복 금지)]
{chr(10).join('• '+q for q in hist[-20:]) if hist else '(없음)'}

[요청]
- 서로 다른 시작어로 3개: ①왜..., ②어떻게..., ③만약...
- 각 질문은 한 줄, 반드시 '?'로 끝내기
- 번호/불릿 없이 질문만 3줄
- 다양화토큰: {nonce}

질문 3개:
"""

    resp = call_openai_api(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=300
    )
    if not resp:
        base = [
            "왜 이 장면이 특히 중요한지 스스로 설명할 수 있나요?",
            "어떻게 이 책의 메시지를 일상에서 실천할 수 있을까요?",
            "만약 당신이 주인공이었다면 어떤 결정을 내렸을까요?"
        ]
        st.session_state["question_history"] = (hist + base)[-50:]
        return base

    raw_lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
    cleaned, seen = [], set(hist[-100:])
    for ln in raw_lines:
        q = ln.lstrip("0123456789.-•* ").strip()
        if not q.endswith("?"):
            q = q.rstrip(".!…") + "?"
        if q in seen:
            continue
        seen.add(q)
        cleaned.append(q)
        if len(cleaned) == 3:
            break

    starters = ["왜", "어떻게", "만약"]
    while len(cleaned) < 3:
        s = starters[len(cleaned) % 3]
        kw_part = f" '{focus_kw}'" if focus_kw else ""
        filler = f"{s} 이 책을 통해{kw_part} 내가 배우거나 바꿀 수 있는 점은 무엇일까요?"
        if filler not in seen:
            cleaned.append(filler)
            seen.add(filler)

    st.session_state["question_history"] = (hist + cleaned)[:50]
    return cleaned[:3]

def check_spelling_and_grammar(text):
    """맞춤법/표현 피드백 최대 3개.
    1차: 맞춤법/띄어쓰기/조사/어미 등 규범 위반만.
    2차: 1차 결과가 없으면 '표현 다듬기' 1~2개 제안."""
    if not text or not text.strip():
        return []

    # --- 1차: 규범 위반만 ---
    system = (
        "너는 한국어 교정 교사다. 국립국어원 한글 맞춤법/띄어쓰기/외래어 표기법 기준으로만 판단한다. "
        "출력은 각 줄 하나, 총 2~3줄. 각 줄은 '원래 표현 → 고친 표현 (이유)' 형식을 반드시 지켜라. "
        "진짜 규범 위반이 없으면 정확히 '없음' 한 줄만 출력해라."
    )
    user = f"""다음 글에서 규범 위반(맞춤법, 띄어쓰기, 조사·어미)만 2~3개 고쳐줘.
형식: "원래 표현 → 고친 표현 (이유)"
글:
{text[:1000]}"""

    resp = call_openai_api(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        max_tokens=300
    ) or ""

    lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
    # '없음'이면 2차로 표현 팁 제공
    if any(ln == "없음" for ln in lines) or not lines:
        style_sys = (
            "너는 한국어 글쓰기 코치다. 규범 위반이 없을 때만 표현 개선 팁을 준다. "
            "각 줄 하나, 최대 2줄. 형식: '표현) 제안 문장 (이유)'"
        )
        style_user = f"""다음 글의 표현을 자연스럽게 다듬을 수 있는 개선 제안을 1~2개만 제시해줘.
형식: "표현) 제안 문장 (이유)"
글:
{text[:800]}"""
        style = call_openai_api(
            [{"role": "system", "content": style_sys},
             {"role": "user", "content": style_user}],
            max_tokens=160
        ) or ""
        tips = []
        for ln in style.splitlines():
            s = ln.strip().lstrip("0123456789.-•* ").strip()
            if s and s.startswith("표현)"):
                tips.append(s)
                if len(tips) == 2:
                    break
        return tips

    # 1차 결과 정제 (최대 3개)
    out = []
    for ln in lines:
        s = ln.lstrip("0123456789.-•* ").strip()
        if "→" in s and "(" in s and ")" in s:
            out.append(s)
            if len(out) == 3:
                break
    return out

def detect_stage(draft: str, outline: dict) -> str:
    """간단 휴리스틱 단계 감지"""
    if not draft.strip():
        return "intro"
    if len(draft) < 150:
        return "intro"
    if any(x in draft for x in ["가장 인상 깊었던", "기억에 남는", "느꼈", "교훈"]):
        return "body"
    return "concl"

def detect_stage_llm(draft, outline):
    """LLM 기반 단계 감지 (intro/body/concl 중 하나만)"""
    prompt = f"""
학생 초안을 보고 현재 어느 단계인지 intro/body/concl 중 하나로만 답해줘. 다른 말은 하지 마.
초안(마지막 800자):
{(draft or '')[-800:]}
개요:
{json.dumps(outline or {}, ensure_ascii=False)}
"""
    resp = call_openai_api([{"role": "user", "content": prompt}], max_tokens=8) or ""
    r = resp.lower()
    if "intro" in r:
        return "intro"
    if "concl" in r or "결론" in r:
        return "concl"
    return "body"

def suggest_next_sentences(context, n=3):
    """현재 초안의 끝을 자연스럽게 잇는 '한 문장' 제안 n개.
    - LLM 응답이 비어도 폴백으로 항상 n개 채움
    - 책 요약/장면/키워드와 focus_kw를 가볍게 반영
    """
    draft_tail = (context.get("draft", "") or "")[-500:]
    book_index_raw = st.session_state.get("book_index", "") or ""
    idx = st.session_state.get("book_index_json", {}) or {}
    focus_kw = st.session_state.get("focus_kw", "")

    # 1) LLM 요청
    prompt = f"""
학생 초안의 마지막 부분을 자연스럽게 이어갈 **한 문장** 제안을 {n}개 만들어줘.
조건:
- 초등학생이 이해하기 쉬운 표현
- 각 제안은 한 문장만 (마침표로 끝내기)
- 너무 길지 않게 (30자~70자 권장)
- 같은 의미/표현 중복 금지

[지금까지 초안(마지막 500자)]
{draft_tail or '(없음)'}

[책 지식(요약/장면/키워드)]
{book_index_raw[:800] or '(없음)'}

[선택된 키워드]
{focus_kw or '(없음)'}
"""
    resp = call_openai_api([{"role": "user", "content": prompt}], max_tokens=300) or ""

    # 2) 1차 후보 정리
    cand_lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
    cleaned, seen = [], set()
    for ln in cand_lines:
        s = ln.lstrip("0123456789.-•* ").strip()
        s = s.replace("  ", " ").rstrip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
        if len(cleaned) >= n:
            break

    # 3) 폴백: 부족하면 키워드/요약/장면 힌트로 생성해 채우기
    if len(cleaned) < n:
        summaries = idx.get("summary", [])[:2]
        scenes = idx.get("key_scenes", [])[:2]
        kw_list = idx.get("keywords", [])
        kw = focus_kw or (kw_list[0] if kw_list else "이야기")

        fallback_pool = []

        for s in summaries:
            if s:
                fallback_pool.append(f"나는 {s.split(' ')[0]} 부분을 떠올리며 내 생각을 더 자세히 적어 보기로 했다.")
        for sc in scenes:
            if sc:
                fallback_pool.append(f"특히 '{sc[:20]}' 장면을 바탕으로 내가 느낀 점을 한 번 더 정리하고 싶다.")

        fallback_pool += [
            f"나는 {kw}에 대해 내가 바꿀 수 있는 작은 실천을 하나 정해 보았다.",
            "이어서 나는 이 장면과 내 경험을 비교하며 비슷한 점과 다른 점을 쓰려고 한다.",
            "마지막으로 나는 이 이야기에서 배운 점을 하루 동안 실천해 보고 기록해 보려 한다.",
        ]

        for fb in fallback_pool:
            if len(cleaned) >= n:
                break
            if fb not in seen:
                cleaned.append(fb)
                seen.add(fb)

    return cleaned[:n]

# =========================
# 4) 추천 도서(기본)
# =========================
# =========================
# 4) 추천 도서(추천 텍스트 원문 포함)
# =========================
RECOMMENDED_BOOKS = {
    "소리없는 아이들 - 황선미": {
        "raw": """농아 장애인 아이들과의 만남과 이해를 그린 작품이다. 아빠의 실직으로 사과 농장을 하시는 할아버지 댁에 오게 된 연수는 동네아이들과 빨간 양철지붕의 농아원 아이들 이렇게 두 아이들을 만나게 된다. 경호와 동생 경미, 동욱이 등 동네아이들과 연수는 참외서리를 하고 쓰으들이라고 놀리는 농아원 아이들에게 누명을 씌운다. 어머니의 죽음이 쓰으들 때문이라고 생각하는 경호다. 가게를 하는 경호네, 태풍으로 물난리가 나고 경호네 가게도 물에 잠긴다. 경호네 식구들이 물건을 밖으로 빼냈는데 밤새 농아애들이 훔쳐간다. 배고픈 농아들, 큰 애들은 굶고 작은 애들만 먹었다가 식중독에 걸린다. 경호 엄마가 따지러 가지만 애들이 너무 불쌍해서 아무 말도 못하고 나오는데 갑자기 달려든 개에게 다리를 물린다. 병원에서는 다 나았다고 했지만 시름시름 앓더니 결국 돌아가시고 만다. 그리고 이 책에는 특별한 등장인물이 한 사람있다. 농아인줄만 알았던 창민이다. 할아버지 농장에서 일하는 부부의 아들로 밝혀지는 창민이는 독일에 보내진 후 잠깐 다니러 온 아이다. 입양된 건 아니고 교회를 통해서 알게 된 선교사 집안에서 학교에 다니고 있다. 당연히 독일말도 한다. 농아의 가족으로 창민은 말이 필요 없는 아이다. 어떤 때는 나한테 목소리가 있다는 것도 깜빡 잊는다. 작가의 이 책에서 개가 중요한 역할을 한다. 경호는 동욱이의 개를 죽게 만들고, 창민이는 경호와 동욱이 모두에게 강아지 한 마리씩을 선물한다. 태풍을 통해서 입은 피해를 복하면서 동네사람들과 농아들은 함께 살아가는 법을 알게 된다. 연수와 그녀의 부모님이 시골에 정착하는 과정도 흐뭇하다."""
    },
    "나와 조금 다를 뿐이야 - 이금이": {
        "raw": """영무는 수아가 예뻐서 수아를 좋아했다. 그래서 수아가 시골 은내리 마을에 있는 영무의 학교에 전학을 온 것이 기뻤다. 하지만 수아와 한 반이 되어 함께 생활하는 동안 영무는 괴로워한다. 왜냐하면 수아에게는 마음대로 병이 있었기 때문이다. 그래서 영무가 수아의 장애 때문에 수아가 영무의 사촌이라는 이유만으로 수아의 모든 학교생활을 책임져야 했다. 아마 내가 영무였어도 사촌이라는 이유만으로 장애인과 같이 다니기 창피하고 싫었을 것이다. 수아는 맘대로 병 때문에 공부 시간에도 제 마음대로 동화책을 읽고 선생님의 허락도 안 받고 화장실도 다녀온다. 그리고 수시로 사라져 버린다. 그래서 영무는 수아가 사라질 때면 수아를 찾아야 하고 수아가 숙제를 안 해오거나 준비물을 안 가져오면 영무가 대신 혼난다. 또 아이들이 수아를 ‘바보’라고 놀리고 영무는 그런 수아가 창피하였다. 그래서 영무는 사람들이 아끼는 수아가 미워 수아를 성남이를 시켜 때리기도 하고 물에 빠트리기도 하였다.
하지만 누구에게나 단점이 있으면 장점도 있듯이 수아에게는 춤과 노래, 암기력이 좋았다. 그래서 한번 본 ‘흥부 놀부’마당 놀이를 잘 따라하였다. 그래서 수아의 같은 반 아이들에게 수아의 춤과 노래 실력을 보여 주었고, 수아의 반 아이들은 모두 수아의 춤과 노래 실력에 감탄을 하였다. 나는 장애인은 모두 보통사람들보다 못한다고 생각하였는데 이 동화에 있는 수아를 보고서야 장애인들도 보통 사람들보다 더 잘하는 것이 있다는 것을 알게 되었다. 그리고 그런 사실이 놀랍고 신기하였다.
그러던 어느 날 영무가 수아에게 춤과 노래를 시키고 할머니들에게 돈을 받았다. 그런데 영무의 고모가 그 광경을 보았다. 영무의 아빠는 그 사실을 아시고 화가 나셔서 영무를 혼내셨다. 그래서 영무는 울면서 수아 때문에 섭섭하고 억울하였던 일과 영무의 마음을 줄줄히 이야기 하였다. 그리고 그 일로서 영무의 가족들이 영무의 마음을 이해하게 되었다. 나는 결국 수아 때문에 영무가 억울했던 일 섭섭했던 일이 풀려서 다행이라고 생각했다. 나는 평소 가족들에게 그런 일이 없지만 만약 그런 일이 있다면 아마 참지 못하였을 것이다. 그런 면에서 섭섭하고 억울하였던 것을 참아낸 영무가 대단하게 느껴졌다.
그리고 수아는 자신이 잘못했다며 울지 말라고‘흥부 놀부’를 추었다. 가족들은 수아의 춤과 노래 실력에 넋을 잃고 보았다. 가족들도 수아가 춤과 노래에 뛰어난 재능이 있다는 것을 알게 된 것이다. 그래서 영무의 고모는 수아의 재능을 키워주기 위해 다시 도시로 전학을 갔다.
나는 장애인을 실제로 대하여 본 적이 없다. 만약 내가 정말 수아 같은 사촌이 있다면 엄청 창피하고 싫었을 것이다. 처음의 영무처럼 말이다. 하지만 수아에게 단점만 있는 것이 아니라 장점도 있었다. 춤과 노래, 암기력에 말이다. 그래서 나는 장애인이든 비장애인이든 장점과 단점이 있는 비슷한 사람이라는 것을 느꼈다. 앞으로는 장애인들을 이상하게 생각하지만 말고 그들을 따뜻한 시선으로 바라봐 줘야 겠다.
또 나는 장애인이라면 무엇이든지 보통 사람들보다 못한다고 생각을 했었다. 하지만 이 동화에 나오는 수아를 보고 장애인이라도 보통사람들보다 더 잘 하는 것이 있다는 것을 알았다. 수아는 보통 사람들이 따라할 수 없을 만큼의 노래실력과 춤 실력을 가지고 있기 때문이다. 그래서 장애인들은 보통 사람들보다 못한 다는 나의 고정관념을 바꿔주었다. 그리고 누구든지 잘하는 것이 있다는 것을 알게 되었다. 나는 이 동화 때문에 장애인들은 나와 조금 다를 뿐이라는 것을 나에게 확실히 알게 되었다. 그리고 서로 다른 사람들이 모여 만들어 내는 세상의 아름다움과 조화로움을 깊이 생각 하게 되었다."""
    },
    "여름과 가을 사이 - 박슬기": {
        "raw": """여름이와 가을이는 8살 때부터 5년을 붙어 다닌 단짝이다. 반 아이들 모두가 그 사실을 잘 알고 있다. 그런데 여름이가 요 며칠 가을이의 연락을 피하는 것 같다. 아니나 다를까 거짓말을 하고 다른 친구를 만나는 여름이를 가을이가 목격하고 말았다. 게다가 둘만 알자 약속했던 아지트에서 여름이가 다른 친구와 함께 있는 거다. 가을이는 여름이를 원망했지만 여름이는 외려 당당하게 말한다. ‘이제 너와 노는 것이 재미없어.’ 상처받은 가을이는 여름이에게 보란 듯이 다른 단짝을 찾겠다는 다짐을 한다. 그러나 새로운 단짝을 만드는 것은 좀처럼 쉽지가 않다. 말을 거는 것조차 어려웠다. 단짝이라면 모름지기 모든 비밀을 다 공유해야 하며 다른 친구가 끼어들 틈을 만들면 안 되었다. 가을이에게 우정이란 그런 거였다. 단짝을 위해서라면 모든 걸 맞춰줄 준비가 되어있었다. 자신의 의견을 묵살하고서라도. 그렇게 맞춰주었건만 여름이는 다른 친구를 만나고 절교 비슷한 선언을 했다. 다른 단짝을 찾기 위해 만났던 이플이도 다른 아이들과 더 즐거운 것 같다. 난 이제 이플이를 위해 다 맞춰줄 작정이었는데! 도대체 뭐가 잘 못 된 거지? 그렇게까지 심하게 말하려던 것은 아니었다. 하지만 분명히 사실이었다. 요즘 들어 가을이와 노는 것이 재미가 없다. 가을이는 남자아이들에게도 관심이 없고 뭐든 공유하려는 눈빛이 부담스럽다. 게다가 여름이는 사춘기가 왔는지 작은 일에도 기분이 널을 뛴다. 언니의 사춘기를 보며 ‘나는 저러지 않아야지’ 했던 짜증들을 반복하다니. 결코 사춘기에 굴복하지 않으려 하지만 뭔지 모를 마음이 복잡하다. 가을에게 사과하고 싶은 마음과 그러고 싶지 않은 마음이 함께다. 해밀이와 놀면 그런 걱정이 사라진다. 해밀이는 빠른 아이다. 이미 남자친구도 있고 머리도 예쁘게 묶고 다닌다. 대화도 잘 통한다. 분명 가을이와 가장 친했었는데....... 여름이는 왜 이런 기분에 휩싸인 걸까? 정말 사춘기 때문인 걸까? 가을이와 여름이가 싸웠다는 소문은 삽시간에 퍼져나갔다. 여름이는 피해자처럼 엎드려있는 가을이가 못마땅하다. 설상가상으로 의문의 쪽지까지 여름이를 괴롭힌다. “3일 안에 제대로 사과하고 화해하지 않으면 곧 당신에게 엄청나게 불행한 일이 닥칠 것이다.” 이 쪽지는 무려 2차례나 보내왔고 마지막 경고는 빨간색 글씨체로 쓰여 더욱 무서웠다. 여름이는 처음 가을이를 의심했다. 소리를 지르기까지 했다. 기어이 무단 조퇴를 감행하기도 한다. 여름이는 내면의 거친 목소리가 자신을 조종하는 것만 같다. 가을이는 쪽지를 보내지 않았다. 의심을 벗기 위해 범인을 찾기로 결심한다. 여름이와 함께. 두 사람은 범인을 잡을 수 있을까? 범인도 잡고 우정도 다시금 잡을 수 있을까?"""
    },
    "인어 소녀 - 차율이": {
        "raw": """『인어 소녀』는 제주 바닷가의 작은 라면집 ‘Moon漁(문어)’에서 시작한다. 주인공 규리는 인간 엄마와 인어 아빠 사이에서 태어난 혼혈 인어로, 다리가 바닷물을 만나면 꼬리지느러미로 변한다. 어린 시절 회색 상어에게 습격당한 기억 때문에 바다를 두려워하지만, 어느 날 갑자기 흔적도 없이 사라진 아빠 ‘온’을 찾기 위해 결국 물속 세계로 내려간다. 바닷속에서 규리는 바다거북으로 변신하는 인어 ‘탄’과 샛별돔 인어 ‘시호’를 만나 도움을 받고, 그들을 따라 인간과는 단절된 인어 세계의 층층이 숨은 규칙과 금기를 알게 된다. 그 과정에서 아빠의 실종에는 ‘카슬’이라는 지배자의 존재가 얽혀 있음을 깨닫는다. 카슬은 바닷가재 인어로, 오래전 아빠가 인간의 다리를 얻는 대가로 “혼혈 인어 아이가 열두 살이 되면 지배자에게 보내야 한다”는 계약을 강요했고, 아빠가 이를 어기자 아빠를 노예처럼 붙잡아 둔 것이다. 규리는 아빠를 구하려고 스스로 카슬의 노예가 되어 ‘인어 청소부’ 같은 허드렛일과 위험한 심부름을 도맡는다. 노역 끝에 닿은 곳은 ‘괴물들이 사는 섬’이라 불리는 플라스틱 섬. 사람들의 쓰레기가 바다에 떠밀려와 굳어 만들어진 인공섬으로, 버림받은 기형 인어 아이들이 모여 서로를 의지하며 살아간다. 규리는 그곳에서 바다가 ‘하얀 바다’로 병들어 가는 현실과, 고래들의 마지막 시간을 간직한 ‘고래 무덤’의 비밀을 마주하며 한층 성장한다. 플라스틱 섬과 고래 무덤을 오가는 모험 속에서 규리는 카슬의 힘의 근원이 전통 인어 ‘신지께’에서 비롯되었음을 알게 되고(카슬은 신지께의 힘으로 인어가 된 인물), 더는 누구의 제물이 되지 않겠다고 결심한다. 결국 규리는 탄·시호와 뜻을 모아 카슬의 억압과 부당한 계약에 맞서며, 플라스틱 섬의 아이들과 아버지를 속박에서 풀어낼 길을 스스로 찾아 나선다. 바다와 육지, 두 세계 사이에서 흔들리던 규리는 “가족을 지키고 자기 자리를 선택하는 일”이야말로 진짜 용기임을 깨닫고, 달빛 비치는 바다를 지나 자신의 삶으로 돌아갈 힘을 얻는다."""
    },
}

def _book_text_from_info(info: dict) -> str:
    """RECOMMENDED_BOOKS 항목에서 안전하게 본문 텍스트를 만들어준다."""
    if info.get("raw"):
        return info["raw"]
    summary = info.get("summary", "")
    scenes = info.get("key_scenes", []) or []
    if summary or scenes:
        extra = "\n\n주요 장면들:\n" + "\n".join(f"- {s}" for s in scenes) if scenes else ""
        return (summary + extra).strip()
    return ""

# =========================
# 5) UI 구성
# =========================
def render_sidebar():
    st.sidebar.header("⚙️ 설정")

    # 모델명 변경 가능 (기본 gpt-5)
    # 모델 고정: gpt-4o-mini (UI 숨김)
    api_status = "🟢 연결됨" if os.getenv("OPENAI_API_KEY") else "🔴 미연결"
    st.sidebar.caption(f"AI 상태: {api_status} · 사용 모델: gpt-4o-mini(고정)")


    st.session_state["role"] = st.sidebar.selectbox("사용자 모드", ["학생", "교사"], index=0)
    st.session_state["n_sugs"] = st.sidebar.slider("AI 제안 개수", 1, 5, st.session_state.get("n_sugs", 3))
    st.session_state["enable_hints"] = st.sidebar.checkbox("맞춤법 검사", st.session_state.get("enable_hints", True))
    st.session_state["enable_questions"] = st.sidebar.checkbox("AI 유도 질문", st.session_state.get("enable_questions", True))

    st.sidebar.divider()
    st.sidebar.subheader("📚 책 선택")

    book_options = ["직접 입력"] + list(RECOMMENDED_BOOKS.keys())
    selected_book = st.sidebar.selectbox("책 선택", book_options)

    # 선택이 바뀐 경우만 반영
    if selected_book != st.session_state.get("selected_book_prev", ""):
        st.session_state["selected_book_prev"] = selected_book
        if selected_book != "직접 입력":
            st.session_state["book_title"] = selected_book
            info = RECOMMENDED_BOOKS[selected_book]
            st.session_state["book_text"] = _book_text_from_info(info)

            st.sidebar.success(f"'{selected_book}' 정보가 로드되었습니다!")
            # 자동 인덱싱
            index_book_text()

    if selected_book == "직접 입력":
        st.session_state["book_title"] = st.sidebar.text_input("책 제목", st.session_state["book_title"])
        uploaded = st.sidebar.file_uploader("책 요약/중요 부분(.txt)", type=["txt"])
        if uploaded:
            try:
                st.session_state["book_text"] = uploaded.read().decode("utf-8")
            except Exception:
                st.session_state["book_text"] = uploaded.read().decode("utf-8", errors="ignore")
            log_event("book_uploaded", {"name": uploaded.name, "chars": len(st.session_state["book_text"])})
            index_book_text()

        if st.sidebar.button("📑 직접 입력한 내용 인덱싱/요약"):
            index_book_text()

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
    # 맞춤법 & 표현
    if st.session_state["enable_hints"] and st.session_state["draft"].strip():
        st.markdown("**✍️ AI 맞춤법 및 표현 도움**")
        if st.button("🔍 맞춤법 검사하기"):
            with st.spinner("AI가 글을 검토하고 있어요..."):
                feedback = check_spelling_and_grammar(st.session_state["draft"]) or []
            if feedback:
                for tip in feedback:
                    st.info(tip)
            else:
                st.success("훌륭해요! 특별한 문제점이 없어 보입니다.")

    # 생각 유도 질문
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

def render_keyword_pills():
    """인덱싱된 키워드로 클릭형 태그 UI"""
    kws = st.session_state.get("book_index_json", {}).get("keywords", [])
    if not kws:
        return
    cols = st.columns(min(5, len(kws)))
    for i, kw in enumerate(kws):
        with cols[i % len(cols)]:
            if st.button(f"#{kw}", key=f"kw_{i}"):
                st.session_state["focus_kw"] = kw
                try:
                    st.toast(f"키워드 '{kw}'를 선택했어요!", icon="✅")
                except Exception:
                    st.info(f"키워드 '{kw}'를 선택했어요!")

def render_suggestion_block(label, key_block, icon):
    st.markdown(f"### {icon} {label} 작성 도움")

    if st.button(f"🤖 AI {label} 제안 받기", key=f"generate_{key_block}"):
        ctx = {
            "book_title": st.session_state["book_title"],
            "book_text": st.session_state["book_text"],
            "outline": st.session_state["outline"],
            "draft": st.session_state["draft"],
        }
        suggestions = generate_ai_suggestions(ctx, key_block, st.session_state["n_sugs"])
        st.session_state[f"suggestions_{key_block}"] = suggestions

    current_suggestions = st.session_state.get(f"suggestions_{key_block}", [])
    if current_suggestions:
        st.markdown("**🎯 AI가 만든 작문 제안들**")
        for i, sug in enumerate(current_suggestions):
            with st.container(border=True):
                st.write(f"**제안 {i+1}:** {sug}")
                edited = st.text_area("수정해서 추가", value=sug, height=70, key=f"edit_{key_block}_{i}")
                c1, c2 = st.columns([1, 1])
                with c1:
                    if st.button("✅ 수정 본문에 추가", key=f"accept_edit_{key_block}_{i}"):
                        text_to_add = (edited or sug).strip()
                        st.session_state["use_chat_mode"] = False
                        _queue_append(text_to_add)  # ✅ 큐 사용
                with c2:
                    if st.button("❌ 패스", key=f"reject_{key_block}_{i}"):
                        log_event("ai_suggestion_rejected", {"block": key_block, "text": sug})
                        st.info("다른 제안을 확인해보세요!")


def render_outline():
    # ===== Step 1. 책 내용 학습(요약/장면/키워드) =====
    st.subheader("📖 Step 1. 책 내용 학습(요약/장면/키워드)")

    learned = bool(st.session_state.get("book_index_json", {}).get("summary"))
    if st.session_state["book_title"]:
        st.caption(f"선택된 책: **{st.session_state['book_title']}**")

    if learned:
        st.success("책 내용이 학습되었습니다. (요약/핵심 장면/키워드 추출 완료)")
    else:
        st.warning("아직 책 내용이 학습되지 않았어요. 추천 도서를 선택/업로드하거나, 아래에 본문을 붙여넣고 인덱싱을 실행해주세요.")

    col_prev, col_ctrl = st.columns([2, 1])

    # 왼쪽: 요약/장면/키워드 미리보기
    with col_prev:
        idx = st.session_state.get("book_index_json", {})
        if idx:
            with st.container(border=True):
                st.markdown("**🧾 3문장 요약**")
                for s in idx.get("summary", []):
                    st.write(f"• {s}")

                st.markdown("**🎬 핵심 장면**")
                for s in idx.get("key_scenes", []):
                    st.write(f"• {s}")

                kws = idx.get("keywords", [])
                if kws:
                    st.markdown("**🔖 키워드(클릭하여 집중 토픽 선택)**")
                    kw_cols = st.columns(min(5, len(kws)))
                    for i, kw in enumerate(kws):
                        with kw_cols[i % len(kw_cols)]:
                            if st.button(f"#{kw}", key=f"kw_step1_{i}"):
                                st.session_state["focus_kw"] = kw
                                try:
                                    st.toast(f"키워드 '{kw}'를 선택했어요!", icon="✅")
                                except Exception:
                                    st.info(f"키워드 '{kw}'를 선택했어요!")

    # 오른쪽: 재요약/붙여넣기 인덱싱
    with col_ctrl:
        st.markdown("**도구**")
        if st.button("🔁 다시 요약/인덱싱"):
            index_book_text()

        with st.expander("📥 본문을 직접 붙여넣기"):
            paste = st.text_area("책 본문/요약 붙여넣기", key="paste_book_text", height=140)
            if st.button("요약/인덱싱 실행", key="reindex_btn"):
                if paste and paste.strip():
                    st.session_state["book_text"] = paste
                    index_book_text()
                else:
                    st.warning("붙여넣은 내용이 없습니다.")

    st.divider()

    # ===== Step 2. 글의 뼈대 및 초안 만들기 =====
    st.subheader("📝 Step 2. 글의 뼈대 및 초안 만들기")

    cols = st.columns(3)
    with cols[0]:
        st.markdown("**서론 (책 소개/읽게 된 이유)**")
        st.text_area(
            "서론 입력",
            key="outline_intro",
            height=120,
            placeholder="• 책을 읽게 된 계기\n• 첫인상이나 기대\n• 간단한 책 소개",
            label_visibility="collapsed"
        )
    with cols[1]:
        st.markdown("**본론 (인상 깊은 장면/느낀 점)**")
        st.text_area(
            "본론 입력",
            key="outline_body",
            height=120,
            placeholder="• 가장 기억에 남는 장면\n• 그 이유와 느낀 점\n• 나의 경험과 연결",
            label_visibility="collapsed"
        )
    with cols[2]:
        st.markdown("**결론 (배운 점/추천 이유)**")
        st.text_area(
            "결론 입력",
            key="outline_concl",
            height=120,
            placeholder="• 책에서 배운 것\n• 추천하고 싶은 이유\n• 앞으로의 다짐",
            label_visibility="collapsed"
        )

    # 세션 동기화
    st.session_state["outline"]["intro"] = st.session_state.get("outline_intro", "")
    st.session_state["outline"]["body"]  = st.session_state.get("outline_body", "")
    st.session_state["outline"]["concl"] = st.session_state.get("outline_concl", "")


def render_snapshot_bar():
    c1, c2 = st.columns([1, 2])
    with c1:
        st.button("💾 임시 저장", on_click=lambda: _save_snapshot())
    with c2:
        versions = st.session_state.get("saved_versions", [])
        if versions:
            idx = st.selectbox("저장본 불러오기", list(range(len(versions))), format_func=lambda i: versions[i]["t"])
            if st.button("↩️ 이 버전으로 되돌리기"):
                st.session_state["draft"] = versions[idx]["text"]
                st.success("해당 시점 버전으로 복구했습니다.")
                st.rerun()

def _save_snapshot():
    if not st.session_state.get("draft", "").strip():
        st.warning("저장할 내용이 없습니다.")
        return
    st.session_state["saved_versions"].append({"t": now(), "text": st.session_state["draft"]})
    st.success("💾 임시 저장 완료!")

def render_editor():
    st.subheader("✏️ Step 3. 글 작성 및 다듬기")

    col1, col2 = st.columns([2, 1])

    # ---------- 좌측: 자유 편집 + 키워드 + 도구 탭 ----------
    with col1:
        st.toggle(
            "Enter로 전송(채팅 입력 모드)",
            key="use_chat_mode",
            help="켜면 Enter로 전송되고, 줄바꿈은 Shift+Enter입니다. 끄면 자유 편집 모드입니다."
        )

        if st.session_state["use_chat_mode"]:
            st.text_area("현재 초안 (읽기 전용)", value=st.session_state["draft"], key="draft_view", height=280, disabled=True)
            new_line = st.chat_input("여기에 입력하고 Enter를 눌러 추가 (Shift+Enter 줄바꿈)")
            if new_line and new_line.strip():
                log_event("draft_appended", {"source": "chat_input", "chars": len(new_line)})
                _queue_append(new_line)
        else:
            st.text_area(
                "내가 쓰고 있는 독서감상문",
                key="draft",
                height=280,
                placeholder="개요를 참고해 직접 작성하거나, 아래 탭의 제안을 활용해 확장하세요.",
            )

        # 집중 키워드 선택
        st.markdown("**🔖 집중 키워드**")
        render_keyword_pills()

        # === 도구 탭: 현재 단계 자동 제안 / 블록별 제안 / 다음 문장 추천 ===
        t1, t2, t3 = st.tabs(["현재 단계 자동 제안", "블록별 제안", "다음 문장 추천"])

        # 1) 현재 단계 자동 제안
        with t1:
            st.caption("초안과 개요를 보고 지금 단계(intro/body/concl)를 추정해 그에 맞는 문장 제안을 생성합니다.")
            if st.button("🔄 새 제안 받기", key="auto_stage_refresh"):
                stage = detect_stage_llm(st.session_state["draft"], st.session_state["outline"])
                st.session_state["last_stage"] = stage
                ctx = {
                    "book_title": st.session_state["book_title"],
                    "book_text": st.session_state["book_text"],
                    "outline": st.session_state["outline"],
                    "draft": st.session_state["draft"],
                }
                st.session_state["auto_stage_suggestions"] = generate_ai_suggestions(ctx, stage, st.session_state["n_sugs"])
                st.success(f"현재 단계 추정: {stage}")

            if st.session_state.get("auto_stage_suggestions"):
                for i, sug in enumerate(st.session_state["auto_stage_suggestions"]):
                    with st.container(border=True):
                        st.write(f"**제안 {i+1}:** {sug}")
                        c1, c2 = st.columns([1,1])
                        with c1:
                            if st.button("추가", key=f"auto_add_{i}"):
                                _queue_append(sug)
                        with c2:
                            edited = st.text_input("수정 후 추가", value=sug, key=f"auto_edit_{i}")
                            if st.button("수정본 추가", key=f"auto_edit_add_{i}"):
                                txt = (edited or sug).strip()
                                _queue_append(txt)

        # 2) 블록별 제안(서론/본론/결론)
        with t2:
            c1b, c2b, c3b = st.columns(3)
            with c1b:
                render_suggestion_block("서론", "intro", "🌟")
            with c2b:
                render_suggestion_block("본론", "body", "💭")
            with c3b:
                render_suggestion_block("결론", "concl", "🎭")

        # 3) 다음 문장 추천
        with t3:
            st.caption("현재 초안의 마지막 부분을 자연스럽게 잇는 한 문장을 추천합니다.")
            if st.button("➡ 다음 문장 3개", key="next_sent_refresh"):
                ctx = {
                    "book_title": st.session_state["book_title"],
                    "draft": st.session_state["draft"],
                    "book_text": st.session_state["book_text"]
                }
                st.session_state["next_sugs"] = suggest_next_sentences(ctx, 3)

            for i, s in enumerate(st.session_state.get("next_sugs", []), 1):
                c1n, c2n = st.columns([6,1])
                with c1n:
                    st.write(f"• {s}")
                with c2n:
                    if st.button("추가", key=f"next_add_{i}"):
                        _queue_append(s)

    # ---------- 우측: 품질 패널(맞춤법/질문) ----------
    with col2:
        render_quality_panel()

    # ---------- 저장/완료 ----------
    st.subheader("💾 Step 3 마무리: 저장/완료")
    render_snapshot_bar()

    if st.session_state["draft"].strip():
        word_count = len(st.session_state["draft"])
        st.info(f"현재 글자 수: {word_count}자")
        c1f, c2f = st.columns(2)
        with c1f:
            st.download_button(
                "📄 텍스트 파일로 저장",
                data=st.session_state["draft"],
                file_name=f"{st.session_state['book_title']}_독서감상문.txt" if st.session_state["book_title"] else "독서감상문.txt",
            )
        with c2f:
            if st.button("🎉 작성 완료!"):
                log_event("writing_completed", {"word_count": word_count, "book": st.session_state["book_title"]})
                st.balloons()
                st.success("독서감상문 작성을 완료했습니다! 수고하셨어요!")
    else:
        st.warning("아직 작성된 내용이 없습니다. 위 탭의 제안/다음 문장 추천을 활용하거나 직접 작성해보세요.")

# =========================
# 6) 메인
# =========================
def main():
    st.set_page_config(
        page_title="AI 독서감상문 작문 도우미",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    _apply_draft_queue()   # ✅ 큐 반영 (필수)
    render_sidebar()

    st.title("📚 AI와 함께 쓰는 독서감상문")
    st.caption("🤖 AI 제안 → 🤔 선택/수정 → ✨ 완성! AI 선생님과 함께 특별한 독서감상문을 만들어보세요.")

    # 진행 상황
    steps_completed = [
        bool(st.session_state.get("book_index_json", {}).get("summary")),   # ✅ Step1: 학습 여부 기준
        any(st.session_state["outline"].values()),
        bool(st.session_state["draft"].strip()),
    ]
    progress = sum(steps_completed) / 3
    st.progress(progress)
    st.caption(f"진행 상황: {int(progress * 100)}% 완료")

    # Step1 + Step2
    render_outline()
    st.divider()

    # Step3
    render_editor()

if __name__ == "__main__":
    main()
