# Metaculus 보정형 예측 봇 (FutureEval + MiniBench)

Metaculus 공식 템플릿(`metac-bot-template`, forecasting-tools 0.3.5)을 고친 봇이다.
`METACULUS_TOKEN`만 넣어도 돌아간다. LLM 키를 넣으면 그 키를 우선 쓴다.

## 템플릿과 다른 점
| 항목 | 템플릿 | 이 봇 |
|---|---|---|
| 이진 질문 프롬프트 | 현상 유지·시나리오 서술 | **기준율 → 증거 → 보정** 4단계 (`forecast_logic.py`) |
| 답 읽기 | 매번 파서 LLM 호출 2회 | `Probability: ZZ%`를 정규식으로 먼저 읽음. 실패할 때만 파서 LLM 사용 |
| 극단값 | 1~99% | 샘플마다, 그리고 집계 후에 다시 1~99%로 자름 |
| 샘플 수 | 5회 | 5회 (`BOT_PREDICTIONS_PER_QUESTION`) |
| 집계 | 이진 중앙값, 다지선다 **평균** | 이진 중앙값, 다지선다 **선택지별 중앙값** 후 합 1로 정규화, 수치·날짜는 라이브러리 중앙값 CDF |
| 뉴스 검색 | 키에 따라 자동 | AskNews 키가 있으면 사용. 없거나 AskNews가 실패하면 검색 없이 진행 |
| 비용 | 상한 없음 | 질문당·실행당 비용 상한, 실행당 질문 수 상한 |
| 실행 주기 | 20분 | 30분 (비공개 저장소의 무료 Actions 시간 안에 들도록) |

비용 상한과 관련해 템플릿·라이브러리 버그 2개를 고쳤다. 자세한 내용은 `budgeted_llm.py` 주석에 있다.
1. 원래 비용 상한은 **경고만 하고 호출을 막지 못한다.** litellm이 사전 확인 단계의 예외를 삼킨다.
2. 원래 비용 집계는 **호출마다 2배로 잡힌다.**

## 어떤 LLM을 쓰나 (자동 선택)
| 순서 | 조건 | 예측 모델 | 답 파서 |
|---|---|---|---|
| 1 | `OPENROUTER_API_KEY` 있음 | `openrouter/anthropic/claude-sonnet-5.5` | `openrouter/openai/gpt-4o-mini` |
| 2 | `ANTHROPIC_API_KEY` 있음 | `anthropic/claude-sonnet-5-5` (adaptive thinking, effort=medium) | `anthropic/claude-haiku-5-5` |
| 3 | 둘 다 없음 | `metaculus/gpt-4o` (Metaculus LLM 프록시) | `metaculus/gpt-4o-mini` |

- 참가 신청 폼으로 받는 **무료 LLM 크레딧은 OpenRouter 키로 나온다.** 템플릿 README에 그렇게 적혀 있다. 그래서 1순위다.
- Metaculus 프록시 모델 문자열(`metaculus/gpt-4o`)은 forecasting-tools 0.3.5 소스의 기본값에서 확인했다. 프록시가 Claude 모델을 허용하는지는 문서로 확인하지 못했다.
- Opus로 바꾸려면 Variables에 `BOT_MODEL=opus`를 넣는다. 그러면 `claude-opus-5.5`(OpenRouter) 또는 `claude-opus-5-5`(Anthropic)를 쓴다. 비용은 약 2배다.
- 강제로 고르려면 `BOT_LLM_PROVIDER=openrouter|anthropic|metaculus_proxy`를 넣는다.

**질문당 예상 비용 [추정]**: Sonnet 5.5 샘플 5회 기준 $0.1~0.4, Opus 5.5는 그 2배. 기본 상한은 질문당 $1, 실행당 $10, 실행당 20문항이다.

---

## 사람이 할 일 (처음 한 번, 약 20분)

### 1. Metaculus 봇 토큰 받기
1. https://www.metaculus.com/futureeval/participate/ 에 접속해 로그인한다.
2. 봇 계정을 만들고 화면에 나오는 **토큰을 복사**해 둔다. 이것이 `METACULUS_TOKEN`이다.
3. 참가 폼 https://forms.gle/aQdYMq9Pisrf1v7d8 의 필수 3문항을 작성한다. **예측을 제출하기 전에 해야 하는 규칙이다.** 같은 폼에서 LLM 크레딧도 신청한다.
4. 크레딧 메일이 오면 그 안의 OpenRouter 키를 복사해 둔다. 이것이 `OPENROUTER_API_KEY`다.

### 2. GitHub 저장소 만들기 (이 `bot/` 폴더가 저장소 루트가 된다)
1. https://github.com/new 에 접속한다.
2. Repository name에 `metaculus-bot`을 입력한다.
3. 공개 범위를 고른다. **Private** 저장소는 Actions 무료 시간이 월 2,000분이고, 30분 주기로 돌리면 약 1,450분을 쓴다. **Public** 저장소는 무료 시간 제한이 없지만 코드가 공개된다.
4. **Create repository**를 클릭한다.
5. 터미널에서 아래 명령을 실행한다. `<내아이디>`는 본인 GitHub 아이디로 바꾼다.
   ```bash
   cd "/Users/seungyunshin/Desktop/codex workspace/ai-income/agent_work/metaculus_bot/bot"
   git init && git add . && git commit -m "Metaculus calibrated bot"
   git branch -M main
   git remote add origin https://github.com/<내아이디>/metaculus-bot.git
   git push -u origin main
   ```
   `.env`는 `.gitignore`에 들어 있어서 올라가지 않는다.

### 3. 토큰 넣는 위치 (GitHub Secrets)
저장소 페이지에서 **Settings → (왼쪽) Secrets and variables → Actions → Secrets 탭 → New repository secret** 순서로 들어간다. 아래 값을 하나씩 추가한다. 이름은 대문자로 정확히 입력한다.

| Name | 필수 | 값 |
|---|---|---|
| `METACULUS_TOKEN` | **필수** | 1단계에서 받은 봇 토큰 |
| `OPENROUTER_API_KEY` | 선택 | Metaculus 크레딧 메일의 OpenRouter 키 |
| `ANTHROPIC_API_KEY` | 선택 | 본인 Claude API 키 (OpenRouter 키가 없을 때 사용) |
| `ASKNEWS_CLIENT_ID`, `ASKNEWS_SECRET` | 선택 | AskNews 키. 없으면 검색 없이 예측한다 |

설정값을 바꾸고 싶으면 같은 화면의 **Variables 탭 → New repository variable**에서 넣는다. 아무것도 넣지 않으면 기본값으로 돈다.

| Name | 기본값 | 의미 |
|---|---|---|
| `BOT_MODEL` | (자동) | `sonnet` / `opus` / `haiku` 또는 전체 모델 문자열 |
| `BOT_LLM_PROVIDER` | `auto` | `openrouter` / `anthropic` / `metaculus_proxy` |
| `BOT_EFFORT` | `medium` | Anthropic 키를 쓸 때의 사고 깊이 (`low`~`max`) |
| `BOT_RESEARCH` | `auto` | `asknews` / `none` |
| `BOT_PREDICTIONS_PER_QUESTION` | `5` | 질문당 샘플 수 (중앙값으로 집계) |
| `BOT_MAX_COST_PER_QUESTION_USD` | `1.0` | 질문당 비용 상한. `0`이면 무제한 |
| `BOT_MAX_COST_PER_RUN_USD` | `10.0` | 실행 1회당 비용 상한. `0`이면 무제한 |
| `BOT_MAX_QUESTIONS_PER_RUN` | `20` | 실행 1회당 질문 수 상한 (FutureEval과 MiniBench 합산) |
| `BOT_PUBLISH` | `true` | `false`면 정기 실행도 제출하지 않음 |

### 4. Actions 켜기
1. 저장소 상단의 **Actions** 탭을 클릭한다.
2. 안내가 나오면 **I understand my workflows, go ahead and enable them**을 클릭한다.

### 5. 테스트 (bot-testing-area, 상금과 무관한 연습장)
1. **Actions → (왼쪽) Test Bot → (오른쪽) Run workflow**를 클릭한다.
2. `dry_run`을 **true**로 고르고 초록색 **Run workflow**를 클릭한다. 단위 테스트와 예측이 돌지만 Metaculus에는 올리지 않는다.
3. 실행이 초록색 체크로 끝나면, 로그 마지막의 `Settings:` 줄과 `Bot produced (dry run) N forecast(s)`를 확인한다.
4. 같은 방법으로 `dry_run`을 **false**로 한 번 더 돌린다. 이번에는 bot-testing-area에 실제로 제출된다.
5. 제출 결과는 Metaculus에서 확인한다. **사용자 이름 → Settings → My Forecasting Bots → Switch to bot account**로 봇 계정으로 바꾼 뒤 로그에 찍힌 질문 링크를 연다. 봇의 이유는 질문 페이지의 `Private` 댓글 탭에 있다.

### 6. 실전
- `Forecast on new AI tournament questions` 워크플로가 **30분마다 자동으로** FutureEval 시즌 토너먼트와 MiniBench의 새 질문에 제출한다. 이미 예측한 질문은 건너뛴다.
- 멈추려면 **Actions → Forecast on new AI tournament questions → 오른쪽 위 `...` → Disable workflow**를 누른다.
- 2주 뒤 MiniBench 결과로 계속할지 판단한다 (`../SETUP.md` 판단 기준).

---

## 로컬에서 테스트하기 (선택)
uv 없이 pip로도 된다. Python 3.11 이상이 필요하다.

**1) 오프라인 단위 테스트**: 키가 필요 없고, 네트워크를 차단한 채로 돈다.
```bash
cd "/Users/seungyunshin/Desktop/codex workspace/ai-income/agent_work/metaculus_bot/bot"
python3.11 -m venv .venv
.venv/bin/pip install "forecasting-tools==0.3.5" "litellm==1.105.1" "asknews>=0.13.11,<0.14" python-dotenv pytest
.venv/bin/python -m pytest tests -q
```

**2) 제출 없이 실제 질문으로 돌려 보기**: LLM 비용이 든다.
```bash
cp .env.example .env    # .env를 열어 METACULUS_TOKEN 등을 채운다
.venv/bin/python main.py --mode test_questions --dry-run
```

`--mode`로 고를 수 있는 값은 `tournament`(기본, FutureEval + MiniBench), `minibench`, `metaculus_cup`, `test_questions`다.

## 파일 구성
| 파일 | 역할 |
|---|---|
| `main.py` | 봇 본체. 템플릿 `main.py`를 개조했다. 다지선다·수치·날짜·조건부 프롬프트는 템플릿 그대로다 |
| `forecast_logic.py` | 이진 프롬프트, 확률 파싱, 클리핑, 중앙값 집계, 질문 수 상한. 네트워크를 쓰지 않는다 |
| `bot_config.py` | 환경변수 읽기와 모델 선택, OpenRouter 가격 등록 |
| `budgeted_llm.py` | 비용 상한이 실제로 호출을 막도록 하는 수정, 이중 집계 수정 |
| `bot_helpers.py` | 템플릿 그대로 (배너 출력) |
| `tests/` | 오프라인 테스트 68개 |
| `.github/workflows/test_bot.yaml` | 수동 테스트 (단위 테스트 → bot-testing-area) |
| `.github/workflows/run_bot_on_tournament.yaml` | 30분마다 실전 제출 |
| `pyproject.toml`, `uv.lock` | 템플릿 그대로. 고치면 `uv sync --locked`가 실패하므로 고치지 않는다 |

## 알려진 한계
- **실제 API 호출은 한 번도 하지 않았다.** OpenRouter, Anthropic, Metaculus 프록시, AskNews 모두 그렇다. 첫 `Test Bot` 실행이 실제 첫 검증이다.
- 비용 상한은 LLM 호출 **직전에** 확인한다. 동시에 나간 샘플 5개는 끝까지 실행되므로 상한을 한 번 정도 넘을 수 있다.
- OpenRouter 경유 Claude에는 thinking/effort 옵션을 따로 보내지 않는다. OpenRouter 기본 동작을 따른다.
- 비용 집계는 litellm 가격표를 쓴다. OpenRouter 가격은 2026-10-26 기준으로 `bot_config.py`에 직접 넣었다. 가격이 바뀌면 그 표를 고친다.
