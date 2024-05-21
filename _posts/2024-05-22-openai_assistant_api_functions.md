---
layout: single
title:  "[Assistants API - 4/4] GPT-4o에서 custom function 사용"
categories: openai
tag: [openai, gpt-4o, assistant, python, function]
toc: true
---

## Assistant + tools
- **code_interpreter**
   - 토큰 길이에 상관 없이, 코드를 생성해서 답변할 수 있는 도구 입니다.
   - 예: 피보나치 수열을 만들어줘 -> 피보나치 수열을 생성하는 함수를 생성한 후 이것의 실행 결과로 답변을 만듦 (cf., Code Interpreter가 아니라면, text 기반의 sequence generation으로 답변을 함)
   - 함수를 생성하는 방식이기 때문에, input, output의 token length 제약을 벗어나서 활용을 할 수 있습니다.
   - 데이터와 그래프 이미지가 포함된 파일을 생성할 수 있습니다.
- **file_search**
   - 우리가 업로드한 자료에서 검색한 이후 답변을 만들어 줍니다. (RAG를 활용하는 것과 비슷)
   - Foundation model을 sFT하는데 걸리는 시간을 생각하면, 대부분의 짧은 배치는 RAG 형태로 구성하고, 긴배치(ex., 1~2주일에 한번)는 sFT를 하는 방식이 적절할 것으로 보입니다.
   - File을 Vector Store에 넣어두고 Assistant가 이를 활용할 수 있습니다.
   - 파일 하나당 최대 크기 512MB, 최대 토큰수 5,000,000까지 지원합니다.
   - [지원하는 파일 타입](https://platform.openai.com/docs/assistants/tools/file-search/supported-files)
   - [Known Limitation](https://platform.openai.com/docs/assistants/tools/file-search/how-it-works)
   - 아직 이미지 파일이나, csv, jsonl을 지원하지 않는데 추후 업데이트 될 것으로 보입니다. (24-05-21 기준)
- **[이번 내용] function**
   - 우리가 직접 만든 함수를 사용할 수 있게 합니다.
   - 함수를 assistant에 넘길때는 Json Schema를 만들어서 넘겨야 합니다.
   - Json Schema는 변환을 해 주는 GPTs가 있으나, 대략 input, output 정의하면 어렵지 않게 만들 수 있습니다.
   - Assistant API는 실행 중에 함수를 호출할 때 실행을 일시 중지하며, 함수 호출 결과를 다시 제공하여 Run 실행을 계속할 수 있습니다. (이는 사용자 피드백을 받아 재개할 수 있다는 의미입니다.)

## 기본 작업
- Open AI 라이브러리 설치
- Dotenv 설치
- Google Drive 연결
- API Key 로드
- Open AI 클라이언트 객체 생성


```python
# 실행을 위한 기본 설치 및 로드
!pip install --upgrade openai -q
!pip show openai | grep Version
!pip install -U python-dotenv
from dotenv import load_dotenv
import json
from openai import OpenAI
import os
from google.colab import drive
```

    [?25l     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/320.6 kB[0m [31m?[0m eta [36m-:--:--[0m
[2K     [91m━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━[0m [32m143.4/320.6 kB[0m [31m4.3 MB/s[0m eta [36m0:00:01[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m320.6/320.6 kB[0m [31m5.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m75.6/75.6 kB[0m [31m8.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m77.9/77.9 kB[0m [31m9.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.3/58.3 kB[0m [31m5.5 MB/s[0m eta [36m0:00:00[0m
    [?25hVersion: 1.30.1
    Collecting python-dotenv
      Downloading python_dotenv-1.0.1-py3-none-any.whl (19 kB)
    Installing collected packages: python-dotenv
    Successfully installed python-dotenv-1.0.1
    


```python
drive.mount('/content/drive', force_remount=True)

load_dotenv(
    dotenv_path='drive/MyDrive/AUTH/.env',
    verbose=True,
)
api_key = os.environ.get("OPEN_API_KEY")
client = OpenAI(api_key=api_key)

# 객체 내용 출력 유틸
def show_json(obj):
  display(json.loads(obj.model_dump_json()))

def _get_response(thread_id):
    return client.beta.threads.messages.list(thread_id=thread_id, order="asc")

# Thread message 출력 유틸
def print_message(thread_id):
    for res in _get_response(thread_id):
        print(f"[{res.role.upper()}]\n{res.content[0].text.value}\n")
```

    Mounted at /content/drive
    

## 사용자 정의 함수 작성
퀴즈 문제 출제 챗봇의 예시입니다. 아래와 같은 형태입니다.
* title
* questions
   * question_text
   * choices: ["선택지1", "선택지2", "선택지3", "선택지4"]


```python
def display_quiz(title, questions, show_numeric=False):
    print(f"제목: {title}\n")
    responses = []

    for q in questions:
        # 질문을 출력합니다.
        print(q["question_text"])
        response = ""

        # 각 선택지를 출력합니다.
        for i, choice in enumerate(q["choices"]):
            if show_numeric:
                print(f"{i+1} {choice}")
            else:
                print(f"{choice}")

        response = input("정답을 선택해 주세요: ")
        responses.append(response)
        print()

    return responses
```


```python
# 문제 구문을 주어 테스트 해 봅시다.
responses = display_quiz(
    "Sample Quiz",
    [
        {
            "question_text": "제일 좋아하는 색상은 무엇입니까?",
            "choices": ["빨강", "파랑", "초록", "노랑"],
        },
        {
            "question_text": "제일 좋아하는 동물은 무엇입니까?",
            "choices": ["강아지", "고양이", "햄스터", "토끼"],
        },
    ],
    show_numeric=True,
)
print("Responses:", responses)
```

    제목: Sample Quiz
    
    제일 좋아하는 색상은 무엇입니까?
    1 빨강
    2 파랑
    3 초록
    4 노랑
    정답을 선택해 주세요: 1
    
    제일 좋아하는 동물은 무엇입니까?
    1 강아지
    2 고양이
    3 햄스터
    4 토끼
    정답을 선택해 주세요: 1
    
    Responses: ['1', '1']
    

## Function Schema 작성
- RAML 형식과 비슷한 느낌입니다.
- ChatGPT에게 Assistant API에 있는 예제로 함수 -> Schema를 보여주고, 내 함수를 똑같이 바꿔달라고 하면 바꿔 줍니다.
- 아니면 아래 구문을 보시고 학습하여 바꾸셔도 될 것 같습니다.


```python
function_schema = {
    "name": "generate_quiz",
    "description": "Generate a quiz to the student, and returns the student's response. A single quiz has multiple questions.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "questions": {
                "type": "array",
                "description": "An array of questions, each with a title and multiple choice options.",
                "items": {
                    "type": "object",
                    "properties": {
                        "question_text": {"type": "string"},
                        "choices": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["question_text", "choices"],
                },
            },
        },
        "required": ["title", "questions"],
    },
}
```

## Function Assistant 생성
- file_search와 function 도구를 같이 활용하는 assistant를 생성합니다.
- [file_search 실습](https://torajim.github.io/openai/openai_assistant_api_file_search/) 에서 만든 "수원도시계획" 파일을 업로드 했던 vector_store를 연결해 봅니다. (vector_store_id는 각자의 아이디로 바꾸셔야 합니다. )
- function은 위에서 정의한 function_schema를 넘겨줬습니다.


```python
# 퀴즈를 출제하는 역할을 하는 챗봇을 생성합니다.
assistant = client.beta.assistants.create(
    name="Quiz Generator",
    instructions="You are an expert in generating multiple choice quizzes. Create quizzes based on uploaded files.",
    model="gpt-4o",
    tools=[
        {"type": "file_search"},
        {"type": "function", "function": function_schema},
    ],
    tool_resources={"file_search":{"vector_store_ids": ["vs_GU0BVMfMjeRwxMJByreSLLZR"]}},
)

ASSISTANT_ID = assistant.id
# 생성된 챗봇의 정보를 JSON 형태로 출력합니다.
show_json(assistant)
```


    {'id': 'asst_pgfKVlv57nTH1v0AdXYkDrxJ',
     'created_at': 1716304146,
     'description': None,
     'instructions': 'You are an expert in generating multiple choice quizzes. Create quizzes based on uploaded files.',
     'metadata': {},
     'model': 'gpt-4o',
     'name': 'Quiz Generator',
     'object': 'assistant',
     'tools': [{'type': 'file_search'},
      {'function': {'name': 'generate_quiz',
        'description': "Generate a quiz to the student, and returns the student's response. A single quiz has multiple questions.",
        'parameters': {'type': 'object',
         'properties': {'title': {'type': 'string'},
          'questions': {'type': 'array',
           'description': 'An array of questions, each with a title and multiple choice options.',
           'items': {'type': 'object',
            'properties': {'question_text': {'type': 'string'},
             'choices': {'type': 'array', 'items': {'type': 'string'}}},
            'required': ['question_text', 'choices']}}},
         'required': ['title', 'questions']}},
       'type': 'function'}],
     'response_format': 'auto',
     'temperature': 1.0,
     'tool_resources': {'code_interpreter': None,
      'file_search': {'vector_store_ids': ['vs_GU0BVMfMjeRwxMJByreSLLZR']}},
     'top_p': 1.0}


## Thread 생성 및 실행
- Vector store를 활용하여 퀴즈를 만들어 보라고 시켜 보겠습니다.
- Thread, Message, Run 차례로 생성 및 실행


```python
thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
    thread_id = thread.id,
    role = "user",
    content= """
       3개의 객관식 퀴즈(multiple choice questions)를 만들어 주세요.
       객관식 퀴즈의 선택지에 번호를 표기해주세요. 1~4까지 숫자로 시작하여야 합니다.
       퀴즈는 내가 업로드한 파일에 관한 내용이어야 합니다.
       내가 제출한 responses에 대한 피드백을 주세요.
       내가 기입한 답, 정답, 제출한 답이 오답이라면 오답에 대한 피드백을 모두 포함해야 합니다.
       모든 내용은 한글로 작성해 주세요.
    """,
)

run = client.beta.threads.runs.create_and_poll(
    thread_id = thread.id,
    assistant_id = assistant.id,
)
show_json(run)
```


    {'id': 'run_Ii9CSYUH34jiNKGptFYAxY8P',
     'assistant_id': 'asst_pgfKVlv57nTH1v0AdXYkDrxJ',
     'cancelled_at': None,
     'completed_at': None,
     'created_at': 1716305677,
     'expires_at': 1716306277,
     'failed_at': None,
     'incomplete_details': None,
     'instructions': 'You are an expert in generating multiple choice quizzes. Create quizzes based on uploaded files.',
     'last_error': None,
     'max_completion_tokens': None,
     'max_prompt_tokens': None,
     'metadata': {},
     'model': 'gpt-4o',
     'object': 'thread.run',
     'required_action': {'submit_tool_outputs': {'tool_calls': [{'id': 'call_Pbq5vKkKa18RbzA4wVUm0Q0p',
         'function': {'arguments': '{"title":"수원도시기본계획 퀴즈","questions":[{"question_text":"수원시 도시기본계획의 배경 및 목적에 대한 설명으로 옳은 것을 고르세요.","choices":["1. 수원시의 역사적 의의를 높이기 위해","2. 수원시의 생활환경을 개선하기 위해","3. 수원시의 경제성장을 촉진하기 위해","4. 수원시의 미관을 향상시키기 위해"]},{"question_text":"수원시의 SWOT 분석 중 기회(Opportunity)에 해당하지 않는 것은?","choices":["1. 교통체계의 개선","2. 주변 도시와의 연계성 강화","3. 수원의 역사 자원 활용","4. 인구 증가로 인한 환경 부담"]},{"question_text":"수원 도시기본계획의 목표연도는 언제인가요?","choices":["1. 2025년","2. 2030년","3. 2035년","4. 2040년"]}]}',
          'name': 'generate_quiz'},
         'type': 'function'}]},
      'type': 'submit_tool_outputs'},
     'response_format': 'auto',
     'started_at': 1716305677,
     'status': 'requires_action',
     'thread_id': 'thread_hHx3HYHu2gTluA9dpMqHghYR',
     'tool_choice': 'auto',
     'tools': [{'type': 'file_search'},
      {'function': {'name': 'generate_quiz',
        'description': "Generate a quiz to the student, and returns the student's response. A single quiz has multiple questions.",
        'parameters': {'type': 'object',
         'properties': {'title': {'type': 'string'},
          'questions': {'type': 'array',
           'description': 'An array of questions, each with a title and multiple choice options.',
           'items': {'type': 'object',
            'properties': {'question_text': {'type': 'string'},
             'choices': {'type': 'array', 'items': {'type': 'string'}}},
            'required': ['question_text', 'choices']}}},
         'required': ['title', 'questions']}},
       'type': 'function'}],
     'truncation_strategy': {'type': 'auto', 'last_messages': None},
     'usage': None,
     'temperature': 1.0,
     'top_p': 1.0,
     'tool_resources': {}}


## Run의 status 확인
- requires_action 임을 확인합니다. (사용자가 질문에 대한 답을 해야 하기 때문에...)

- TeddyNote said,
   - 퀴즈 생성기가 퀴즈 문제를 만들고 난 다음에 사용자의 정답 제출을 기다리고 있습니다. 이때도 required_action 이 호출될 수 있습니다.
   - 비록, 함수의 argument 입력은 필요 없지만 프롬프트(prompt) 에서 제출한 정답에 대한 피드백 을 요청했기 때문에 required_action 이 호출된 것입니다.
   - 그럼, 실행한 Run 의 required_action 을 출력하여 세부 정보를 확인해 보겠습니다.


```python
print(run.status)
```

    requires_action
    


```python
# tool_calls를 출력합니다.
tool_call = run.required_action.submit_tool_outputs.tool_calls[0]
name = tool_call.function.name
arguments = json.loads(tool_call.function.arguments)

# tool_calls 정보를 출력합니다.
print(f"[Function]\n{name}\n")
print(f"[Arguments]\n")
print(json.dumps(arguments, indent=4, ensure_ascii=False))
```

    [Function]
    generate_quiz
    
    [Arguments]
    
    {
        "title": "수원도시기본계획 퀴즈",
        "questions": [
            {
                "question_text": "수원시 도시기본계획의 배경 및 목적에 대한 설명으로 옳은 것을 고르세요.",
                "choices": [
                    "1. 수원시의 역사적 의의를 높이기 위해",
                    "2. 수원시의 생활환경을 개선하기 위해",
                    "3. 수원시의 경제성장을 촉진하기 위해",
                    "4. 수원시의 미관을 향상시키기 위해"
                ]
            },
            {
                "question_text": "수원시의 SWOT 분석 중 기회(Opportunity)에 해당하지 않는 것은?",
                "choices": [
                    "1. 교통체계의 개선",
                    "2. 주변 도시와의 연계성 강화",
                    "3. 수원의 역사 자원 활용",
                    "4. 인구 증가로 인한 환경 부담"
                ]
            },
            {
                "question_text": "수원 도시기본계획의 목표연도는 언제인가요?",
                "choices": [
                    "1. 2025년",
                    "2. 2030년",
                    "3. 2035년",
                    "4. 2040년"
                ]
            }
        ]
    }
    

## GPT가 생성한 문제를 기존 custom function에 대입하여 답변 생성
- 틀린 문제에 피드백을 받기로 했기 때문에, 일부러 오답을 넣어 봤습니다.


```python
responses = display_quiz(arguments["title"], arguments["questions"])
print("기입한 답(순서대로)")
print(responses)
```

    제목: 수원도시기본계획 퀴즈
    
    수원시 도시기본계획의 배경 및 목적에 대한 설명으로 옳은 것을 고르세요.
    1. 수원시의 역사적 의의를 높이기 위해
    2. 수원시의 생활환경을 개선하기 위해
    3. 수원시의 경제성장을 촉진하기 위해
    4. 수원시의 미관을 향상시키기 위해
    정답을 선택해 주세요: 2
    
    수원시의 SWOT 분석 중 기회(Opportunity)에 해당하지 않는 것은?
    1. 교통체계의 개선
    2. 주변 도시와의 연계성 강화
    3. 수원의 역사 자원 활용
    4. 인구 증가로 인한 환경 부담
    정답을 선택해 주세요: 4
    
    수원 도시기본계획의 목표연도는 언제인가요?
    1. 2025년
    2. 2030년
    3. 2035년
    4. 2040년
    정답을 선택해 주세요: 1
    
    기입한 답(순서대로)
    ['2', '4', '1']
    

## Run에 tool_output을 다시 submit 합니다.
client.beta.threads.runs.submit_tool_outputs 함수는 우리의 입력을 다시 제출할 수 있도록 해줍니다.

참고

- tool_call_id: 대기중인 tool_call 의 id 를 입력합니다.
- output: 사용자가 입력할 내용을 json.dumps 로 json 형식으로 변환하여 제출합니다.


```python
run = client.beta.threads.runs.submit_tool_outputs_and_poll(
    thread_id=thread.id,
    run_id=run.id,
    tool_outputs=[
        {
            "tool_call_id": tool_call.id,
            "output": json.dumps(responses),
        }
    ],
)
show_json(run)
```


    {'id': 'run_Ii9CSYUH34jiNKGptFYAxY8P',
     'assistant_id': 'asst_pgfKVlv57nTH1v0AdXYkDrxJ',
     'cancelled_at': None,
     'completed_at': 1716305756,
     'created_at': 1716305677,
     'expires_at': None,
     'failed_at': None,
     'incomplete_details': None,
     'instructions': 'You are an expert in generating multiple choice quizzes. Create quizzes based on uploaded files.',
     'last_error': None,
     'max_completion_tokens': None,
     'max_prompt_tokens': None,
     'metadata': {},
     'model': 'gpt-4o',
     'object': 'thread.run',
     'required_action': None,
     'response_format': 'auto',
     'started_at': 1716305743,
     'status': 'completed',
     'thread_id': 'thread_hHx3HYHu2gTluA9dpMqHghYR',
     'tool_choice': 'auto',
     'tools': [{'type': 'file_search'},
      {'function': {'name': 'generate_quiz',
        'description': "Generate a quiz to the student, and returns the student's response. A single quiz has multiple questions.",
        'parameters': {'type': 'object',
         'properties': {'title': {'type': 'string'},
          'questions': {'type': 'array',
           'description': 'An array of questions, each with a title and multiple choice options.',
           'items': {'type': 'object',
            'properties': {'question_text': {'type': 'string'},
             'choices': {'type': 'array', 'items': {'type': 'string'}}},
            'required': ['question_text', 'choices']}}},
         'required': ['title', 'questions']}},
       'type': 'function'}],
     'truncation_strategy': {'type': 'auto', 'last_messages': None},
     'usage': {'completion_tokens': 635,
      'prompt_tokens': 25300,
      'total_tokens': 25935},
     'temperature': 1.0,
     'top_p': 1.0,
     'tool_resources': {}}



```python
print_message(thread.id)
```

    [USER]
    
           3개의 객관식 퀴즈(multiple choice questions)를 만들어 주세요.
           객관식 퀴즈의 선택지에 번호를 표기해주세요. 1~4까지 숫자로 시작하여야 합니다.
           퀴즈는 내가 업로드한 파일에 관한 내용이어야 합니다.
           내가 제출한 responses에 대한 피드백을 주세요.
           내가 기입한 답, 정답, 제출한 답이 오답이라면 오답에 대한 피드백을 모두 포함해야 합니다.
           모든 내용은 한글로 작성해 주세요.
        
    
    [ASSISTANT]
    ### 수원도시기본계획 퀴즈 피드백
    
    #### 문제 1
    **질문:** 수원시 도시기본계획의 배경 및 목적에 대한 설명으로 옳은 것을 고르세요.
    
    **선택지:**
    1. 수원시의 역사적 의의를 높이기 위해
    2. 수원시의 생활환경을 개선하기 위해
    3. 수원시의 경제성장을 촉진하기 위해
    4. 수원시의 미관을 향상시키기 위해
    
    **제출한 답:** 2
    **정답:** 2
    
    잘 선택하셨습니다! 수원시 도시기본계획의 주요 목적 중 하나는 생활환경을 개선하는 것입니다.
    
    #### 문제 2
    **질문:** 수원시의 SWOT 분석 중 기회(Opportunity)에 해당하지 않는 것은?
    
    **선택지:**
    1. 교통체계의 개선
    2. 주변 도시와의 연계성 강화
    3. 수원의 역사 자원 활용
    4. 인구 증가로 인한 환경 부담
    
    **제출한 답:** 4
    **정답:** 4
    
    정확하게 답하셨습니다. 인구 증가로 인한 환경 부담은 SWOT 분석에서 '위협(Threat)'에 해당하는 요소입니다.
    
    #### 문제 3
    **질문:** 수원 도시기본계획의 목표연도는 언제인가요?
    
    **선택지:**
    1. 2025년
    2. 2030년
    3. 2035년
    4. 2040년
    
    **제출한 답:** 1
    **정답:** 2
    
    오답입니다. 수원 도시기본계획의 목표연도는 2030년입니다【4:0†source】.
    
    

## Reference
- [Function calling - Assistants API(Beta) - Open AI](https://platform.openai.com/docs/assistants/tools/function-calling)
- [EP02. #openai의 새로운 기능 #assistant API 3가지 도구 활용법 - 테디노트](https://www.youtube.com/watch?v=BMW1NJkL7Ks)
- [소스 코드 참조 - Colab](https://colab.research.google.com/drive/19duWIYLHMi6lF4-W4f-XsaKV3f304DQP?usp=sharing)
