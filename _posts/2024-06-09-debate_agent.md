---
layout: single
title:  "Langchain, tavily, openai을 이용한 웹 서치를 하는 토론 에이전트(agent)구현 (동탄-광교 어디가 살기 좋은가?)"
categories: llm
tag: [openai, gpt-4o, debate, python, agent, langchain, tavily]
toc: true
---

```python
!pip install -U langchain
!pip install -U langchain-openai
!pip install -U langchain_community
!pip install -U langchainhub
!pip install -U python-dotenv
!pip install --upgrade openai -q
```
   

### Open API key load


```python
from google.colab import drive
from dotenv import load_dotenv
import os

drive.mount('/content/drive', force_remount=True)

load_dotenv(
    dotenv_path='drive/MyDrive/AUTH/.env',
    verbose=True,
)
openai_api_key = os.environ.get("OPEN_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")
print(openai_api_key[0:5])
print(tavily_api_key[0:5])
```

    Mounted at /content/drive
    sk-vs
    tvly-
    

## DialogueAgent 및 DialogueSimulator
  테디노트 영상에서 구현한 다중 에이전트 시뮬레이션을 구현하는 방법을 재현
  Agent들은 (1) 말하기 전에 생각하기, (2) 대화를 종료하기 등을 선보인다.

  * DialogueAgent : 토론자
  * DialogueSimulator: 토론자간의 대화 조율 역할


### Import related modules


```python
import functools
import random
from collections import OrderedDict
from typing import Callable, List

import tenacity
from langchain.output_parsers import RegexParser
from langchain.prompts import (
    PromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
```

### DialogueAgent
* ```send``` 메서드는 현재까지의 대화 기록과 에이전트의 접두사를 사용하여 채팅 모델에 메시지를 전달하고 응답을 반환합니다.
* ```receive``` 메서드는 다른 에이전트가 보낸 메시지를 대화 기록에 추가합니다.


```python
class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        message = self.model.invoke(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")
```

### DialogueSimulator
* `inject` 메서드는 주어진 이름(`name`)과 메시지(`message`)로 대화를 시작하고, 모든 에이전트가 해당 메시지를 받도록 합니다.
* `step` 메서드는 다음 발언자를 선택하고, 해당 발언자가 메시지를 보내면 모든 에이전트가 메시지를 받도록 합니다. 그리고 현재 단계를 증가시킵니다.
여러 에이전트 간의 대화를 시뮬레이션하는 기능을 제공합니다.

`DialogueAgent`는 개별 에이전트를 나타내며, `DialogueSimulator`는 에이전트들 간의 대화를 조정하고 관리합니다.


```python
class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        # 에이전트 목록을 설정합니다.
        self.agents = agents
        # 시뮬레이션 단계를 초기화합니다.
        self._step = 0
        # 다음 발언자를 선택하는 함수를 설정합니다.
        self.select_next_speaker = selection_function

    def reset(self):
        # 모든 에이전트를 초기화합니다.
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        name 의 message 로 대화를 시작합니다.
        """
        # 모든 에이전트가 메시지를 받습니다.
        for agent in self.agents:
            agent.receive(name, message)

        # 시뮬레이션 단계를 증가시킵니다.
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. 다음 발언자를 선택합니다.
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. 다음 발언자에게 메시지를 전송합니다.
        message = speaker.send()

        # 3. 모든 에이전트가 메시지를 받습니다.
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. 시뮬레이션 단계를 증가시킵니다.
        self._step += 1

        # 발언자의 이름과 메시지를 반환합니다.
        return speaker.name, message
```

### DialogueAgentWithTools
`DialogueAgent`를 확장하여 도구를 사용할 수 있도록 `DialogueAgentWithTools` 클래스를 정의합니다.

* `DialogueAgentWithTools` 클래스는 `DialogueAgent` 클래스를 상속받아 구현되었습니다.
* `send` 메서드는 에이전트가 메시지를 생성하고 반환하는 역할을 합니다.
* `create_openai_tools_agent` 함수를 사용하여 에이전트 체인을 초기화합니다.
  * 초기화시 에이전트가 사용할 도구(tools) 를 정의합니다.


```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub


class DialogueAgentWithTools(DialogueAgent):
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
        tools,
    ) -> None:
        # 부모 클래스의 생성자를 호출합니다.
        super().__init__(name, system_message, model)
        # 주어진 도구 이름과 인자를 사용하여 도구를 로드합니다.
        self.tools = tools

    def send(self) -> str:
        """
        메시지 기록에 챗 모델을 적용하고 메시지 문자열을 반환합니다.
        """
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_openai_tools_agent(self.model, self.tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, tools=self.tools, verbose=False)
        # AI 메시지를 생성합니다.
        message = AIMessage(
            content=agent_executor.invoke(
                {
                    "input": "\n".join(
                        [self.system_message.content]
                        + [self.prefix]
                        + self.message_history
                    )
                }
            )["output"]
        )

        # 생성된 메시지의 내용을 반환합니다.
        return message.content
```

### 도구 설정
여기서는 인터넷 검색 도구만을 활용해 봅니다.


```python
# TavilySearchResults 클래스를 langchain_community.tools.tavily_search 모듈에서 가져옵니다.
from langchain_community.tools.tavily_search import TavilySearchResults

# TavilySearchResults 클래스의 인스턴스를 생성합니다
# k=10은 검색 결과를 10개까지 가져오겠다는 의미입니다
search = TavilySearchResults(k=20, tavily_api_key=tavily_api_key)

names_search = {
    "동탄 시민 연합": [search],  # 동탄 시민 연합 도구
    "광교 시민 연합": [search],  # 광교 시민 연합 도구
}
# 토론 주제 선정
topic = "2024년 현재, 동탄과 광교는 어디가 더 살기 좋은 곳인가?"
word_limit = 100  # 작업 브레인스토밍을 위한 단어 제한
```

### LLM을 활용하여 주제 설명에 세부 내용 추가하기
LLM(Large Language Model)을 사용하여 주어진 주제에 대한 설명을 보다 상세하게 만들 수 있습니다.

이를 위해서는 먼저 주제에 대한 간단한 설명이나 개요를 LLM에 입력으로 제공합니다. 그런 다음 LLM에게 해당 주제에 대해 더 자세히 설명해줄 것을 요청합니다.

LLM은 방대한 양의 텍스트 데이터를 학습했기 때문에, 주어진 주제와 관련된 추가적인 정보와 세부 사항을 생성해낼 수 있습니다. 이를 통해 초기의 간단한 설명을 보다 풍부하고 상세한 내용으로 확장할 수 있습니다.

* 주어진 대화 주제(topic)와 참가자(names)를 기반으로 대화에 대한 설명(conversation_description)을 생성합니다.
* agent_descriptor_system_message 는 대화 참가자에 대한 설명을 추가할 수 있다는 내용의 SystemMessage입니다.
* generate_agent_description 함수는 각 참가자(name)에 대하여 LLM 이 생성한 설명을 생성합니다.
   * agent_specifier_prompt 는 대화 설명과 참가자 이름, 단어 제한(word_limit)을 포함하는 HumanMessage로 구성됩니다.
   * ChatOpenAI 모델을 사용하여 agent_specifier_prompt 를 기반으로 참가자에 대한 설명(agent_description)을 생성합니다.


```python
conversation_description = f"""Here is the topic of conversation: {topic}
The participants are: {', '.join(names_search.keys())}"""

agent_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of the conversation participant."
)


def generate_agent_description(name):
    agent_specifier_prompt = [
        agent_descriptor_system_message,
        HumanMessage(
            content=f"""{conversation_description}
            Please reply with a description of {name}, in {word_limit} words or less in expert tone.
            Speak directly to {name}.
            Give them a point of view.
            Do not add anything else. Answer in KOREAN."""
        ),
    ]
    # ChatOpenAI를 사용하여 에이전트 설명을 생성합니다.
    agent_description = ChatOpenAI(temperature=0, api_key=openai_api_key)(agent_specifier_prompt).content
    return agent_description


# 각 참가자의 이름에 대한 에이전트 설명을 생성합니다.
agent_descriptions = {name: generate_agent_description(name) for name in names_search}

# 생성한 에이전트 설명을 출력합니다.
agent_descriptions
```

    /usr/local/lib/python3.10/dist-packages/langchain_core/_api/deprecation.py:119: LangChainDeprecationWarning: The method `BaseChatModel.__call__` was deprecated in langchain-core 0.1.7 and will be removed in 0.3.0. Use invoke instead.
      warn_deprecated(
    




    {'동탄 시민 연합': '동탄 시민 연합은 동탄 지역의 주요 이해관계자들로 구성된 단체로, 동탄의 발전과 주거 환경 향상을 목표로 활동합니다. 동탄을 중심으로 한 도시 계획 및 인프라 구축에 대한 전문적인 지식과 열정을 가지고 있습니다. 동탄을 지지하는 입장에서 광교와의 비교를 진행할 것입니다.',
     '광교 시민 연합': '광교 시민 연합은 광교의 발전과 주변 환경을 중시하는 시민들로 구성되어 있습니다. 광교를 더욱 발전시키고 지역 사회에 긍정적인 영향을 미치기 위해 노력하고 계십니다. 동탄과의 비교에서 광교의 장점을 강조하며, 지역의 매력을 알리는 역할을 하고 있습니다.'}



생성된 내용 말고, 직접 agent_descriptions를 추가할 수 있습니다.


```python
agent_descriptions = {
    "동탄 시민 연합": "동탄 시민 연합은 동탄 지역의 주요 이해관계자들로 구성된 단체로, 동탄의 발전과 주거 환경 향상을 목표로 활동합니다. 동탄을 중심으로 한 도시 계획 및 인프라 구축에 대한 전문적인 지식과 열정을 가지고 있습니다. 동탄을 지지하는 입장에서 광교와의 비교를 진행할 것입니다. 주로 광교에 없는 SRT역, GTX-A역, 광교보다 많은 인구수, 인덕원선 예정, 광역 비즈니스 콤플렉스, 호수공원 분수쇼, 시범단지 교육환경 등을 장점으로 내세우고 있습니다.",
    "광교 시민 연합": "광교 시민 연합은 광교의 발전과 주변 환경을 중시하는 시민들로 구성되어 있습니다. 광교를 더욱 발전시키고 지역 사회에 긍정적인 영향을 미치기 위해 노력하고 계십니다. 동탄과의 비교에서 광교의 장점을 강조하며, 지역의 매력을 알리는 역할을 하고 있습니다. 주로 동탄보다 가까운 강남 접근성, 신분당선으로 인한 판교, 분당 접근성, 평균적으로 높은 집값, 좀더 좋은 호수공원, 대학교, 법원, 경기융합타운등을 통한 높은 생활 수준등을 장점으로 내세우고 있습니다. ",
}
```

### 전역 System Message 설정
System message는 대화형 AI 시스템에서 사용자의 입력에 앞서 시스템이 생성하는 메시지입니다.

이러한 메시지는 대화의 맥락을 설정하고, 사용자에게 각 에이전트의 입장과 목적 을 알려주는 역할을 합니다.

효과적인 system message를 작성하면 사용자와의 상호작용을 원활하게 하고, 대화의 질을 높일 수 있습니다.

프롬프트 설명

* 에이전트의 이름과 설명을 알립니다.
* 에이전트는 도구를 사용하여 정보를 찾고 대화 상대방의 주장을 반박해야 합니다.
* 에이전트는 출처를 인용해야 하며, 가짜 인용을 하거나 찾아보지 않은 출처를 인용해서는 안 됩니다.
* 에이전트는 자신의 관점에서 말을 마치는 즉시 대화를 중단해야 합니다.


```python
def generate_system_message(name, description, tools):
    return f"""{conversation_description}

Your name is {name}.

Your description is as follows: {description}

Your goal is to persuade your conversation partner of your point of view.

DO look up information with your tool to refute your partner's claims.
DO cite your sources.

DO NOT fabricate fake citations.
DO NOT cite any source that you did not look up.

DO NOT restate something that has already been said in the past.
DO NOT add anything else.

Stop speaking the moment you finish speaking from your perspective.

Answer in KOREAN.
"""

agent_system_messages = {
    name: generate_system_message(name, description, tools)
    for (name, tools), description in zip(names_search.items(), agent_descriptions.values())
}

# 에이전트 시스템 메시지를 순회합니다.
for name, system_message in agent_system_messages.items():
    # 에이전트의 이름을 출력합니다.
    print(name)
    # 에이전트의 시스템 메시지를 출력합니다.
    print(system_message)
```

    동탄 시민 연합
    Here is the topic of conversation: 2024년 현재, 동탄과 광교는 어디가 더 살기 좋은 곳인가?
    The participants are: 동탄 시민 연합, 광교 시민 연합
        
    Your name is 동탄 시민 연합.
    
    Your description is as follows: 동탄 시민 연합은 동탄 지역의 주요 이해관계자들로 구성된 단체로, 동탄의 발전과 주거 환경 향상을 목표로 활동합니다. 동탄을 중심으로 한 도시 계획 및 인프라 구축에 대한 전문적인 지식과 열정을 가지고 있습니다. 동탄을 지지하는 입장에서 광교와의 비교를 진행할 것입니다. 주로 광교에 없는 SRT역, GTX-A역, 광교보다 많은 인구수, 인덕원선 예정, 광역 비즈니스 콤플렉스, 호수공원 분수쇼, 시범단지 교육환경 등을 장점으로 내세우고 있습니다.
    
    Your goal is to persuade your conversation partner of your point of view.
    
    DO look up information with your tool to refute your partner's claims.
    DO cite your sources.
    
    DO NOT fabricate fake citations.
    DO NOT cite any source that you did not look up.
    
    DO NOT restate something that has already been said in the past.
    DO NOT add anything else.
    
    Stop speaking the moment you finish speaking from your perspective.
    
    Answer in KOREAN.
    
    광교 시민 연합
    Here is the topic of conversation: 2024년 현재, 동탄과 광교는 어디가 더 살기 좋은 곳인가?
    The participants are: 동탄 시민 연합, 광교 시민 연합
        
    Your name is 광교 시민 연합.
    
    Your description is as follows: 광교 시민 연합은 광교의 발전과 주변 환경을 중시하는 시민들로 구성되어 있습니다. 광교를 더욱 발전시키고 지역 사회에 긍정적인 영향을 미치기 위해 노력하고 계십니다. 동탄과의 비교에서 광교의 장점을 강조하며, 지역의 매력을 알리는 역할을 하고 있습니다. 주로 동탄보다 가까운 강남 접근성, 신분당선으로 인한 판교, 분당 접근성, 평균적으로 높은 집값, 좀더 좋은 호수공원, 대학교, 법원, 경기융합타운등을 통한 높은 생활 수준등을 장점으로 내세우고 있습니다. 
    
    Your goal is to persuade your conversation partner of your point of view.
    
    DO look up information with your tool to refute your partner's claims.
    DO cite your sources.
    
    DO NOT fabricate fake citations.
    DO NOT cite any source that you did not look up.
    
    DO NOT restate something that has already been said in the past.
    DO NOT add anything else.
    
    Stop speaking the moment you finish speaking from your perspective.
    
    Answer in KOREAN.
    
    

`topic_specifier_prompt`를 정의하여 주어진 주제를 더 구체화하는 프롬프트를 생성합니다.

* temperature 를 조절하여 더 다양한 주제를 생성할 수 있습니다.


```python
topic_specifier_prompt = [
    # 주제를 더 구체적으로 만들 수 있습니다.
    SystemMessage(content="You can make a topic more specific."),
    HumanMessage(
        content=f"""{topic}

        You are the moderator.
        Please make the topic more specific.
        Please reply with the specified quest in 100 words or less.
        Speak directly to the participants: {*names_search,}.
        Do not add anything else.
        Answer in Korean."""  # 다른 것은 추가하지 마세요.
    ),
]
# 구체화된 주제를 생성합니다.
specified_topic = ChatOpenAI(temperature=1.0, api_key=openai_api_key)(topic_specifier_prompt).content

print(f"Original topic:\n{topic}\n")  # 원래 주제를 출력합니다.
print(f"Detailed topic:\n{specified_topic}\n")  # 구체화된 주제를 출력합니다.
```

    Original topic:
    2024년 현재, 동탄과 광교는 어디가 더 살기 좋은 곳인가?
    
    Detailed topic:
    어디가 더 살기 좋은 곳인 가? 광교와 동탄을 비교하여 교육 시설과 교통 편의성 측면에서 어느 곳이 더 우수한가요?
    
    

아래와 같이 직접 지정하는 것도 가능합니다.


```python
# 직접 세부 주제 설정
specified_topic = "2024년 현재, 4인 가족은 동탄과 광교 중 어디에서 더 살기 편리한가요?"
```

### 토론 Loop
토론 루프는 프로그램의 핵심 실행 부분으로, 주요 작업이 반복적으로 수행되는 곳입니다.

* 여기서 주요 작업은 각 에이전트의 메시지 청취 -> 도구를 활용하여 근거 탐색 -> 반박 의견 제시 등을 포함합니다.


```python
agents = [
    DialogueAgentWithTools(
        name=name,
        system_message=SystemMessage(content=system_message),
        model=ChatOpenAI(model_name="gpt-4o", temperature=0.5, api_key=openai_api_key),
        tools=tools,
    )
    for (name, tools), system_message in zip(
        names_search.items(), agent_system_messages.values()
    )
]

agents
```




    [<__main__.DialogueAgentWithTools at 0x787317f632b0>,
     <__main__.DialogueAgentWithTools at 0x787317f63a00>]



`select_next_speaker` 함수는 다음 발언자를 선택하는 역할을 합니다.


```python
def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    # 다음 발언자를 선택합니다.
    # step을 에이전트 수로 나눈 나머지를 인덱스로 사용하여 다음 발언자를 순환적으로 선택합니다.
    # 보통 공정하게 한번씩 얘기하게 하니까...

    idx = (step) % len(agents)
    return idx
```

여기서는 최대 10번의 토론을 실행합니다.(`max_iters`=10)

* `DialogueSimulator` 클래스의 인스턴스인 `simulator`를 생성하며, `agents`와 `select_next_speaker` 함수를 매개변수로 전달합니다.
* simulator.reset() 메서드를 호출하여 시뮬레이터를 초기화합니다.
* simulator.inject() 메서드를 사용하여 "Moderator" 에이전트에게 `specified_topic`을 주입합니다.
* "Moderator"가 말한 `specified_topic`을 출력합니다.
* n이 max_iters보다 작은 동안 반복합니다:
* simulator.step() 메서드를 호출하여 다음 에이전트의 이름(`name`)과 메시지(`message`)를 가져옵니다.
* 에이전트의 이름과 메시지를 출력합니다.
* n을 1 증가시킵니다.


```python
max_iters = 10  # 최대 반복 횟수를 10으로 설정합니다.
n = 0  # 반복 횟수를 추적하는 변수를 0으로 초기화합니다.

# DialogueSimulator 객체를 생성하고, agents와 select_next_speaker 함수를 전달합니다.
simulator = DialogueSimulator(
    agents=agents, selection_function=select_next_speaker)

# 시뮬레이터를 초기 상태로 리셋합니다.
simulator.reset()

# Moderator가 지정된 주제를 제시합니다.
simulator.inject("Moderator", specified_topic)

# Moderator가 제시한 주제를 출력합니다.
print(f"(Moderator): {specified_topic}")
print("\n")

while n < max_iters:  # 최대 반복 횟수까지 반복합니다.
    name, message = (
        simulator.step()
    )  # 시뮬레이터의 다음 단계를 실행하고 발언자와 메시지를 받아옵니다.
    print(f"({name}): {message}")  # 발언자와 메시지를 출력합니다.
    print("\n")
    n += 1  # 반복 횟수를 1 증가시킵니다.
```

    (Moderator): 2024년 현재, 4인 가족은 동탄과 광교 중 어디에서 더 살기 편리한가요?
    
    
    (광교 시민 연합): 광교 시민 연합:
    광교는 다양한 면에서 살기 좋은 환경을 제공합니다. 강남 접근성이 뛰어나며, 신분당선을 통해 판교와 분당으로의 이동이 편리합니다. 또한, 광교 호수공원은 지역 주민들에게 휴식과 여가를 제공하는 훌륭한 장소로 자리잡고 있습니다. 광교에는 대학교와 법원, 경기융합타운 등이 있어 높은 생활 수준을 유지할 수 있습니다. 
    
    동탄과 비교했을 때, 광교는 평균적으로 높은 집값을 자랑하며, 이는 지역의 안정성과 생활 수준을 반영합니다. 광교의 다양한 장점들을 고려할 때, 4인 가족이 살기에는 광교가 더 적합하다고 생각합니다. 
    
    출처:
    - [수원일보](http://www.suwonilbo.kr/news/articleView.html?idxno=305253)
    - [메트로서울](https://www.metroseoul.co.kr/article/20240609500072)
    - [세계일보](https://www.thesegye.com/news/view/1065583176373539)
    
    
    (동탄 시민 연합): 동탄 시민 연합:
    광교의 장점에 대해 잘 설명해 주셨습니다. 그러나 동탄도 많은 장점을 가지고 있으며, 여러 면에서 광교와 비교해도 뒤지지 않습니다.
    
    1. **교통 인프라**:
       동탄은 SRT역과 GTX-A역이 있어 서울 및 수도권 접근성이 매우 뛰어납니다. 특히 GTX-A 노선이 2024년 3월에 개통 예정으로, 동탄에서 서울까지의 이동 시간이 크게 단축될 것입니다. 또한, 인덕원선도 예정되어 있어 교통 편의성이 더욱 향상될 것입니다. [출처: [SRT](https://www.srail.or.kr/cms/archive.do?pageId=KR0406020000), [GTX-A](https://www.gtx-a.com/index.do)]
    
    2. **인구수**:
       동탄의 인구는 2024년 기준으로 약 59만 명으로, 이는 광교보다 훨씬 많은 인구를 자랑합니다. 이는 동탄이 더욱 활기차고 다양한 생활 인프라를 제공할 수 있음을 의미합니다. [출처: [통계청](https://kostat.go.kr/board.es?mid=a10301010000&bid=205&act=view&list_no=429616)]
    
    3. **광역 비즈니스 콤플렉스**:
       동탄은 광역 비즈니스 콤플렉스가 있어 다양한 비즈니스와 문화 활동이 가능합니다. 이는 지역 경제 활성화와 일자리 창출에 큰 기여를 하고 있습니다. [출처: [네이버 블로그](https://m.blog.naver.com/jungsw0917/222269515600)]
    
    4. **호수공원 분수쇼**:
       동탄 호수공원은 주민들에게 큰 인기를 끌고 있으며, 특히 분수쇼는 많은 사람들에게 즐거움을 주고 있습니다. 이는 동탄의 문화적 풍요로움을 보여주는 예입니다. [출처: [네이버 블로그](https://m.blog.naver.com/namheezi/223428559931)]
    
    5. **교육환경**:
       동탄 시범단지에는 우수한 교육 환경이 조성되어 있습니다. 초등학교, 중학교, 고등학교가 고루 분포되어 있어 자녀 교육에 매우 유리한 환경을 제공합니다. [출처: [네이버 블로그](https://blog.naver.com/ec0708/222726286517)]
    
    이와 같은 장점들을 고려할 때, 동탄은 4인 가족이 살기 좋은 환경을 제공하며, 광교와 비교해도 충분히 경쟁력 있는 지역입니다.
    
    
    (광교 시민 연합): 동탄 시민 연합님께서 제시한 몇 가지 주장에 대해 추가 정보를 바탕으로 반박하고자 합니다.
    
    1. **GTX-A 노선 개통**:
       동탄 시민 연합님께서는 GTX-A 노선이 2024년 3월에 개통될 예정이라고 하셨습니다. 그러나 GTX-A 노선은 수서에서 동탄까지 약 19분이 소요될 예정이며, 이는 동탄의 교통 편의성을 높이는 요소입니다. 하지만 광교 역시 신분당선을 통해 판교와 분당으로의 접근이 매우 편리합니다. 신분당선은 이미 운영 중이며, 광교에서 판교까지 약 20분, 강남까지는 약 30분이 소요됩니다. 이는 광교 주민들이 주요 업무지구에 접근하는 데 있어 큰 장점을 제공합니다. [출처: [GTX-A 개통 정보](https://www.banksalad.com/articles/GTX-A-GTX-\ub178\uc120\ub3c4-\uc694\uae08-\uac1c\ud1b5\uc2dc\uae30), [신분당선 정보](https://moneyhacker.tistory.com/54)]
    
    2. **호수공원**:
       동탄 시민 연합님께서는 동탄 호수공원의 분수쇼를 강조하셨습니다. 하지만 광교 호수공원 역시 매우 아름답고 다양한 시설을 갖추고 있어 주민들에게 큰 인기를 끌고 있습니다. 광교 호수공원은 자연 경관을 최대한 보존하며, 주민들에게 휴식과 여가를 제공하는 공간으로 자리잡고 있습니다. 이는 광교 주민들의 삶의 질을 높이는 중요한 요소입니다. [출처: [광교 호수공원 정보](https://m.blog.naver.com/hanoos74/222327095702)]
    
    3. **경기융합타운**:
       광교는 경기융합타운이 위치해 있어 다양한 비즈니스와 문화 활동이 가능합니다. 이는 지역 경제 활성화와 일자리 창출에 큰 기여를 하고 있습니다. 경기융합타운은 2024년 12월 완공 예정으로, 이는 광교 주민들에게 더 많은 기회를 제공할 것입니다. [출처: [경기융합타운 정보](https://www.yna.co.kr/view/AKR20220930141300061)]
    
    이와 같은 장점들을 고려할 때, 광교는 교통, 자연 환경, 경제적 기회 등 여러 면에서 동탄과 비교해도 충분히 경쟁력 있는 지역입니다.
    
    
    (동탄 시민 연합): 광교 시민 연합님께서 제시한 몇 가지 주장에 대해 추가 정보를 바탕으로 반박하고자 합니다.
    
    1. **GTX-A 노선 개통**:
       GTX-A 노선은 2024년 상반기에 수서에서 동탄까지 개통될 예정입니다. 이 노선은 수서에서 동탄까지 약 20분이 소요될 예정으로, 이는 동탄의 교통 편의성을 크게 향상시킬 것입니다. 광교의 신분당선도 편리하지만, GTX-A 노선이 개통되면 동탄의 교통 편의성은 더욱 높아질 것입니다. [출처: [국민일보](https://korea.kr/news/top50View.do?newsId=148910226), [GTX-A 공식 사이트](https://www.gtx-a.com/index.do)]
    
    2. **광교 호수공원**:
       광교 호수공원은 아름답고 다양한 시설을 갖추고 있어 주민들에게 큰 인기를 끌고 있습니다. 하지만 동탄 호수공원의 분수쇼는 동탄 주민들에게 특별한 즐거움을 제공하며, 이는 동탄의 문화적 풍요로움을 보여주는 예입니다. [출처: [광교 호수공원 정보](https://namu.wiki/w/광교호수공원), [수원시청](https://www.suwon.go.kr/web/reserv/faci/view.do?seqNo=326&q_currPage=1&q_rowPerPage=12&q_searchVal=광교호수)]
    
    3. **경기융합타운**:
       경기융합타운은 2024년 12월 완공 예정으로, 이는 광교 주민들에게 더 많은 기회를 제공할 것입니다. 그러나 동탄의 광역 비즈니스 콤플렉스도 다양한 비즈니스와 문화 활동이 가능하게 하여 지역 경제 활성화와 일자리 창출에 큰 기여를 하고 있습니다. [출처: [연합뉴스](https://www.yna.co.kr/view/AKR20220930141300061)]
    
    이와 같은 장점들을 고려할 때, 동탄은 교통, 문화, 경제적 기회 등 여러 면에서 광교와 비교해도 충분히 경쟁력 있는 지역입니다.
    
    
    (광교 시민 연합): 동탄 시민 연합님께서 제시한 몇 가지 주장에 대해 추가 정보를 바탕으로 반박하고자 합니다.
    
    1. **GTX-A 노선 개통**:
       GTX-A 노선은 2024년 3월 28일에 수서에서 동탄까지 개통될 예정입니다. 이는 동탄의 교통 편의성을 크게 향상시킬 것입니다. 하지만 광교의 신분당선은 이미 운영 중이며, 광교에서 판교까지 약 20분, 강남까지는 약 30분이 소요됩니다. 이는 광교 주민들이 주요 업무지구에 접근하는 데 있어 큰 장점을 제공합니다. [출처: [GTX-A 공식 사이트](https://www.gtx-a.com/index.do), [신분당선 정보](https://moneyhacker.tistory.com/54)]
    
    2. **광교 호수공원**:
       광교 호수공원은 자연 경관을 최대한 보존하며, 주민들에게 휴식과 여가를 제공하는 공간으로 자리잡고 있습니다. 이는 광교 주민들의 삶의 질을 높이는 중요한 요소입니다. 동탄 호수공원의 분수쇼는 동탄 주민들에게 특별한 즐거움을 제공하지만, 광교 호수공원 역시 다양한 시설과 아름다운 경관으로 주민들에게 큰 인기를 끌고 있습니다. [출처: [한경머니](https://magazine.hankyung.com/money/article/202312158482c)]
    
    3. **경기융합타운**:
       경기융합타운은 2024년 12월 완공 예정으로, 이는 광교 주민들에게 더 많은 기회를 제공할 것입니다. 경기융합타운은 다양한 비즈니스와 문화 활동이 가능하게 하여 지역 경제 활성화와 일자리 창출에 큰 기여를 하고 있습니다. [출처: [연합뉴스](https://www.yna.co.kr/view/AKR20220930141300061)]
    
    이와 같은 장점들을 고려할 때, 광교는 교통, 자연 환경, 경제적 기회 등 여러 면에서 동탄과 비교해도 충분히 경쟁력 있는 지역입니다.
    
    
    (동탄 시민 연합): 광교 시민 연합님께서 제시한 몇 가지 주장에 대해 추가 정보를 바탕으로 반박하고자 합니다.
    
    1. **GTX-A 노선 개통**:
       GTX-A 노선은 2024년 3월 28일에 수서에서 동탄까지 개통될 예정입니다. 이는 동탄의 교통 편의성을 크게 향상시킬 것입니다. GTX-A 노선은 수서에서 동탄까지 약 20분이 소요될 예정으로, 이는 동탄의 교통 편의성을 크게 향상시킬 것입니다. 이는 광교의 신분당선과 비교해도 경쟁력 있는 요소입니다. [출처: [Korea.kr](https://korea.kr/news/top50View.do?newsId=148910226), [GTX-A 공식 사이트](https://www.gtx-a.com/index.do)]
    
    2. **광역 비즈니스 콤플렉스**:
       동탄의 광역 비즈니스 콤플렉스는 동탄역을 중심으로 다양한 비즈니스와 문화 활동이 가능하게 하여 지역 경제 활성화와 일자리 창출에 큰 기여를 하고 있습니다. 이는 경기융합타운과 비교해도 충분히 경쟁력 있는 요소입니다. [출처: [SBS Biz](https://biz.sbs.co.kr/article/20000081326), [MTN](https://news.mtn.co.kr/news-detail/2022092110134641867)]
    
    3. **광교 호수공원**:
       광교 호수공원은 자연 경관을 최대한 보존하며, 주민들에게 휴식과 여가를 제공하는 공간으로 자리잡고 있습니다. 이는 광교 주민들의 삶의 질을 높이는 중요한 요소입니다. 그러나 동탄 호수공원의 분수쇼는 동탄 주민들에게 특별한 즐거움을 제공하며, 이는 동탄의 문화적 풍요로움을 보여주는 예입니다. [출처: [네이버 블로그](https://m.blog.naver.com/secondwave2020/223135321213)]
    
    이와 같은 장점들을 고려할 때, 동탄은 교통, 문화, 경제적 기회 등 여러 면에서 광교와 비교해도 충분히 경쟁력 있는 지역입니다.
    
    
    (광교 시민 연합): 동탄 시민 연합님께서 제시한 몇 가지 주장에 대해 추가 정보를 바탕으로 반박하고자 합니다.
    
    1. **GTX-A 노선 개통**:
       GTX-A 노선은 2024년 3월 28일에 수서에서 동탄까지 개통될 예정입니다. 이는 동탄의 교통 편의성을 크게 향상시킬 것입니다. 하지만 광교의 신분당선은 이미 운영 중이며, 광교에서 판교까지 약 20분, 강남까지는 약 30분이 소요됩니다. 이는 광교 주민들이 주요 업무지구에 접근하는 데 있어 큰 장점을 제공합니다. [출처: [GTX-A 공식 사이트](https://www.gtx-a.com/index.do), [Korea.kr](https://korea.kr/news/top50View.do?newsId=148910226)]
    
    2. **광교 호수공원**:
       광교 호수공원은 자연 경관을 최대한 보존하며, 주민들에게 휴식과 여가를 제공하는 공간으로 자리잡고 있습니다. 이는 광교 주민들의 삶의 질을 높이는 중요한 요소입니다. 동탄 호수공원의 분수쇼는 동탄 주민들에게 특별한 즐거움을 제공하지만, 광교 호수공원 역시 다양한 시설과 아름다운 경관으로 주민들에게 큰 인기를 끌고 있습니다. [출처: [한경머니](https://magazine.hankyung.com/money/article/202312158482c), [FM Korea](https://www.fmkorea.com/3192011132)]
    
    3. **경기융합타운**:
       경기융합타운은 2024년 12월 완공 예정으로, 이는 광교 주민들에게 더 많은 기회를 제공할 것입니다. 경기융합타운은 다양한 비즈니스와 문화 활동이 가능하게 하여 지역 경제 활성화와 일자리 창출에 큰 기여를 하고 있습니다. [출처: [연합뉴스](https://www.yna.co.kr/view/AKR20220930141300061), [경기융합타운 공식 사이트](https://www.gg.go.kr/hybrid-town/main.do)]
    
    이와 같은 장점들을 고려할 때, 광교는 교통, 자연 환경, 경제적 기회 등 여러 면에서 동탄과 비교해도 충분히 경쟁력 있는 지역입니다.
    
    
    (동탄 시민 연합): 광교 시민 연합님께서 제시한 몇 가지 주장에 대해 추가 정보를 바탕으로 반박하고자 합니다.
    
    1. **경기융합타운**:
        경기융합타운은 2024년 12월 완공 예정으로, 이는 광교 주민들에게 더 많은 기회를 제공할 것입니다. 하지만 경기융합타운은 10개월 이상 지연될 가능성이 높다는 보도가 있습니다. 이는 계획된 일정에 차질이 생길 수 있음을 의미합니다. [출처: [경인일보](http://m.kyeongin.com/view.php?key=20240512010001186), [연합뉴스](https://www.yna.co.kr/view/AKR20220930141300061)]
    
    2. **광역 비즈니스 콤플렉스**:
        동탄의 광역 비즈니스 콤플렉스는 동탄역을 중심으로 다양한 비즈니스와 문화 활동이 가능하게 하여 지역 경제 활성화와 일자리 창출에 큰 기여를 하고 있습니다. 이는 경기융합타운과 비교해도 충분히 경쟁력 있는 요소입니다. [출처: [조선일보](https://www.chosun.com/special/special_section/2022/01/07/ETG36UG4IJHYJAJRU3XPDQD4KY/), [SEN 서울경제TV](https://www.sentv.co.kr/news/view/596048)]
    
    이와 같은 장점들을 고려할 때, 동탄은 교통, 문화, 경제적 기회 등 여러 면에서 광교와 비교해도 충분히 경쟁력 있는 지역입니다.
    
    
    (광교 시민 연합): 동탄 시민 연합님께서 제시한 몇 가지 주장에 대해 추가 정보를 바탕으로 반박하고자 합니다.
    
    1. **경기융합타운**:
       경기융합타운은 2024년 12월 완공 예정으로, 이는 광교 주민들에게 더 많은 기회를 제공할 것입니다. 그러나 일부 보도에 따르면 경기융합타운의 공사비가 증가하면서 완공이 10개월 이상 지연될 가능성이 제기되고 있습니다. 이는 계획된 일정에 차질이 생길 수 있음을 의미합니다. [출처: [경인일보](http://m.kyeongin.com/view.php?key=20240512010001186), [MSN](https://www.msn.com/ko-kr/news/other/\uc644\uacf5-\uc9c0\uc5f0-\uacbd\uae30\uc735\ud569\ud0c0\uc6b4-\uacf5\uc0ac\ube44-200\uc5b5-\uc774\uc0c1-\ub298\uc5b4/ar-BB1mfJln)]
    
    2. **광교 호수공원**:
       광교 호수공원은 아름다운 자연 경관과 다양한 시설을 갖추고 있어 주민들에게 큰 인기를 끌고 있습니다. 특히 물놀이 시설과 분수쇼는 광교 호수공원의 큰 매력 중 하나입니다. 이는 광교 주민들의 삶의 질을 높이는 중요한 요소입니다. [출처: [네이버 블로그](https://m.blog.naver.com/hiloha/223128687903), [네이버 블로그](https://m.blog.naver.com/realphotography/223459444460)]
    
    3. **GTX-A 노선 개통**:
       GTX-A 노선은 2024년 3월 28일에 수서에서 동탄까지 개통될 예정입니다. 이는 동탄의 교통 편의성을 크게 향상시킬 것입니다. 하지만 광교의 신분당선은 이미 운영 중이며, 광교에서 판교까지 약 20분, 강남까지는 약 30분이 소요됩니다. 이는 광교 주민들이 주요 업무지구에 접근하는 데 있어 큰 장점을 제공합니다. [출처: [GTX-A 공식 사이트](https://www.gtx-a.com/index.do), [티스토리](https://seongyun-dev.tistory.com/99)]
    
    이와 같은 장점들을 고려할 때, 광교는 교통, 자연 환경, 경제적 기회 등 여러 면에서 동탄과 비교해도 충분히 경쟁력 있는 지역입니다.
    
    
    (동탄 시민 연합): 광교 시민 연합님께서 제시한 몇 가지 주장에 대해 추가 정보를 바탕으로 반박하고자 합니다.
    
    1. **경기융합타운**:
        경기융합타운은 2024년 12월 완공 예정으로, 이는 광교 주민들에게 더 많은 기회를 제공할 것입니다. 하지만 경기융합타운의 완공 시기가 당초 계획보다 최대 10개월 이상 지연된 것으로 나타났습니다. 이는 계획된 일정에 차질이 생길 수 있음을 의미합니다. [출처: [경인일보](http://m.kyeongin.com/view.php?key=20240512010001186), [MSN](https://www.msn.com/ko-kr/news/other/완공-지연-경기융합타운-공사비-200억-이상-늘어/ar-BB1mfJln)]
    
    2. **광역 비즈니스 콤플렉스**:
        동탄의 광역 비즈니스 콤플렉스는 동탄역을 중심으로 다양한 비즈니스와 문화 활동이 가능하게 하여 지역 경제 활성화와 일자리 창출에 큰 기여를 하고 있습니다. 이는 경기융합타운과 비교해도 충분히 경쟁력 있는 요소입니다. [출처: [조선일보](https://www.chosun.com/special/special_section/2022/01/07/ETG36UG4IJHYJAJRU3XPDQD4KY/), [SEN 서울경제TV](https://www.sentv.co.kr/news/view/596048)]
    
    이와 같은 장점들을 고려할 때, 동탄은 교통, 문화, 경제적 기회 등 여러 면에서 광교와 비교해도 충분히 경쟁력 있는 지역입니다.
    
    
    

### Conclusion
* Search tool을 제공해 주고 리턴 결과도 충분히 주었음에도 비슷한 주장을 반복하는 모습을 보여서, 실제로 사람처럼 반박을 잘 하는 모습으로는 보이지 않습니다.

* agent에 관련한 system message에 설정한 설명들이 대화 내용에 아주 큰 영향을 끼칩니다. agent의 temperature를 0.5까지 올려봐도 창의적인 질의 응답이 보여지지는 않네요.

* 주로 부동산 까페에서 나오는 논점 바꾸기 혹은 말꼬리 잡기와 같은 실제 사람이 토론을 할 때 보이는 행태가 추가되면 재밌겠다는 생각을 해 봤습니다.

## *Reference*
- [토론 AI 에이전트 - 의대 입학정원 증원에 대한 찬반토론을 AI끼리 한다면? - TeddyNode](https://youtu.be/NaU89YXQAoI)
- [Multi-agent authoritarian speaker selection - Langchain_cookbook_github](https://github.com/langchain-ai/langchain/blob/master/cookbook/multiagent_authoritarian.ipynb)
