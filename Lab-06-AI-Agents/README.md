# Lab 06: Building a Tool-Using AI Agent with Memory and RAG

## 1. Laboratory Session

* **Session:** Lab 06. Building a Tool-Using AI Agent

* **Topic:** Function/Tool Calling, Desktop GUI Automation (`pyautogui`), Persistent SQLite Memory and RAG-Powered Agentic Workflows

## 2. Objective

This practical session aims to build an AI agent. It should be able to pick and run system tools on its own (Calculator, Notepad, File Search/Creation, System Info). It should also keep a lasting record of the user through SQLite.. It should pull university knowledge from the Lab 05 ChromaDB vector store.

## 3.. Implementation

1. **Desktop. Multi-Tool Agent (`part_a_tools_agent.py`):** I built 6 system tools with `pyautogui`, `subprocess`, `psutil` and `glob` (Calculator automation, Date/Time, Notepad text entry, PDF file search, text file creation and hardware specs). Then I ran all 6 benchmark queries to test them.

2. **Persistent SQLite Memory & RAG Agent (`rag_memory_agent.py`):** I hooked up an `SQLite database. It saves user profile facts. Brings them back across turns. This works together with a Calculator tool and the Lab 05 ChromaDB RAG retriever (`all-MiniLM-v2` + `SmolLM2-135M-Instruct`).

* **Part A Multi-Tool Agent:** [part_a_tools_agent.py](./part_a_tools_agent.py)

* **Parts B & C Memory + RAG Agent:** [rag_memory_agent.py](./rag_memory_agent.py)

## 4.. Evidence

* **Desktop Tool Execution:** I got Windows Calculator and Windows Notepad to open through code. The agent typed in the input on its `25 * 4 = 100` and `"Hello Students"`), checked system specs and made `test.txt`.

* **Stateful Memory Recall:** User identity facts went into SQLite (`agent_memory.db`). The agent brought them back correctly in later turns.

* **RAG Knowledge Retrieval:** The agent was linked to Lab 05s ChromaDB index. It answered university document questions next to math and memory tasks.

### Execution Screenshots

![screenshot1](./screenshots/ss1.png)

![screenshot2](./screenshots/ss2.png)

![screenshot3](./screenshots/ss3.png)

![screenshot4](./screenshots/ss4.png)

![screenshot5](./screenshots/ss5.png)

![screenshot6](./screenshots/ss6.png)

![screenshot7](./screenshots/ss7.png)

![screenshot8](./screenshots/ss8.png)

![screenshot9](./screenshots/ss9.png)

## 5. Observations

* **Deterministic Tool Routing vs. Pure Generation:** Sending math, system checks and file work to Python tools removes the wrong answers about numbers and system state that plain LLMs often give.

* **Value of Persistent SQLite Memory:** Normal chat loops lose context once you clear them. Saving facts in SQLite lets the agent remember user details across separate terminal sessions.

## 6. Learning Outcomes

* I learned how to give an LLM workflow action tools and automated desktop GUI interactions.

* I set up lasting long-term memory with SQLite, next to term conversational context.

* I joined Retrieval-Augmented Generation (RAG) with tool-using agents to make one unified University AI Assistant.

---

**Generative AI Declaration:** Generative AI tools helped me in a support role (A2: Planning & Structuring Projects, A5: Language Refinement and A8: Technical Guidance & Debugging Support). They were used to plan the repository layout, adjust Hugging Face model pipelines and polish the documentation. I did all code running, experiments. Result checks on my own.
