# Sprintly (AutoPM) 🧠🚀  
_A multi-agent product management copilot built with NVIDIA Nemotron + LangGraph_

> Automating the product management lifecycle — from messy customer feedback to Jira backlogs and CI/CD plans.

---

## 💡 Inspiration

Sprintly was born from watching real product managers (including one of our teammates’ dad) spend countless hours on
coordination instead of creativity.

We saw the same pattern over and over:

- Reading long research reports and survey data  
- Manually writing Jira epics, stories, and acceptance criteria  
- Constantly syncing with DevOps and engineering to plan releases  

When we saw a hackathon track focused on **product manager workflows**, we decided to build an AI system that acts like
a digital PM partner — one that **researches, plans, and coordinates execution** end-to-end.

At the same time, we were experimenting with **NVIDIA Nemotron models** and were impressed by their reasoning and
planning capabilities. That led to Sprintly: a **multi-agent system orchestrated by LangGraph** that thinks and works
like a PM.

---

## 🚀 What Sprintly Does

**Sprintly is a connected workflow assistant for product managers.**  
From idea → research → Jira backlog → DevOps plan, all in one continuous flow.

### 🧠 1. ResearchAI — The Insight Generator  
**Model:** Nemotron Nano 9B v2  

- Ingests customer feedback, survey data, and research docs  
- Summarizes key trends and pain points  
- Surfaces product opportunities and passes structured insights downstream  

**Why it’s useful:** PMs get **instant research summaries** and structured insights instead of digging through raw data.

---

### 📋 2. JiraAI — The Task Architect  
**Model:** Llama 3.3 Nemotron Super 49B v1.5  

- Converts ResearchAI outputs into **epics, user stories, and acceptance criteria**  
- Maintains linking between epics → stories → priorities  
- Integrates with **Jira Cloud API** to automatically create and update issues

**Why it’s useful:** Removes the manual grunt work of writing Jira tickets so PMs can focus on **direction and scope**.

---

### ⚙️ 3. DevOpsPlannerAI — The Execution Partner  
**Model:** Nemotron Mini 4B Instruct  

- Scans the repository (tests, package files, existing workflows)  
- Proposes **CI/CD workflows**, sprint plans, and release checklists  
- Can generate `.github/workflows/*.yml` and open pull requests via **GitHub REST API**

**Why it’s useful:** Bridges the gap between **planning and engineering**, helping PMs connect strategy → deployment.

---

### 🌐 LangGraph — The Orchestrator

Sprintly uses **LangGraph** as a routing and memory layer:

- Routes each user request to **ResearchAI, JiraAI, DevOpsPlannerAI**, or a combination  
- Shares state between agents so output from one is clean input to the next  
- Supports flows like:

```text
ResearchAI  →  JiraAI  →  DevOpsPlannerAI
(insights)     (backlog)   (CI/CD & sprint plan)
