# ReleaseTrain.io — Software Ecosystem Framework

Architecture overview, data flow, frontend modules, and a complete API reference for **releasetrain.io**.  
The platform ingests ecosystem signals (GitHub, CVE/NVD, Reddit), enriches and classifies updates, and exposes them via a clean API and triage UI.

---

## Quick Links

-  **Live:** <https://releasetrain.io/>
-  **Client:** <https://github.com/SE4CPS/releasetrain-client>
-  **Project Board:** <https://github.com/orgs/SE4CPS/projects/24>

---

## System Architecture

```mermaid
graph LR
  A["a) Bot1 (GitHub) [online]"] --> D["d) MongoDB Data Lake [atlas]"]
  B["b) Bot2 (CVE/NVD) [online]"] --> D
  C["c) Bot3 (Reddit) [online]"] --> D

  Note1["Improve data quality [ongoing]"] --> D

  subgraph E_SG["e) AI / ML"]
    direction LR
    Type["Type [ok]"]
    Security["Security [ok]"]
    Breaking["Breaking [ok]"]
    Sentimental["Sentiment [todo]"]
    RAG["LLM / RAG [pilot]"]
    Type --> E["Classified Output"]
    Security --> E
    Breaking --> E
    Sentimental --> E
    RAG --> E
  end

  subgraph Azure["g) Azure Cloud"]
    AZ["gpt-4o"] <--> RAG
  end

  E <--> Agent[" Agentic AI (LLM+API+DB+Rules)"]
  Agent <--> D
  Agent <--> F["f) Backend API (Express)"]

  F <--> FEEntry["Frontend Router"]

  subgraph H["h) Frontend"]
    direction LR
    FE1["1) Ecosystem Graph Architecture [ok]"]
    FE2["2) Ecosystem UML Architecture [ok]"]
    FE3["3) Ecosystem Software Updates [ok]"]
    FE4["4) Triage Buttons [ok]"]
    FE5["5) Release Notes Generator [todo]"]
    FEEntry --> FE1 & FE2 & FE3 & FE4 & FE5
  end
