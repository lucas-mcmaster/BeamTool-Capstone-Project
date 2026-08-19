# BeamTool — AI-Assisted Structural Beam Analysis & Selection Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Framework-Flask-green?logo=flask)
![NumPy](https://img.shields.io/badge/Math-NumPy-013243?logo=numpy)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458?logo=pandas)
![Gemini API](https://img.shields.io/badge/AI-Google%20Gemini%202.5%20Flash-orange?logo=google)

> **Capstone Design Project | University of Toronto**  
> *Team Members: Lucas McMaster, Stephanie Temovsky, Raghav Saxena, Mitchell Brat,*  
> *Supervisor & Industry Advisor: Prof. Fatemeh Jazinizadeh*

---

## Overview

Traditional structural engineering workflows during early-stage conceptual design rely heavily on manual trial-and-error, conservative experience-based rules of thumb, or commercial analysis packages (such as SkyCiv or WebStructural). While these tools perform deterministic analysis once a beam section is already selected, none automatically synthesize user requirements to recommend optimal standard sections from scratch. This often results in structural overdesign, excessive material consumption, and inflated embodied carbon footprints.

**BeamTool** is a full-stack engineering web application that bridges this gap. It integrates:

1. **Classical Beam Theory & Singularity Function Solver (Macaulay's Method)** for exact, continuous internal force, moment, and deflection calculations across arbitrary loading/support conditions.
2. **Multi-Criteria Decision Analysis (MCDA)** engine that filters, evaluates, and ranks candidate structural sections from a curated database against cost, weight, and safety constraints.
3. **AI-Powered Structural Engineering Assistant** powered by Google Gemini 2.5 Flash, dynamically injected with real-time application state to provide pedagogical guidance, explain trade-offs, and prevent overdesign.

---

## System Architecture

```text
                  +---------------------------------------+
                  |       Frontend Web Interface          |
                  |  (HTML5 / CSS3 / Vanilla JavaScript)  |
                  +---------------------------------------+
                            |                  ^
                REST / JSON |                  | Real-Time Feedback
                            v                  | (SVG Schematics & Diagrams)
                  +---------------------------------------+
                  |         Flask API Gateway             |
                  |             (app.py)                  |
                  +---------------------------------------+
                     /                 |                 \
        +-------------------+  +-------------------+  +-------------------+
        |  Analysis Engine  |  | Comparator Engine |  | AI Design Tutor   |
        | (Macaulay Solver) |  |   (MCDA / SAW)    |  | (Gemini 2.5 Flash)|
        +-------------------+  +-------------------+  +-------------------+
                  ^                      ^
                  |                      |
        +-------------------+  +-------------------+
        | Structural Inputs |  |   Beam Database   |
        | (Loads, Supports) |  |   (CSV Catalog)   |
        +-------------------+  +-------------------+
```

---

## Core Modules & Technical Details

### 1. Structural Analysis Engine (`analysis_engine.py`)
- **Macaulay’s Singularity Method**: Solves the governing Euler-Bernoulli differential equation:
  $$EI \frac{d^2y}{dx^2} = M(x)$$
  superimposing point loads, distributed loads, applied moments, and support configurations (pin, roller, fixed) into a unified analytical formulation.
- **Simultaneous Matrix Resolution**: Formulates the equilibrium and geometric boundary equations as a linear system $Ax = B$, solving for unknown reaction forces, support moments, and integration constants using `numpy.linalg.solve`.
- **Instability & Mechanism Detection**: Catches kinematic mechanisms and ill-conditioned boundary conditions via `numpy.linalg.LinAlgError` exceptions before executing downstream routines.
- **Vectorized Continuous Output**: Evaluates equations across 2,000+ discretized points to compute continuous arrays for Shear Force Diagrams (SFD), Bending Moment Diagrams (BMD), and elastic Deflection curves ($y(x) \cdot \frac{1}{EI}$).

### 2. Multi-Criteria Comparator Engine (`comparator_engine.py`)
- **Database Preprocessing**: Ingests standard structural shapes (W-Flanges, HSS, CHS, Channels, Angles in Structural Steel and 6061-T6 Aluminum) via Pandas.
- **Hard Constraint Filtering**:
  - Slenderness verification ($L/d \ge 10$) to enforce Euler-Bernoulli validity.
  - Environmental suitability matching (Indoor, Outdoor, Marine, Seismic).
  - Minimum Safety Factor ($SF \ge 1.0$) based on Von Mises yield criteria ($\sigma_{max} = \frac{M_{max} \cdot c}{I_{xx}}$).
  - Maximum allowable serviceability deflection ($L/240$).
  - Self-weight compensation ($w = \rho A g$) automatically added as an active dead load.
- **Simple Additive Weighting (SAW) Model**: Normalizes and scores surviving candidates:
  $$\text{Score} = w_{\text{cost}} \cdot (1 - \hat{C}) + w_{\text{weight}} \cdot (1 - \hat{W}) + w_{\text{SF}} \cdot \hat{SF} - \text{Penalties}$$
  *Baseline Weights: Cost = 0.50, Weight = 0.40, Safety Factor = 0.10 (with dynamic adjustment for user constraints).*
- **Trade-Off Tracking**: Identifies and logs rejected candidates that narrowly missed user constraints to assist engineering decision-making.

### 3. Context-Aware AI Design Assistant (`app.py`)
- Integrated via Google GenAI SDK (`gemini-2.5-flash`).
- **Dynamic Context Injection**: Extracts live UI state (loads, supports, results, top recommendation, rejected candidates, trade-offs) and injects it into a curated pedagogical system prompt.
- **Token Optimization**: Encodes structural parameters in shorthand syntax (`P:5000@2.5m`), bounds rejected candidates to top 8, and maintains a rolling 10-message conversational memory.

---

## Repository Structure

```text
BeamTool-Capstone-Project/
├── app.py                     # Flask backend routing and REST API endpoints
├── analysis_engine.py         # Singularity function structural analysis engine
├── comparator_engine.py       # Multi-criteria decision ranking and filtering logic
├── Beam Data - Sheet3.csv     # Structured database of structural steel and aluminum sections
├── templates/
│   └── index.html             # Responsive dual-mode UI (Analysis & Design Mode)
├── .gitignore
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Google Gemini API key (optional, for AI tutoring chat)

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lucas-mcmaster/BeamTool-Capstone-Project.git
   cd BeamTool-Capstone-Project
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate

   # On Windows:
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install flask pandas numpy python-dotenv google-genai
   ```

4. **Configure Environment Variables (Optional for AI Chat):**
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

5. **Run the Application:**
   ```bash
   python app.py
   ```
   Open your browser and navigate to: `http://localhost:5001`

---

##  Verification & Validation

| Parameter | Beam Validated | Theoretical (Hand Calc) | BeamTool Computed | % Error | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Max Bending Moment** | W200x22 (Steel) | 11,267.67 N·m | 11,267.67 N·m | 0.00% | **PASS** |
| **Max Bending Moment** | AL-W180x18 (Aluminum) | 11,030.78 N·m | 11,030.78 N·m | 2.10% | **PASS** |
| **Max Deflection** | W200x22 (Steel) | 3.372 mm | 3.370 mm | 0.07% | **PASS** |
| **Max Deflection** | AL-W180x18 (Aluminum) | 17.156 mm | 17.210 mm | 0.32% | **PASS** |
| **MCDA Ranking Accuracy** | CHS Sections (Marine) | Manual Ranking Match | Comparator Match | NA | **PASS** |

---

## License & Acknowledgments

Developed as part of the MIE491 Capstone Design Project at the University of Toronto (Department of Mechanical & Industrial Engineering). Not for public distribution
