# Screenshot Checklist for LaTeX Report

To complete your final `main.tex` compile before submitting your project, you **must manually capture** the following 4 screenshots and place them in the same folder as `main.tex`. Update the LaTeX code (`\includegraphics{...}`) with your filename.

- [ ] **1. Architecture Diagram**
  - **Location in Report:** Section 3 (System Architecture)
  - **What to capture:** The pipeline/architecture diagram you drew or generated describing how GovLens works. Save the image as `architecture_diagram.png`.

- [ ] **2. Retrieved Chunks & Scores**
  - **Location in Report:** Section 7 (Retrieval System)
  - **What to capture:** Open the GovLens Streamlit app, ask a question, click on the **"🔍 View Retrieved Context & Scores"** expander, and take a screenshot of the chunks and their score numbers (including the Domain Router Boost tag). Save as `retrieved_chunks.png`.

- [ ] **3. UI Interface**
  - **Location in Report:** Section 13 (Deployment)
  - **What to capture:** An attractive, wide screenshot of the GovLens Streamlit webpage showing the "Ask about government budgets & elections" hero section, your logo, and the clickable buttons. Save as `ui_interface.png`.

- [ ] **4. Build / Terminal Logs**
  - **Location in Report:** Section 13 (Deployment)
  - **What to capture:** A screenshot of your terminal showing the `streamlit run app.py` startup process OR your Render.com dashboard logs showing the "Build successful" or "Live" output. Save as `deployment_logs.png`.

### How to insert them into your LaTeX file:
Find the `% [Insert screenshot of...]` lines in your `main.tex` and replace them with:
`\includegraphics[width=0.9\textwidth]{your_image_name_here.png}`
