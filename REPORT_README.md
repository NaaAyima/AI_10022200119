# Compiling Your LaTeX Report

This guide explains how to compile the `main.tex` file into a polished PDF document using MiKTeX (or TeXworks, which is included with MiKTeX).

## Requirements
Ensure you have [MiKTeX](https://miktex.org/download) installed on your Windows machine. TeXworks is automatically installed alongside it.

## Compiling via TeXworks (UI GUI)
1. Open the **TeXworks** application from your Windows Start Menu.
2. Click **File > Open** and select the `main.tex` file generated in this directory.
3. In the top-left corner, ensure the compilation drop-down menu is set to **pdfLaTeX**.
4. Click the **Green Play Button** (Compile).
5. **IMPORTANT:** Click the Green Play button a **second time**. 

## Compiling via Command Line
If you prefer the terminal, simply open Git Bash or PowerShell in this directory (`C:\Users\USER\Desktop\AI_10022200119`) and run:
```bash
pdflatex main.tex
pdflatex main.tex
```

## Why run it TWICE?
LaTeX compiles your document linearly. On the first run, it identifies where all your headings, sections, and figures are, and saves that information internally. On the **second run**, it generates the actual **Table of Contents** and hyperlinks. 

If you only run it once, your Table of Contents will be completely blank!
