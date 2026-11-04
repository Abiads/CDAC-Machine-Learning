import json
import re
import os

BASE_DIR = r"c:\Users\conne\Downloads\ML_CDAC_ALILOG\notes\interactive_revision"

with open(os.path.join(BASE_DIR, "questions.json"), "r", encoding="utf-8") as f:
    questions = json.load(f)

with open(os.path.join(BASE_DIR, "blanks.json"), "r", encoding="utf-8") as f:
    blanks = json.load(f)

with open(os.path.join(BASE_DIR, "last_minute_revision.html"), "r", encoding="utf-8") as f:
    rev_html = f.read()

# Extract sections of last_minute_revision
rev_match = re.search(r'<main class="container">(.*?)</main>', rev_html, re.DOTALL)
rev_body = rev_match.group(1) if rev_match else ""
rev_body = re.sub(r'<nav class="jump-nav".*?</nav>', '', rev_body, flags=re.DOTALL)

# Extract additional sections from fill_in_the_blanks.html (sections 4, 5, 6, 7)
with open(os.path.join(BASE_DIR, "fill_in_the_blanks.html"), "r", encoding="utf-8") as f:
    fib_html = f.read()

fib_match = re.search(r'(<!-- SECTION 4 -->.*)</main>', fib_html, re.DOTALL)
fib_extra_sections = fib_match.group(1) if fib_match else ""

questions_json_str = json.dumps(questions, ensure_ascii=False)
blanks_json_str = json.dumps(blanks, ensure_ascii=False)

template = '''<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CDAC ML & Statistics Master Hub | Interactive MCQs, Blanks & Revision</title>
  <meta name="description" content="All-in-one interactive revision portal for CDAC Machine Learning and Statistics: 66 verified MCQs, interactive fill-in-the-blanks, comprehensive cheat sheets, attempt history tracking, and web search research tools.">

  <!-- Vercel Web Analytics -->
  <script>
    window.va = window.va || function () { (window.vaq = window.vaq || []).push(arguments); };
  </script>
  <script defer src="/_vercel/insights/script.js"></script>

  <!-- Vercel Speed Insights -->
  <script>
    window.si = window.si || function () { (window.siq = window.siq || []).push(arguments); };
  </script>
  <script defer src="/_vercel/speed-insights/script.js"></script>

  <style>
    :root {
      --bg: #0b0f19;
      --surface: #131b2e;
      --surface-hover: #1c2742;
      --surface-border: #243252;
      --surface-elevated: #1a243d;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --text-subtle: #64748b;
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
      --success: #10b981;
      --success-bg: rgba(16, 185, 129, 0.15);
      --success-border: #059669;
      --error: #ef4444;
      --error-bg: rgba(239, 68, 68, 0.15);
      --error-border: #dc2626;
      --warning: #f59e0b;
      --warning-bg: rgba(245, 158, 11, 0.15);
      --warning-border: #d97706;
      --info: #0ea5e9;
      --info-bg: rgba(14, 165, 233, 0.15);
      --card-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.4);
      --glow: 0 0 20px rgba(99, 102, 241, 0.25);
    }

    html.light {
      --bg: #f8fafc;
      --surface: #ffffff;
      --surface-hover: #f1f5f9;
      --surface-border: #cbd5e1;
      --surface-elevated: #ffffff;
      --text: #0f172a;
      --text-muted: #64748b;
      --text-subtle: #94a3b8;
      --accent: #4f46e5;
      --accent-hover: #4338ca;
      --accent-gradient: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
      --success: #059669;
      --success-bg: #ecfdf5;
      --success-border: #10b981;
      --error: #dc2626;
      --error-bg: #fef2f2;
      --error-border: #ef4444;
      --warning: #d97706;
      --warning-bg: #fffbeb;
      --warning-border: #f59e0b;
      --info: #0284c7;
      --info-bg: #f0f9ff;
      --card-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.06), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
      --glow: 0 0 15px rgba(79, 70, 229, 0.15);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      -webkit-tap-highlight-color: transparent;
    }

    body {
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      transition: background-color 0.25s ease, color 0.25s ease;
      line-height: 1.6;
    }

    /* Header */
    header {
      background-color: var(--surface);
      border-bottom: 1px solid var(--surface-border);
      position: sticky;
      top: 0;
      z-index: 50;
      backdrop-filter: blur(12px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }

    .header-top {
      max-width: 1200px;
      margin: 0 auto;
      padding: 0.75rem 1rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }

    .logo-area {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      cursor: pointer;
    }

    .logo-icon {
      width: 40px;
      height: 40px;
      border-radius: 10px;
      background: var(--accent-gradient);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.3rem;
      box-shadow: var(--glow);
    }

    .logo-text {
      display: flex;
      flex-direction: column;
    }

    .logo-title {
      font-size: 1.15rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }

    .logo-subtitle {
      font-size: 0.75rem;
      color: var(--text-muted);
      font-weight: 500;
    }

    .header-controls {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .status-badge {
      background-color: var(--surface-hover);
      border: 1px solid var(--surface-border);
      padding: 0.35rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 0.35rem;
      transition: all 0.2s;
    }

    .timer-badge {
      background: rgba(99, 102, 241, 0.12);
      border: 1px solid rgba(99, 102, 241, 0.3);
      color: var(--accent);
      padding: 0.35rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 700;
      font-family: ui-monospace, SFMono-Regular, monospace;
      display: flex;
      align-items: center;
      gap: 0.35rem;
      cursor: pointer;
    }

    .timer-badge:hover {
      background: rgba(99, 102, 241, 0.2);
    }

    .btn-header {
      background-color: var(--surface-hover);
      border: 1px solid var(--surface-border);
      color: var(--text);
      padding: 0.4rem 0.75rem;
      border-radius: 8px;
      font-size: 0.825rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      transition: all 0.2s;
      text-decoration: none;
    }

    .btn-header:hover {
      background-color: var(--surface-border);
      color: var(--text);
    }

    .btn-header.accent {
      background: var(--accent);
      color: #ffffff;
      border-color: var(--accent);
    }

    .btn-header.accent:hover {
      background: var(--accent-hover);
    }

    /* Navigation Tabs Bar */
    .tabs-bar-wrapper {
      background-color: var(--surface-elevated);
      border-top: 1px solid var(--surface-border);
    }

    .tabs-bar {
      max-width: 1200px;
      margin: 0 auto;
      padding: 0 1rem;
      display: flex;
      gap: 0.5rem;
      overflow-x: auto;
      scrollbar-width: none;
    }

    .tabs-bar::-webkit-scrollbar {
      display: none;
    }

    .tab-btn {
      padding: 0.75rem 1rem;
      border: none;
      background: transparent;
      color: var(--text-muted);
      font-size: 0.875rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      white-space: nowrap;
      position: relative;
      transition: color 0.2s;
    }

    .tab-btn:hover {
      color: var(--text);
    }

    .tab-btn.active {
      color: var(--accent);
      font-weight: 700;
    }

    .tab-btn.active::after {
      content: '';
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: var(--accent);
      border-radius: 3px 3px 0 0;
    }

    .tab-pill-count {
      background: var(--surface-border);
      color: var(--text-muted);
      padding: 0.1rem 0.45rem;
      border-radius: 9999px;
      font-size: 0.7rem;
      font-weight: 700;
    }

    .tab-btn.active .tab-pill-count {
      background: rgba(99, 102, 241, 0.2);
      color: var(--accent);
    }

    /* Main Container */
    .app-container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 1.5rem 1rem 3rem 1rem;
      width: 100%;
      flex: 1;
    }

    /* Tab Content Views */
    .tab-view {
      display: none;
      animation: fadeIn 0.2s ease-in-out;
    }

    .tab-view.active {
      display: block;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Generic UI Components */
    .card {
      background-color: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 14px;
      padding: 1.5rem;
      margin-bottom: 1.25rem;
      box-shadow: var(--card-shadow);
    }

    .btn {
      background-color: var(--surface-hover);
      border: 1px solid var(--surface-border);
      color: var(--text);
      padding: 0.55rem 1.1rem;
      border-radius: 8px;
      font-size: 0.875rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.45rem;
      transition: all 0.2s;
      text-decoration: none;
    }

    .btn:hover {
      background-color: var(--surface-border);
    }

    .btn-primary {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }

    .btn-primary:hover {
      background: var(--accent-hover);
      box-shadow: var(--glow);
    }

    .btn-danger {
      background: var(--error-bg);
      color: var(--error);
      border-color: var(--error);
    }

    .btn-danger:hover {
      background: var(--error);
      color: white;
    }

    .btn-success {
      background: var(--success);
      color: white;
      border-color: var(--success);
    }

    .btn-warning {
      background: var(--warning-bg);
      color: var(--warning);
      border-color: var(--warning);
    }

    .btn-warning:hover {
      background: var(--warning);
      color: white;
    }

    .btn-sm {
      padding: 0.35rem 0.75rem;
      font-size: 0.8rem;
    }

    /* Filter Bar */
    .filter-bar {
      background-color: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 14px;
      padding: 1rem 1.25rem;
      margin-bottom: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
      box-shadow: var(--card-shadow);
    }

    .filter-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.75rem;
    }

    .filter-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
    }

    .filter-pill {
      background-color: var(--surface-hover);
      border: 1px solid var(--surface-border);
      color: var(--text-muted);
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      transition: all 0.2s;
    }

    .filter-pill:hover {
      color: var(--text);
      border-color: var(--text-subtle);
    }

    .filter-pill.active {
      background: rgba(99, 102, 241, 0.18);
      color: var(--accent);
      border-color: var(--accent);
      font-weight: 700;
    }

    .filter-pill.danger.active {
      background: var(--error-bg);
      color: var(--error);
      border-color: var(--error);
    }

    .filter-pill.warning.active {
      background: var(--warning-bg);
      color: var(--warning);
      border-color: var(--warning);
    }

    .select-dropdown {
      background-color: var(--surface-hover);
      border: 1px solid var(--surface-border);
      color: var(--text);
      padding: 0.4rem 0.8rem;
      border-radius: 8px;
      font-size: 0.825rem;
      font-weight: 600;
      outline: none;
      cursor: pointer;
    }

    .select-dropdown:focus {
      border-color: var(--accent);
    }

    /* Scoreboard */
    .scoreboard-box {
      background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.7));
      border: 1px solid var(--surface-border);
      border-radius: 14px;
      padding: 1.15rem 1.35rem;
      margin-bottom: 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 1rem;
    }

    .score-stats {
      display: flex;
      align-items: center;
      gap: 1.25rem;
      flex-wrap: wrap;
    }

    .score-stat-item {
      display: flex;
      flex-direction: column;
    }

    .score-stat-label {
      font-size: 0.725rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }

    .score-stat-val {
      font-size: 1.25rem;
      font-weight: 800;
      color: var(--text);
    }

    .score-stat-val.correct { color: var(--success); }
    .score-stat-val.wrong { color: var(--error); }
    .score-stat-val.accent { color: var(--accent); }

    .progress-bar-container {
      width: 100%;
      height: 7px;
      background-color: var(--surface-border);
      border-radius: 9999px;
      overflow: hidden;
      margin-top: 0.5rem;
    }

    .progress-bar-fill {
      height: 100%;
      background: var(--accent-gradient);
      width: 0%;
      transition: width 0.3s ease;
      border-radius: 9999px;
    }

    /* Quiz Card Layout */
    .quiz-grid-layout {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 1.5rem;
      align-items: start;
    }

    @media (max-width: 960px) {
      .quiz-grid-layout {
        grid-template-columns: 1fr;
      }
    }

    .quiz-card {
      background-color: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 16px;
      padding: 1.75rem;
      box-shadow: var(--card-shadow);
      position: relative;
    }

    .quiz-card-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 1rem;
      margin-bottom: 1.25rem;
      flex-wrap: wrap;
    }

    .quiz-meta-tags {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.45rem;
    }

    .badge-tag {
      padding: 0.2rem 0.6rem;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }

    .badge-track {
      background: rgba(99, 102, 241, 0.15);
      color: var(--accent);
      border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .badge-session {
      background: var(--surface-hover);
      color: var(--text-muted);
      border: 1px solid var(--surface-border);
    }

    .badge-source {
      background: rgba(14, 165, 233, 0.12);
      color: var(--info);
      border: 1px solid rgba(14, 165, 233, 0.3);
    }

    .badge-mistake-flag {
      background: var(--error-bg);
      color: var(--error);
      border: 1px solid var(--error-border);
    }

    .card-actions-top {
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }

    .btn-bookmark {
      background: transparent;
      border: 1px solid var(--surface-border);
      color: var(--text-muted);
      border-radius: 8px;
      padding: 0.35rem 0.65rem;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      transition: all 0.2s;
    }

    .btn-bookmark:hover {
      color: var(--warning);
      border-color: var(--warning);
    }

    .btn-bookmark.marked {
      background: var(--warning-bg);
      color: var(--warning);
      border-color: var(--warning-border);
    }

    .btn-web-search {
      background: transparent;
      border: 1px solid var(--surface-border);
      color: var(--text-muted);
      border-radius: 8px;
      padding: 0.35rem 0.65rem;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      transition: all 0.2s;
      text-decoration: none;
    }

    .btn-web-search:hover {
      color: var(--info);
      border-color: var(--info);
    }

    .question-prompt {
      font-size: 1.15rem;
      font-weight: 700;
      line-height: 1.55;
      color: var(--text);
      margin-bottom: 1.5rem;
    }

    .options-list {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      margin-bottom: 1.5rem;
    }

    .option-btn {
      background-color: var(--surface-hover);
      border: 1.5px solid var(--surface-border);
      border-radius: 10px;
      padding: 0.95rem 1.15rem;
      text-align: left;
      font-size: 0.95rem;
      color: var(--text);
      cursor: pointer;
      display: flex;
      align-items: flex-start;
      gap: 0.85rem;
      transition: all 0.15s ease;
      line-height: 1.45;
      width: 100%;
    }

    .option-btn:hover:not(:disabled) {
      background-color: var(--surface-border);
      transform: translateY(-1px);
    }

    .option-key {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 26px;
      height: 26px;
      border-radius: 6px;
      background: var(--surface-border);
      color: var(--text-muted);
      font-size: 0.8rem;
      font-weight: 700;
      flex-shrink: 0;
      margin-top: 1px;
    }

    .option-text {
      flex: 1;
    }

    .option-btn.selected-correct {
      background-color: var(--success-bg) !important;
      border-color: var(--success) !important;
      color: var(--success) !important;
      font-weight: 600;
    }

    .option-btn.selected-correct .option-key {
      background-color: var(--success);
      color: #ffffff;
    }

    .option-btn.selected-wrong {
      background-color: var(--error-bg) !important;
      border-color: var(--error) !important;
      color: var(--error) !important;
      font-weight: 600;
    }

    .option-btn.selected-wrong .option-key {
      background-color: var(--error);
      color: #ffffff;
    }

    .option-btn.reveal-correct {
      background-color: var(--success-bg) !important;
      border-color: var(--success) !important;
      color: var(--success) !important;
      border-style: dashed;
    }

    .option-btn.reveal-correct .option-key {
      background-color: var(--success);
      color: #ffffff;
    }

    .option-btn.exam-selected {
      background-color: rgba(99, 102, 241, 0.15) !important;
      border-color: var(--accent) !important;
      color: var(--text) !important;
      font-weight: 600;
    }

    .option-btn.exam-selected .option-key {
      background-color: var(--accent);
      color: #ffffff;
    }

    /* Explanation Box */
    .explanation-box {
      background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(99, 102, 241, 0.05));
      border: 1.5px solid var(--success-border);
      border-radius: 12px;
      padding: 1.25rem 1.4rem;
      margin-top: 1.25rem;
      animation: fadeIn 0.25s ease-in-out;
    }

    .explanation-box.wrong-banner {
      background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(245, 158, 11, 0.05));
      border-color: var(--error-border);
    }

    .explanation-title {
      font-size: 0.9rem;
      font-weight: 700;
      color: var(--success);
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.5rem;
    }

    .explanation-title.wrong {
      color: var(--error);
    }

    .explanation-body {
      font-size: 0.925rem;
      color: var(--text);
      line-height: 1.6;
    }

    .explanation-footer {
      margin-top: 0.85rem;
      padding-top: 0.75rem;
      border-top: 1px solid var(--surface-border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.5rem;
      font-size: 0.8rem;
      color: var(--text-muted);
    }

    /* Quiz Card Bottom Nav */
    .quiz-card-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.75rem;
      margin-top: 1.75rem;
      padding-top: 1.25rem;
      border-top: 1px solid var(--surface-border);
    }

    .nav-arrow-group {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    /* Jump Grid Sidebar */
    .jump-palette-card {
      background-color: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 16px;
      padding: 1.25rem;
      box-shadow: var(--card-shadow);
      position: sticky;
      top: 5rem;
    }

    .jump-palette-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
    }

    .jump-palette-title {
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }

    .jump-grid {
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 0.4rem;
      max-height: 380px;
      overflow-y: auto;
      padding-right: 0.25rem;
    }

    .jump-btn {
      aspect-ratio: 1;
      border: 1px solid var(--surface-border);
      background-color: var(--surface-hover);
      color: var(--text-muted);
      border-radius: 8px;
      font-size: 0.75rem;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      transition: all 0.15s;
    }

    .jump-btn:hover {
      border-color: var(--accent);
      color: var(--text);
    }

    .jump-btn.current {
      border: 2px solid var(--accent);
      box-shadow: var(--glow);
      font-weight: 800;
      color: var(--text);
    }

    .jump-btn.correct {
      background-color: var(--success-bg);
      border-color: var(--success);
      color: var(--success);
    }

    .jump-btn.wrong {
      background-color: var(--error-bg);
      border-color: var(--error);
      color: var(--error);
    }

    .jump-btn.marked::after {
      content: '★';
      position: absolute;
      top: -3px;
      right: 0px;
      color: var(--warning);
      font-size: 0.65rem;
    }

    .jump-legend {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.45rem;
      margin-top: 1rem;
      padding-top: 0.75rem;
      border-top: 1px solid var(--surface-border);
      font-size: 0.725rem;
      color: var(--text-muted);
    }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 0.35rem;
    }

    .legend-dot {
      width: 10px;
      height: 10px;
      border-radius: 3px;
    }

    /* Attempts View */
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }

    .metric-card {
      background-color: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 14px;
      padding: 1.25rem;
      box-shadow: var(--card-shadow);
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .metric-icon {
      width: 48px;
      height: 48px;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.4rem;
      flex-shrink: 0;
    }

    .metric-icon.indigo { background: rgba(99, 102, 241, 0.15); color: var(--accent); }
    .metric-icon.green { background: var(--success-bg); color: var(--success); }
    .metric-icon.amber { background: var(--warning-bg); color: var(--warning); }
    .metric-icon.red { background: var(--error-bg); color: var(--error); }

    .metric-data {
      display: flex;
      flex-direction: column;
    }

    .metric-label {
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }

    .metric-value {
      font-size: 1.4rem;
      font-weight: 800;
      color: var(--text);
    }

    .attempts-table-card {
      background-color: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 14px;
      padding: 1.5rem;
      box-shadow: var(--card-shadow);
    }

    .table-responsive {
      width: 100%;
      overflow-x: auto;
    }

    table.data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
      text-align: left;
    }

    table.data-table th, table.data-table td {
      padding: 0.85rem 1rem;
      border-bottom: 1px solid var(--surface-border);
      vertical-align: middle;
    }

    table.data-table th {
      background-color: var(--surface-hover);
      color: var(--text-muted);
      font-weight: 700;
      text-transform: uppercase;
      font-size: 0.75rem;
      letter-spacing: 0.05em;
    }

    table.data-table tr:hover td {
      background-color: var(--surface-hover);
    }

    /* Fill in the blanks */
    .fib-card {
      background-color: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 14px;
      padding: 1.5rem;
      margin-bottom: 1.25rem;
      box-shadow: var(--card-shadow);
    }

    .fib-prompt {
      font-size: 1.05rem;
      font-weight: 600;
      line-height: 1.6;
      color: var(--text);
      margin-bottom: 1rem;
    }

    .blank-input-group {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-bottom: 0.85rem;
    }

    .blank-input {
      background-color: var(--surface-hover);
      border: 1.5px solid var(--surface-border);
      color: var(--text);
      padding: 0.55rem 0.95rem;
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 600;
      outline: none;
      min-width: 250px;
      flex: 1;
      transition: all 0.2s;
    }

    .blank-input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
    }

    .blank-input.correct {
      border-color: var(--success);
      background-color: var(--success-bg);
      color: var(--success);
    }

    .blank-input.wrong {
      border-color: var(--error);
      background-color: var(--error-bg);
    }

    /* Web Search View */
    .search-hero {
      background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(236, 72, 153, 0.08));
      border: 1px solid rgba(99, 102, 241, 0.3);
      border-radius: 16px;
      padding: 2rem;
      margin-bottom: 1.75rem;
      text-align: center;
    }

    .search-input-box {
      max-width: 700px;
      margin: 1.25rem auto 0 auto;
      display: flex;
      gap: 0.5rem;
    }

    .search-input {
      background-color: var(--surface);
      border: 1.5px solid var(--surface-border);
      color: var(--text);
      padding: 0.8rem 1.25rem;
      border-radius: 10px;
      font-size: 1rem;
      width: 100%;
      outline: none;
      box-shadow: var(--card-shadow);
    }

    .search-input:focus {
      border-color: var(--accent);
      box-shadow: var(--glow);
    }

    .search-quick-engines {
      display: flex;
      align-items: center;
      justify-content: center;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 1rem;
    }

    .topic-chips {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 0.45rem;
      margin-top: 1.25rem;
    }

    .topic-chip {
      background-color: var(--surface);
      border: 1px solid var(--surface-border);
      color: var(--text-muted);
      padding: 0.3rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }

    .topic-chip:hover {
      color: var(--accent);
      border-color: var(--accent);
      transform: translateY(-1px);
    }

    /* Modals */
    .modal-backdrop {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(4px);
      z-index: 100;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 1rem;
    }

    .modal-backdrop.active {
      display: flex;
    }

    .modal-content {
      background-color: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 20px;
      max-width: 650px;
      width: 100%;
      max-height: 90vh;
      overflow-y: auto;
      padding: 2rem;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
      position: relative;
      animation: modalPop 0.25s ease-out;
    }

    .modal-content.wide {
      max-width: 900px;
    }

    @keyframes modalPop {
      from { opacity: 0; transform: scale(0.95); }
      to { opacity: 1; transform: scale(1); }
    }

    .modal-close-btn {
      position: absolute;
      top: 1.25rem;
      right: 1.25rem;
      background: var(--surface-hover);
      border: 1px solid var(--surface-border);
      color: var(--text-muted);
      width: 32px;
      height: 32px;
      border-radius: 8px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1rem;
    }

    .modal-close-btn:hover {
      color: var(--text);
    }

    /* Toast Notification */
    .toast-container {
      position: fixed;
      bottom: 1.5rem;
      right: 1.5rem;
      z-index: 150;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      pointer-events: none;
    }

    .toast {
      background-color: var(--surface);
      border: 1px solid var(--surface-border);
      color: var(--text);
      padding: 0.75rem 1.25rem;
      border-radius: 10px;
      box-shadow: var(--card-shadow);
      font-size: 0.85rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      pointer-events: auto;
      animation: slideInRight 0.25s ease-out;
    }

    @keyframes slideInRight {
      from { opacity: 0; transform: translateX(30px); }
      to { opacity: 1; transform: translateX(0); }
    }

    /* Embedded Tables in Cheat Sheet */
    .rev-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
      text-align: left;
      margin: 1rem 0;
    }

    .rev-table th, .rev-table td {
      padding: 0.75rem 1rem;
      border: 1px solid var(--surface-border);
      vertical-align: top;
    }

    .rev-table th {
      background-color: var(--surface-hover);
      color: var(--text);
      font-weight: 700;
      border-bottom: 2px solid var(--accent);
    }

    .example-box {
      background-color: var(--surface-hover);
      border: 1.5px solid var(--surface-border);
      border-left: 4px solid var(--success);
      border-radius: 10px;
      padding: 1.25rem;
      margin: 1rem 0;
      font-size: 0.9rem;
    }

    .example-title {
      font-weight: 700;
      color: var(--success);
      font-size: 0.9rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.5rem;
    }

    .trap-box {
      background-color: var(--error-bg);
      border: 1.5px solid var(--error-border);
      border-left: 4px solid var(--error);
      border-radius: 10px;
      padding: 1.25rem;
      margin: 1rem 0;
      font-size: 0.9rem;
    }

    .trap-title {
      font-weight: 700;
      color: var(--error);
      font-size: 0.9rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.5rem;
    }

    pre.code-block {
      background-color: #0b0f19;
      color: #38bdf8;
      padding: 1rem;
      border-radius: 8px;
      font-family: ui-monospace, SFMono-Regular, monospace;
      font-size: 0.85rem;
      overflow-x: auto;
      margin: 0.75rem 0;
      line-height: 1.5;
    }

    /* Mobile Responsive */
    @media (max-width: 640px) {
      .header-top { padding: 0.5rem 0.75rem; }
      .app-container { padding: 1rem 0.5rem 2rem 0.5rem; }
      .quiz-card { padding: 1.25rem; }
      .question-prompt { font-size: 1.05rem; }
      .option-btn { padding: 0.8rem; font-size: 0.9rem; }
      .score-stats { gap: 0.75rem; }
      .modal-content { padding: 1.25rem; }
      .hide-mobile { display: none; }
    }
  </style>
</head>
<body>

  <!-- Sticky Header -->
  <header>
    <div class="header-top">
      <div class="logo-area" onclick="switchTab('quiz')">
        <div class="logo-icon">🧠</div>
        <div class="logo-text">
          <span class="logo-title">CDAC ML & Stats Master Hub</span>
          <span class="logo-subtitle">66 MCQs • Blanks • Cheat Sheets • Attempt Tracker</span>
        </div>
      </div>
      <div class="header-controls">
        <div id="storageStatusBadge" class="status-badge" title="Local session persists automatically on this device">
          💾 <span>0/66 Saved</span>
        </div>
        <div id="timerDisplay" class="timer-badge" title="Click to Pause / Resume Timer">
          ⏱️ <span id="timerText">00:00</span>
        </div>
        <button id="themeToggleBtn" class="btn-header" title="Toggle Light / Dark Mode">
          <span id="themeIcon">☀️</span>
        </button>
        <button id="globalSearchBtn" class="btn-header accent" title="Search all MCQs, Blanks & Notes">
          <span>🔍</span> <span class="hide-mobile">Search</span>
        </button>
      </div>
    </div>

    <!-- Navigation Tabs Bar -->
    <div class="tabs-bar-wrapper">
      <nav class="tabs-bar" aria-label="Portal Navigation Tabs">
        <button class="tab-btn active" data-tab="quiz" onclick="switchTab('quiz')">
          <span>🎯</span> MCQ Quiz Engine <span class="tab-pill-count">66</span>
        </button>
        <button class="tab-btn" data-tab="blanks" onclick="switchTab('blanks')">
          <span>📝</span> Fill-in-the-Blanks & Review <span class="tab-pill-count">14+</span>
        </button>
        <button class="tab-btn" data-tab="cheat" onclick="switchTab('cheat')">
          <span>⚡</span> Last-Minute Cheat Sheet <span class="tab-pill-count">9 Sections</span>
        </button>
        <button class="tab-btn" data-tab="attempts" onclick="switchTab('attempts')">
          <span>📊</span> Attempts & Mistakes Tracker <span id="navMistakesBadge" class="tab-pill-count">0</span>
        </button>
        <button class="tab-btn" data-tab="research" onclick="switchTab('research')">
          <span>🌐</span> Web Research & Tools
        </button>
      </nav>
    </div>
  </header>

  <!-- Main Container -->
  <main class="app-container">

    <!-- ========================================== -->
    <!-- TAB 1: MCQ QUIZ ENGINE                     -->
    <!-- ========================================== -->
    <section id="quizView" class="tab-view active">
      
      <!-- Filter Bar -->
      <div class="filter-bar">
        <div class="filter-row">
          <div class="filter-pills">
            <button class="filter-pill active" data-track="all" onclick="setTrackFilter('all')">All Tracks (66)</button>
            <button class="filter-pill" data-track="Machine Learning" onclick="setTrackFilter('Machine Learning')">🤖 Machine Learning (38)</button>
            <button class="filter-pill" data-track="Statistics" onclick="setTrackFilter('Statistics')">📊 Statistics (28)</button>
          </div>
          <div style="display: flex; align-items: center; gap: 0.5rem;">
            <label style="font-size: 0.8rem; font-weight: 600; color: var(--text-muted);">Session:</label>
            <select id="sessionDropdown" class="select-dropdown" onchange="onSessionDropdownChange(this.value)">
              <option value="all">All Sessions</option>
            </select>
          </div>
        </div>

        <div class="filter-row" style="border-top: 1px solid var(--surface-border); padding-top: 0.75rem;">
          <div class="filter-pills">
            <button id="pillAllStatus" class="filter-pill active" onclick="setStatusFilter('all')">📋 All</button>
            <button id="pillMistakes" class="filter-pill danger" onclick="setStatusFilter('mistakes')">
              ⚠️ Review Mistakes (<span id="mistakesCountDisplay">0</span>)
            </button>
            <button id="pillMarked" class="filter-pill warning" onclick="setStatusFilter('marked')">
              ⭐️ Starred for Review (<span id="markedCountDisplay">0</span>)
            </button>
            <button id="pillUnanswered" class="filter-pill" onclick="setStatusFilter('unanswered')">
              ⏳ Unanswered (<span id="unansweredCountDisplay">66</span>)
            </button>
          </div>

          <div style="display: flex; align-items: center; gap: 0.5rem;">
            <button id="toggleQuizModeBtn" class="btn btn-sm" onclick="toggleQuizMode()" title="Switch between instant Practice mode and exam Timed mode">
              <span id="quizModeIcon">💡</span> <span id="quizModeText">Practice Mode (Instant Feedback)</span>
            </button>
            <button class="btn btn-sm" onclick="shuffleCurrentQuestions()" title="Shuffle order of questions">
              🔀 Shuffle
            </button>
          </div>
        </div>
      </div>

      <!-- Scoreboard Banner -->
      <div class="scoreboard-box">
        <div class="score-stats">
          <div class="score-stat-item">
            <span class="score-stat-label">Question</span>
            <span class="score-stat-val accent"><span id="currentQNum">1</span> / <span id="totalQNum">66</span></span>
          </div>
          <div class="score-stat-item">
            <span class="score-stat-label">Correct</span>
            <span id="correctScore" class="score-stat-val correct">0</span>
          </div>
          <div class="score-stat-item">
            <span class="score-stat-label">Wrong</span>
            <span id="wrongScore" class="score-stat-val wrong">0</span>
          </div>
          <div class="score-stat-item">
            <span class="score-stat-label">Accuracy</span>
            <span id="accuracyScore" class="score-stat-val">0%</span>
          </div>
        </div>

        <div style="display: flex; align-items: center; gap: 0.5rem;">
          <button class="btn btn-primary btn-sm" onclick="finishAndSaveAttempt()" title="Calculate final score, save attempt to device history, and review mistakes">
            💾 Finish & Record Attempt
          </button>
          <button class="btn btn-sm" onclick="resetActiveQuizSession()" title="Reset answers and restart fresh">
            🔄 Reset
          </button>
        </div>

        <div class="progress-bar-container">
          <div id="progressBarFill" class="progress-bar-fill"></div>
        </div>
      </div>

      <!-- Main Quiz Card & Jump Grid Layout -->
      <div class="quiz-grid-layout">
        
        <!-- Active Question Card -->
        <div id="quizCard" class="quiz-card">
          <div class="quiz-card-header">
            <div class="quiz-meta-tags">
              <span id="cardTrackBadge" class="badge-tag badge-track">Machine Learning</span>
              <span id="cardSessionBadge" class="badge-tag badge-session">Session 1</span>
              <span id="cardSourceBadge" class="badge-tag badge-source">Slide 12</span>
              <span id="cardMistakeFlag" class="badge-tag badge-mistake-flag" style="display: none;">❌ Answered Incorrectly</span>
            </div>
            <div class="card-actions-top">
              <button id="cardBookmarkBtn" class="btn-bookmark" onclick="toggleCurrentBookmark()">
                <span id="bookmarkStarIcon">☆</span> Star for Review
              </button>
              <a id="cardWebSearchBtn" href="#" target="_blank" rel="noopener noreferrer" class="btn-web-search" title="Search this question concept on Google">
                🔍 Search Web
              </a>
            </div>
          </div>

          <div id="cardQuestionPrompt" class="question-prompt">
            Loading question...
          </div>

          <div id="cardOptionsList" class="options-list">
            <!-- Rendered by JS -->
          </div>

          <!-- In-depth Pedagogical Explanation Box (revealed on answer) -->
          <div id="cardExplanationBox" class="explanation-box" style="display: none;">
            <div id="cardExplanationTitle" class="explanation-title">
              <span>💡 In-Depth Pedagogical Explanation</span>
              <span id="cardAnswerStatusText"></span>
            </div>
            <div id="cardExplanationBody" class="explanation-body">
              <!-- Explanation text -->
            </div>
            <div class="explanation-footer">
              <span id="cardDeckRef">Deck Reference</span>
              <button class="btn btn-sm" style="padding: 0.2rem 0.5rem; font-size: 0.75rem;" onclick="searchCurrentConceptOnline()">
                🔍 Search Topic on Google
              </button>
            </div>
          </div>

          <!-- Bottom Navigation Controls -->
          <div class="quiz-card-footer">
            <div class="nav-arrow-group">
              <button id="prevQuestionBtn" class="btn" onclick="goToPrevQuestion()">
                ⬅ Previous
              </button>
              <button id="nextQuestionBtn" class="btn btn-primary" onclick="goToNextQuestion()">
                Next ➡
              </button>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
              <span style="font-size: 0.8rem; color: var(--text-muted);">Shortcuts: Arrow Keys or 1-4 / A-D</span>
            </div>
          </div>

        </div>

        <!-- Jump Palette Sidebar -->
        <div class="jump-palette-card">
          <div class="jump-palette-header">
            <div class="jump-palette-title">
              <span>🧭</span> Question Navigator
            </div>
            <span id="jumpSummaryText" style="font-size: 0.75rem; color: var(--text-muted); font-weight: 600;">0 / 66 answered</span>
          </div>

          <div id="jumpGrid" class="jump-grid">
            <!-- 66 buttons generated by JS -->
          </div>

          <div class="jump-legend">
            <div class="legend-item"><span class="legend-dot" style="background: var(--surface-hover); border: 1px solid var(--surface-border);"></span> Unanswered</div>
            <div class="legend-item"><span class="legend-dot" style="background: var(--success-bg); border: 1px solid var(--success);"></span> Correct</div>
            <div class="legend-item"><span class="legend-dot" style="background: var(--error-bg); border: 1px solid var(--error);"></span> Incorrect</div>
            <div class="legend-item"><span class="legend-dot" style="background: var(--warning-bg); border: 1px solid var(--warning);"></span> ★ Starred</div>
          </div>
        </div>

      </div>

    </section>

    <!-- ========================================== -->
    <!-- TAB 2: FILL-IN-THE-BLANKS & REVIEW GUIDE   -->
    <!-- ========================================== -->
    <section id="blanksView" class="tab-view">
      <div class="card" style="margin-bottom: 1.5rem; background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(16, 185, 129, 0.05));">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
          <div>
            <h2 style="font-size: 1.35rem; font-weight: 800; margin-bottom: 0.25rem;">📝 Interactive Fill-in-the-Blanks & Conceptual Review</h2>
            <p style="font-size: 0.875rem; color: var(--text-muted);">
              Test your recall on formulas, DataFrames, distributions, probability problems & ML concepts. Type your answer to test yourself or click reveal!
            </p>
          </div>
          <div style="display: flex; gap: 0.5rem;">
            <button class="btn btn-sm" onclick="toggleAllBlanksAnswers()">👁️ <span id="toggleBlanksLabel">Reveal All Answers</span></button>
            <button class="btn btn-sm btn-primary" onclick="resetBlanksPractice()">🔄 Clear Inputs</button>
          </div>
        </div>
      </div>

      <!-- Quick section jump pills -->
      <div class="filter-pills" style="margin-bottom: 1.5rem;">
        <a href="#fib-sec-1" class="filter-pill">Session 1: DataFrames (8)</a>
        <a href="#fib-sec-2" class="filter-pill">Session 4: Data Features (3)</a>
        <a href="#fib-sec-3" class="filter-pill">Session 5: Skewness & Kurtosis (3)</a>
        <a href="#sec-4" class="filter-pill">Session 6: Probability & Dice</a>
        <a href="#sec-5" class="filter-pill">Session 7: Binomial & Poisson Problems</a>
        <a href="#sec-6" class="filter-pill">Sessions 9-11: Hypothesis Testing</a>
        <a href="#sec-7" class="filter-pill">ML Comparison Matrix</a>
      </div>

      <!-- Interactive Blanks Container (generated dynamically) -->
      <div id="interactiveBlanksContainer"></div>

      <!-- Extra Conceptual Sections from fill_in_the_blanks.html -->
      <div id="extraFibSections" style="margin-top: 2rem;">
        __FIB_EXTRA__
      </div>
    </section>

    <!-- ========================================== -->
    <!-- TAB 3: LAST-MINUTE REVISION CHEAT SHEET    -->
    <!-- ========================================== -->
    <section id="cheatView" class="tab-view">
      
      <!-- Interactive Hypothesis Test Decision Tree Widget -->
      <div class="card" style="border-left: 5px solid var(--accent); margin-bottom: 2rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem;">
          <h2 style="font-size: 1.25rem; font-weight: 800; display: flex; align-items: center; gap: 0.5rem;">
            <span>🧭</span> Interactive Statistical Test Decision Tool
          </h2>
          <span class="status-badge">⚡ High-Yield Exam Helper</span>
        </div>
        <p style="font-size: 0.875rem; color: var(--text-muted); margin-bottom: 1rem;">
          Click your experimental setup to immediately reveal the exact statistical test to apply, formulas, and Python syntax:
        </p>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 0.75rem; margin-bottom: 1.25rem;">
          <button class="btn" style="text-align: left; justify-content: flex-start;" onclick="selectTestScenario('1-sample')">
            📌 1 Group vs Known Value
          </button>
          <button class="btn" style="text-align: left; justify-content: flex-start;" onclick="selectTestScenario('2-independent')">
            👥 2 Independent Groups (A vs B)
          </button>
          <button class="btn" style="text-align: left; justify-content: flex-start;" onclick="selectTestScenario('2-paired')">
            🔄 2 Paired Groups (Before vs After)
          </button>
          <button class="btn" style="text-align: left; justify-content: flex-start;" onclick="selectTestScenario('3-plus')">
            🏢 3+ Independent Groups (A vs B vs C)
          </button>
          <button class="btn" style="text-align: left; justify-content: flex-start;" onclick="selectTestScenario('categorical')">
            📊 2 Categorical Variables (Gender vs Preference)
          </button>
        </div>

        <div id="testScenarioResult" class="explanation-box" style="display: none; margin-top: 0.5rem;">
          <!-- Rendered by JS -->
        </div>
      </div>

      <!-- Rich 9 Revision Sections from last_minute_revision.html -->
      <div id="cheatSheetBody">
        __REV_BODY__
      </div>

    </section>

    <!-- ========================================== -->
    <!-- TAB 4: ATTEMPTS & MISTAKES TRACKER         -->
    <!-- ========================================== -->
    <section id="attemptsView" class="tab-view">
      
      <!-- Live Session Status Card -->
      <div class="card" style="border-left: 5px solid var(--accent); margin-bottom: 2rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
          <div>
            <h2 style="font-size: 1.25rem; font-weight: 800; margin-bottom: 0.25rem; display: flex; align-items: center; gap: 0.5rem;">
              <span>💾</span> Current Active Device Session
            </h2>
            <p style="font-size: 0.85rem; color: var(--text-muted);">
              Every answer marked on this device is continuously preserved in LocalStorage.
            </p>
          </div>
          <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
            <button class="btn btn-warning" onclick="practiceCurrentMistakes()">
              ⚠️ Practice My Mistakes (<span id="attemptsMistakesCount">0</span>)
            </button>
            <button class="btn btn-primary" onclick="finishAndSaveAttempt()">
              💾 Save As Completed Attempt
            </button>
            <button class="btn btn-danger btn-sm" onclick="resetActiveQuizSession()">
              🧹 Clear Active Session
            </button>
          </div>
        </div>
      </div>

      <!-- High-level Metric Cards -->
      <div class="metric-grid">
        <div class="metric-card">
          <div class="metric-icon indigo">📋</div>
          <div class="metric-data">
            <span class="metric-label">Completed Attempts</span>
            <span id="metricTotalAttempts" class="metric-value">0</span>
          </div>
        </div>
        <div class="metric-card">
          <div class="metric-icon green">🏆</div>
          <div class="metric-data">
            <span class="metric-label">Best Score</span>
            <span id="metricBestScore" class="metric-value">0%</span>
          </div>
        </div>
        <div class="metric-card">
          <div class="metric-icon amber">🎯</div>
          <div class="metric-data">
            <span class="metric-label">Average Score</span>
            <span id="metricAvgScore" class="metric-value">0%</span>
          </div>
        </div>
        <div class="metric-card">
          <div class="metric-icon red">⚠️</div>
          <div class="metric-data">
            <span class="metric-label">Active Mistakes</span>
            <span id="metricActiveMistakes" class="metric-value">0</span>
          </div>
        </div>
      </div>

      <!-- Attempts History Log Table -->
      <div class="attempts-table-card">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.25rem; flex-wrap: wrap; gap: 0.75rem;">
          <h3 style="font-size: 1.15rem; font-weight: 800; display: flex; align-items: center; gap: 0.45rem;">
            <span>📜</span> Attempt History Log (Device Storage)
          </h3>
          <div style="display: flex; align-items: center; gap: 0.5rem;">
            <button class="btn btn-sm" onclick="exportAttemptsData()">📥 Export Data (JSON)</button>
            <button class="btn btn-sm" onclick="document.getElementById('importFileInput').click()">📤 Import Data</button>
            <input type="file" id="importFileInput" accept=".json" style="display: none;" onchange="importAttemptsData(event)">
            <button class="btn btn-sm btn-danger" onclick="clearAllAttemptsHistory()">🗑️ Clear History</button>
          </div>
        </div>

        <div class="table-responsive">
          <table class="data-table">
            <thead>
              <tr>
                <th>Attempt #</th>
                <th>Date & Time</th>
                <th>Track / Scope</th>
                <th>Score</th>
                <th>Accuracy</th>
                <th>Time Spent</th>
                <th>Grade</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody id="attemptsTableBody">
              <tr>
                <td colspan="8" style="text-align: center; color: var(--text-muted); padding: 2rem;">
                  No attempts recorded yet. Answer questions in the MCQ Quiz and click "Finish & Record Attempt"!
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

    </section>

    <!-- ========================================== -->
    <!-- TAB 5: WEB RESEARCH & SEARCH HUB           -->
    <!-- ========================================== -->
    <section id="researchView" class="tab-view">
      
      <div class="search-hero">
        <h2 style="font-size: 1.6rem; font-weight: 800; margin-bottom: 0.5rem;">
          🔍 CDAC ML & Statistics Research Hub
        </h2>
        <p style="font-size: 0.95rem; color: var(--text-muted); max-width: 600px; margin: 0 auto;">
          Instantly search through all 66 MCQs, Fill-in-the-Blanks, and Cheat Sheet notes, or launch 1-click external web research.
        </p>

        <div class="search-input-box">
          <input type="text" id="liveSearchInput" class="search-input" placeholder="Type a concept (e.g., p-value, ROC AUC, kurtosis, imputation, Lasso)..." oninput="executeLiveSearch(this.value)">
          <button class="btn btn-primary" onclick="launchWebSearch('google')">Search Google</button>
        </div>

        <div class="search-quick-engines">
          <button class="btn btn-sm" onclick="launchWebSearch('google')">🌐 Google</button>
          <button class="btn btn-sm" onclick="launchWebSearch('wikipedia')">📖 Wikipedia</button>
          <button class="btn btn-sm" onclick="launchWebSearch('arxiv')">🔬 ArXiv</button>
          <button class="btn btn-sm" onclick="launchWebSearch('sklearn')">📚 Scikit-Learn</button>
          <button class="btn btn-sm" onclick="launchWebSearch('statquest')">📺 StatQuest YouTube</button>
        </div>

        <div class="topic-chips">
          <span style="font-size: 0.75rem; color: var(--text-muted); font-weight: 700; align-self: center; margin-right: 0.25rem;">QUICK TOPICS:</span>
          <span class="topic-chip" onclick="applySearchTerm('p-value')">P-Value & Alpha</span>
          <span class="topic-chip" onclick="applySearchTerm('Type I')">Type I vs Type II Error</span>
          <span class="topic-chip" onclick="applySearchTerm('Central Limit')">Central Limit Theorem</span>
          <span class="topic-chip" onclick="applySearchTerm('kurtosis')">Skewness & Kurtosis</span>
          <span class="topic-chip" onclick="applySearchTerm('confusion matrix')">Confusion Matrix & F1</span>
          <span class="topic-chip" onclick="applySearchTerm('regularization')">L1 Lasso vs L2 Ridge</span>
          <span class="topic-chip" onclick="applySearchTerm('ANOVA')">One-Way ANOVA</span>
          <span class="topic-chip" onclick="applySearchTerm('imputation')">Missing Data Imputation</span>
          <span class="topic-chip" onclick="applySearchTerm('Poisson')">Poisson & Binomial</span>
        </div>
      </div>

      <!-- Live Search Results Container -->
      <div id="liveSearchResults" style="display: none;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
          <h3 id="searchResultsTitle" style="font-size: 1.15rem; font-weight: 700;">Search Results</h3>
          <span id="searchResultsCount" class="status-badge">0 matches found</span>
        </div>
        <div id="searchResultsList" style="display: flex; flex-direction: column; gap: 1rem;"></div>
      </div>

    </section>

  </main>

  <!-- ========================================== -->
  <!-- MODALS                                     -->
  <!-- ========================================== -->

  <!-- Attempt Result Celebration Modal -->
  <div id="attemptResultModal" class="modal-backdrop">
    <div class="modal-content" style="text-align: center;">
      <button class="modal-close-btn" onclick="closeAttemptResultModal()">✕</button>
      
      <div style="font-size: 3rem; margin-bottom: 0.5rem;" id="modalResultEmoji">🎉</div>
      <h2 id="modalResultTitle" style="font-size: 1.6rem; font-weight: 800; margin-bottom: 0.25rem;">Attempt Completed!</h2>
      <p id="modalResultSubtitle" style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 1.5rem;">
        Your answers and mistake records have been saved to this device.
      </p>

      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin-bottom: 1.5rem;">
        <div class="card" style="padding: 1rem; margin-bottom: 0;">
          <span class="score-stat-label">Score</span>
          <div id="modalScoreVal" style="font-size: 1.4rem; font-weight: 800; color: var(--accent);">0 / 66</div>
        </div>
        <div class="card" style="padding: 1rem; margin-bottom: 0;">
          <span class="score-stat-label">Accuracy</span>
          <div id="modalAccuracyVal" style="font-size: 1.4rem; font-weight: 800; color: var(--success);">0%</div>
        </div>
        <div class="card" style="padding: 1rem; margin-bottom: 0;">
          <span class="score-stat-label">Time Spent</span>
          <div id="modalTimeVal" style="font-size: 1.4rem; font-weight: 800; color: var(--text);">0m 0s</div>
        </div>
      </div>

      <div id="modalMistakesBanner" class="explanation-box wrong-banner" style="display: none; margin-bottom: 1.5rem; text-align: left;">
        <div class="explanation-title wrong">
          <span>⚠️ Mistakes Detected</span>
          <span id="modalMistakesCountText">0 questions wrong</span>
        </div>
        <p style="font-size: 0.85rem; color: var(--text);">
          Retesting only your mistakes is the fastest proven way to achieve 100% mastery before the exam!
        </p>
      </div>

      <div style="display: flex; flex-direction: column; gap: 0.65rem;">
        <button id="modalRetestMistakesBtn" class="btn btn-warning" style="width: 100%;" onclick="retestMistakesFromModal()">
          🔄 Re-test Only Mistakes Now
        </button>
        <button class="btn btn-primary" style="width: 100%;" onclick="viewAttemptInHistoryFromModal()">
          📊 View In Attempt History
        </button>
        <button class="btn" style="width: 100%;" onclick="closeAttemptResultModal()">
          Continue Reviewing Questions
        </button>
      </div>
    </div>
  </div>

  <!-- Attempt Details Inspector Modal -->
  <div id="attemptInspectModal" class="modal-backdrop">
    <div class="modal-content wide">
      <button class="modal-close-btn" onclick="closeInspectModal()">✕</button>
      <div style="margin-bottom: 1.25rem;">
        <h2 id="inspectModalTitle" style="font-size: 1.35rem; font-weight: 800;">Attempt Inspection</h2>
        <p id="inspectModalSubtitle" style="font-size: 0.85rem; color: var(--text-muted);"></p>
      </div>

      <div id="inspectQuestionsList" style="display: flex; flex-direction: column; gap: 1rem; max-height: 60vh; overflow-y: auto; padding-right: 0.5rem;">
        <!-- Rendered by JS -->
      </div>

      <div style="margin-top: 1.5rem; display: flex; justify-content: flex-end; gap: 0.5rem;">
        <button id="inspectRetestBtn" class="btn btn-warning">🔄 Retest Mistakes From This Attempt</button>
        <button class="btn" onclick="closeInspectModal()">Close</button>
      </div>
    </div>
  </div>

  <!-- Toast Notification Container -->
  <div id="toastContainer" class="toast-container"></div>

  <!-- ========================================== -->
  <!-- APPLICATION LOGIC (DATA & STATE)           -->
  <!-- ========================================== -->
  <script>
    // Embedded Databases
    const RAW_QUESTIONS = __QUESTIONS_JSON__;
    const RAW_BLANKS = __BLANKS_JSON__;

    // Storage Keys
    const KEY_SESSION = 'cdac_ml_quiz_active_session_v2';
    const KEY_ATTEMPTS = 'cdac_ml_quiz_attempts_history_v2';
    const KEY_THEME = 'cdac_ml_quiz_theme_v2';
    const KEY_BLANKS = 'cdac_ml_blanks_answers_v2';

    // Application State
    let currentQuestions = [...RAW_QUESTIONS];
    let currentIndex = 0;
    let userAnswers = {}; // qId -> selectedIndex
    let markedQuestions = new Set(); // qIds bookmarked
    let activeTrackFilter = 'all';
    let activeSessionFilter = 'all';
    let activeStatusFilter = 'all'; // 'all' | 'mistakes' | 'marked' | 'unanswered'
    let isExamMode = false;
    let timerSeconds = 0;
    let timerInterval = null;
    let isTimerRunning = true;
    let currentActiveTab = 'quiz';

    // Vercel Custom Event Tracking
    function trackEvent(name, data = {}) {
      if (typeof window.va === 'function') {
        try { window.va('event', { name, data }); } catch (e) {}
      }
    }

    // Toast Helper
    function showToast(message, type = 'info') {
      const container = document.getElementById('toastContainer');
      const toast = document.createElement('div');
      toast.className = 'toast';
      const icon = type === 'success' ? '✅' : type === 'warning' ? '⚠️' : type === 'error' ? '❌' : 'ℹ️';
      toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
      container.appendChild(toast);
      setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        toast.style.transition = 'all 0.25s ease';
        setTimeout(() => toast.remove(), 250);
      }, 3000);
    }

    // Theme Management
    function initTheme() {
      const savedTheme = localStorage.getItem(KEY_THEME) || 'dark';
      applyTheme(savedTheme);
    }

    function applyTheme(theme) {
      const htmlEl = document.documentElement;
      const iconEl = document.getElementById('themeIcon');
      if (theme === 'light') {
        htmlEl.classList.remove('dark');
        htmlEl.classList.add('light');
        if (iconEl) iconEl.textContent = '🌙';
      } else {
        htmlEl.classList.remove('light');
        htmlEl.classList.add('dark');
        if (iconEl) iconEl.textContent = '☀️';
      }
      localStorage.setItem(KEY_THEME, theme);
    }

    document.getElementById('themeToggleBtn').addEventListener('click', () => {
      const current = document.documentElement.classList.contains('light') ? 'light' : 'dark';
      applyTheme(current === 'light' ? 'dark' : 'light');
    });

    // Tab Switching
    function switchTab(tabId) {
      currentActiveTab = tabId;
      document.querySelectorAll('.tab-btn').forEach(btn => {
        if (btn.getAttribute('data-tab') === tabId) {
          btn.classList.add('active');
        } else {
          btn.classList.remove('active');
        }
      });

      document.querySelectorAll('.tab-view').forEach(view => {
        view.classList.remove('active');
      });

      const targetView = document.getElementById(tabId + 'View');
      if (targetView) targetView.classList.add('active');

      if (tabId === 'attempts') {
        renderAttemptsDashboard();
      } else if (tabId === 'blanks') {
        renderInteractiveBlanks();
      }

      window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Timer Logic
    function startTimer() {
      if (timerInterval) clearInterval(timerInterval);
      timerInterval = setInterval(() => {
        if (isTimerRunning) {
          timerSeconds++;
          updateTimerDisplay();
        }
      }, 1000);
    }

    function updateTimerDisplay() {
      const mins = Math.floor(timerSeconds / 60).toString().padStart(2, '0');
      const secs = (timerSeconds % 60).toString().padStart(2, '0');
      const el = document.getElementById('timerText');
      if (el) el.textContent = `${mins}:${secs}`;
    }

    document.getElementById('timerDisplay').addEventListener('click', () => {
      isTimerRunning = !isTimerRunning;
      showToast(isTimerRunning ? 'Timer resumed' : 'Timer paused');
    });

    // Populate Session Dropdown
    function populateSessionDropdown() {
      const dropdown = document.getElementById('sessionDropdown');
      const sessions = Array.from(new Set(RAW_QUESTIONS.map(q => q.session))).sort();
      dropdown.innerHTML = '<option value="all">All Sessions</option>';
      sessions.forEach(sess => {
        const opt = document.createElement('option');
        opt.value = sess;
        opt.textContent = sess;
        dropdown.appendChild(opt);
      });
    }

    // Local Storage: Save Active Session
    function saveActiveSession() {
      try {
        const state = {
          userAnswers,
          markedQuestions: Array.from(markedQuestions),
          activeTrackFilter,
          activeSessionFilter,
          activeStatusFilter,
          currentIndex,
          isExamMode,
          timerSeconds
        };
        localStorage.setItem(KEY_SESSION, JSON.stringify(state));
        updateStorageBadge();
      } catch (e) {
        console.warn('Storage save failed:', e);
      }
    }

    function updateStorageBadge() {
      const badge = document.getElementById('storageStatusBadge');
      const count = Object.keys(userAnswers).length;
      if (badge) {
        badge.innerHTML = `💾 <span>${count}/66 Saved</span>`;
      }
    }

    // Local Storage: Restore Active Session
    function restoreActiveSession() {
      try {
        const raw = localStorage.getItem(KEY_SESSION);
        if (!raw) return;
        const state = JSON.parse(raw);
        if (state.userAnswers) userAnswers = state.userAnswers;
        if (state.markedQuestions) markedQuestions = new Set(state.markedQuestions);
        if (state.activeTrackFilter) activeTrackFilter = state.activeTrackFilter;
        if (state.activeSessionFilter) activeSessionFilter = state.activeSessionFilter;
        if (state.activeStatusFilter) activeStatusFilter = state.activeStatusFilter;
        if (typeof state.currentIndex === 'number') currentIndex = state.currentIndex;
        if (typeof state.isExamMode === 'boolean') isExamMode = state.isExamMode;
        if (typeof state.timerSeconds === 'number') timerSeconds = state.timerSeconds;

        const dropdown = document.getElementById('sessionDropdown');
        if (dropdown && state.activeSessionFilter) {
          dropdown.value = state.activeSessionFilter;
        }

        updateQuizModeButton();
      } catch (e) {
        console.warn('Storage restore failed:', e);
      }
    }

    // Filter Logic
    function applyFilters() {
      currentQuestions = RAW_QUESTIONS.filter(q => {
        if (activeTrackFilter !== 'all' && q.track !== activeTrackFilter) return false;
        if (activeSessionFilter !== 'all' && q.session !== activeSessionFilter) return false;
        if (activeStatusFilter === 'mistakes') {
          const ans = userAnswers[q.id];
          return ans !== undefined && ans !== q.correctIndex;
        } else if (activeStatusFilter === 'marked') {
          return markedQuestions.has(q.id);
        } else if (activeStatusFilter === 'unanswered') {
          return userAnswers[q.id] === undefined;
        }
        return true;
      });

      if (currentIndex >= currentQuestions.length) {
        currentIndex = 0;
      }

      updateFilterPillStates();
      updateScoreboard();
      renderCurrentCard();
      renderJumpGrid();
    }

    function setTrackFilter(track) {
      activeTrackFilter = track;
      document.querySelectorAll('[data-track]').forEach(btn => {
        if (btn.getAttribute('data-track') === track) btn.classList.add('active');
        else btn.classList.remove('active');
      });
      currentIndex = 0;
      saveActiveSession();
      applyFilters();
    }

    function onSessionDropdownChange(val) {
      activeSessionFilter = val;
      currentIndex = 0;
      saveActiveSession();
      applyFilters();
    }

    function setStatusFilter(status) {
      activeStatusFilter = status;
      currentIndex = 0;
      saveActiveSession();
      applyFilters();
    }

    function updateFilterPillStates() {
      const pillAll = document.getElementById('pillAllStatus');
      const pillMistakes = document.getElementById('pillMistakes');
      const pillMarked = document.getElementById('pillMarked');
      const pillUnanswered = document.getElementById('pillUnanswered');

      [pillAll, pillMistakes, pillMarked, pillUnanswered].forEach(p => p.classList.remove('active'));

      if (activeStatusFilter === 'all') pillAll.classList.add('active');
      else if (activeStatusFilter === 'mistakes') pillMistakes.classList.add('active');
      else if (activeStatusFilter === 'marked') pillMarked.classList.add('active');
      else if (activeStatusFilter === 'unanswered') pillUnanswered.classList.add('active');
    }

    // Quiz Mode Toggle (Practice vs Exam)
    function toggleQuizMode() {
      isExamMode = !isExamMode;
      updateQuizModeButton();
      renderCurrentCard();
      saveActiveSession();
      showToast(isExamMode ? 'Exam Mode: Instant feedback hidden until finished' : 'Practice Mode: Instant answers & explanations active');
    }

    function updateQuizModeButton() {
      const icon = document.getElementById('quizModeIcon');
      const text = document.getElementById('quizModeText');
      if (isExamMode) {
        icon.textContent = '⏱️';
        text.textContent = 'Exam Mode (Submit to Grade)';
      } else {
        icon.textContent = '💡';
        text.textContent = 'Practice Mode (Instant Feedback)';
      }
    }

    // Scoreboard Calculation
    function updateScoreboard() {
      let correct = 0;
      let wrong = 0;
      let answered = 0;
      let totalMistakesInAll = 0;

      RAW_QUESTIONS.forEach(q => {
        const ans = userAnswers[q.id];
        if (ans !== undefined) {
          answered++;
          if (ans === q.correctIndex) correct++;
          else {
            wrong++;
            totalMistakesInAll++;
          }
        }
      });

      const total = RAW_QUESTIONS.length;
      const accuracy = answered > 0 ? Math.round((correct / answered) * 100) : 0;
      const progressPercent = Math.round((answered / total) * 100);

      document.getElementById('correctScore').textContent = correct;
      document.getElementById('wrongScore').textContent = wrong;
      document.getElementById('accuracyScore').textContent = `${accuracy}%`;
      document.getElementById('progressBarFill').style.width = `${progressPercent}%`;

      document.getElementById('currentQNum').textContent = currentQuestions.length > 0 ? currentIndex + 1 : 0;
      document.getElementById('totalQNum').textContent = currentQuestions.length;

      document.getElementById('mistakesCountDisplay').textContent = totalMistakesInAll;
      document.getElementById('markedCountDisplay').textContent = markedQuestions.size;
      document.getElementById('unansweredCountDisplay').textContent = total - answered;

      const navMistakesBadge = document.getElementById('navMistakesBadge');
      if (navMistakesBadge) navMistakesBadge.textContent = totalMistakesInAll;

      const attemptsMistakesCount = document.getElementById('attemptsMistakesCount');
      if (attemptsMistakesCount) attemptsMistakesCount.textContent = totalMistakesInAll;

      const metricActiveMistakes = document.getElementById('metricActiveMistakes');
      if (metricActiveMistakes) metricActiveMistakes.textContent = totalMistakesInAll;
    }

    // Render Current Question Card
    function renderCurrentCard() {
      const card = document.getElementById('quizCard');
      if (currentQuestions.length === 0) {
        card.innerHTML = `
          <div style="text-align: center; padding: 3rem 1rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎉</div>
            <h3 style="font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem;">No questions match this filter!</h3>
            <p style="color: var(--text-muted); margin-bottom: 1.5rem;">
              ${activeStatusFilter === 'mistakes' ? 'You have cleared all mistakes in this section! Excellent work.' : 'Try selecting a different filter or session.'}
            </p>
            <button class="btn btn-primary" onclick="setStatusFilter('all')">Show All Questions</button>
          </div>
        `;
        return;
      }

      const q = currentQuestions[currentIndex];
      const selectedIdx = userAnswers[q.id];
      const isAnswered = selectedIdx !== undefined;
      const isMistake = isAnswered && selectedIdx !== q.correctIndex;

      // Meta tags
      document.getElementById('cardTrackBadge').textContent = q.track;
      document.getElementById('cardSessionBadge').textContent = q.session;
      document.getElementById('cardSourceBadge').textContent = `${q.sourceDeck} (${q.sourceSlide})`;
      
      const mistakeFlag = document.getElementById('cardMistakeFlag');
      if (mistakeFlag) mistakeFlag.style.display = isMistake ? 'inline-block' : 'none';

      // Bookmark button
      const bookmarkBtn = document.getElementById('cardBookmarkBtn');
      const starIcon = document.getElementById('bookmarkStarIcon');
      if (markedQuestions.has(q.id)) {
        bookmarkBtn.classList.add('marked');
        starIcon.textContent = '★';
      } else {
        bookmarkBtn.classList.remove('marked');
        starIcon.textContent = '☆';
      }

      // Web search quick link
      const webSearchBtn = document.getElementById('cardWebSearchBtn');
      const searchQuery = encodeURIComponent(`CDAC "${q.track}" ${q.session} ${q.question.slice(0, 50)}`);
      webSearchBtn.href = `https://www.google.com/search?q=${searchQuery}`;

      // Prompt
      document.getElementById('cardQuestionPrompt').textContent = `Q${q.id}. ${q.question}`;

      // Options
      const optionsList = document.getElementById('cardOptionsList');
      optionsList.innerHTML = '';

      q.options.forEach((optText, idx) => {
        const btn = document.createElement('button');
        btn.className = 'option-btn';
        
        const keySpan = document.createElement('span');
        keySpan.className = 'option-key';
        keySpan.textContent = String.fromCharCode(65 + idx);

        const textSpan = document.createElement('span');
        textSpan.className = 'option-text';
        textSpan.textContent = optText;

        btn.appendChild(keySpan);
        btn.appendChild(textSpan);

        if (isExamMode) {
          if (selectedIdx === idx) {
            btn.classList.add('exam-selected');
          }
          btn.addEventListener('click', () => handleOptionClick(q.id, idx));
        } else {
          if (isAnswered) {
            if (idx === q.correctIndex) {
              btn.classList.add(selectedIdx === idx ? 'selected-correct' : 'reveal-correct');
            } else if (selectedIdx === idx) {
              btn.classList.add('selected-wrong');
            }
            btn.disabled = true;
          } else {
            btn.addEventListener('click', () => handleOptionClick(q.id, idx));
          }
        }

        optionsList.appendChild(btn);
      });

      // Explanation box
      const explanationBox = document.getElementById('cardExplanationBox');
      const explanationTitle = document.getElementById('cardExplanationTitle');
      const explanationBody = document.getElementById('cardExplanationBody');
      const explanationStatus = document.getElementById('cardAnswerStatusText');
      const deckRef = document.getElementById('cardDeckRef');

      if (!isExamMode && isAnswered) {
        explanationBox.style.display = 'block';
        if (selectedIdx === q.correctIndex) {
          explanationBox.classList.remove('wrong-banner');
          explanationTitle.classList.remove('wrong');
          explanationStatus.textContent = '✅ Correct Answer!';
        } else {
          explanationBox.classList.add('wrong-banner');
          explanationTitle.classList.add('wrong');
          explanationStatus.textContent = '❌ Incorrect Choice';
        }
        explanationBody.textContent = q.explanation;
        deckRef.textContent = `Source: ${q.sourceDeck} • ${q.sourceSlide}`;
      } else {
        explanationBox.style.display = 'none';
      }

      // Button states
      document.getElementById('prevQuestionBtn').disabled = currentIndex === 0;
      document.getElementById('nextQuestionBtn').disabled = currentIndex === currentQuestions.length - 1;
    }

    // Option Selection Handler
    function handleOptionClick(qId, selectedIdx) {
      userAnswers[qId] = selectedIdx;
      saveActiveSession();
      updateScoreboard();
      renderCurrentCard();
      renderJumpGrid();
      trackEvent('answer_question', { qId, selectedIdx });
    }

    // Navigation Arrows
    function goToPrevQuestion() {
      if (currentIndex > 0) {
        currentIndex--;
        saveActiveSession();
        renderCurrentCard();
        renderJumpGrid();
      }
    }

    function goToNextQuestion() {
      if (currentIndex < currentQuestions.length - 1) {
        currentIndex++;
        saveActiveSession();
        renderCurrentCard();
        renderJumpGrid();
      }
    }

    function jumpToQuestionIndex(idx) {
      currentIndex = idx;
      saveActiveSession();
      renderCurrentCard();
      renderJumpGrid();
    }

    // Bookmark Toggle
    function toggleCurrentBookmark() {
      const q = currentQuestions[currentIndex];
      if (!q) return;
      if (markedQuestions.has(q.id)) {
        markedQuestions.delete(q.id);
        showToast('Question removed from review list');
      } else {
        markedQuestions.add(q.id);
        showToast('Question starred for review! ★');
      }
      saveActiveSession();
      updateScoreboard();
      renderCurrentCard();
      renderJumpGrid();
    }

    // Jump Grid Rendering
    function renderJumpGrid() {
      const grid = document.getElementById('jumpGrid');
      grid.innerHTML = '';

      currentQuestions.forEach((q, idx) => {
        const btn = document.createElement('button');
        btn.className = 'jump-btn';
        btn.textContent = q.id;

        if (idx === currentIndex) btn.classList.add('current');

        const ans = userAnswers[q.id];
        if (ans !== undefined) {
          if (!isExamMode) {
            if (ans === q.correctIndex) btn.classList.add('correct');
            else btn.classList.add('wrong');
          } else {
            btn.classList.add('correct');
          }
        }

        if (markedQuestions.has(q.id)) {
          btn.classList.add('marked');
        }

        btn.addEventListener('click', () => jumpToQuestionIndex(idx));
        grid.appendChild(btn);
      });

      const answeredCount = currentQuestions.filter(q => userAnswers[q.id] !== undefined).length;
      document.getElementById('jumpSummaryText').textContent = `${answeredCount} / ${currentQuestions.length} answered`;
    }

    // Shuffle Order
    function shuffleCurrentQuestions() {
      for (let i = currentQuestions.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [currentQuestions[i], currentQuestions[j]] = [currentQuestions[j], currentQuestions[i]];
      }
      currentIndex = 0;
      saveActiveSession();
      renderCurrentCard();
      renderJumpGrid();
      showToast('Questions shuffled! 🔀');
    }

    // Reset Current Session
    function resetActiveQuizSession() {
      if (confirm('Are you sure you want to reset your current answers and timer on this device? (Past saved attempts will remain safe)')) {
        userAnswers = {};
        timerSeconds = 0;
        currentIndex = 0;
        saveActiveSession();
        updateScoreboard();
        renderCurrentCard();
        renderJumpGrid();
        showToast('Active quiz session reset');
      }
    }

    // Web Search for Current Question Concept
    function searchCurrentConceptOnline() {
      const q = currentQuestions[currentIndex];
      if (!q) return;
      const url = `https://www.google.com/search?q=${encodeURIComponent('Machine learning statistics ' + q.question)}`;
      window.open(url, '_blank');
    }

    // ==========================================
    // ATTEMPT HISTORY & MISTAKES PERSISTENCE     //
    // ==========================================

    function loadAttemptsHistory() {
      try {
        const raw = localStorage.getItem(KEY_ATTEMPTS);
        return raw ? JSON.parse(raw) : [];
      } catch (e) {
        return [];
      }
    }

    function saveAttemptsHistory(attempts) {
      try {
        localStorage.setItem(KEY_ATTEMPTS, JSON.stringify(attempts));
      } catch (e) {
        console.warn('Could not save attempts history:', e);
      }
    }

    // Finish & Record Attempt
    function finishAndSaveAttempt() {
      const answeredKeys = Object.keys(userAnswers);
      if (answeredKeys.length === 0) {
        alert('Please answer at least one question before recording an attempt!');
        return;
      }

      let correctCount = 0;
      let wrongCount = 0;
      const wrongQuestions = [];
      const correctQuestionIds = [];

      RAW_QUESTIONS.forEach(q => {
        const selected = userAnswers[q.id];
        if (selected !== undefined) {
          if (selected === q.correctIndex) {
            correctCount++;
            correctQuestionIds.push(q.id);
          } else {
            wrongCount++;
            wrongQuestions.push({
              id: q.id,
              track: q.track,
              session: q.session,
              question: q.question,
              selectedText: q.options[selected] || 'N/A',
              correctText: q.options[q.correctIndex],
              explanation: q.explanation,
              sourceDeck: q.sourceDeck,
              sourceSlide: q.sourceSlide
            });
          }
        }
      });

      const totalAttempted = correctCount + wrongCount;
      const accuracyPercent = Math.round((correctCount / totalAttempted) * 100);
      const mins = Math.floor(timerSeconds / 60);
      const secs = timerSeconds % 60;
      const timeSpentFormatted = `${mins}m ${secs}s`;

      const newAttempt = {
        id: 'att_' + Date.now(),
        timestamp: Date.now(),
        dateString: new Date().toLocaleString(),
        trackFilter: activeTrackFilter === 'all' ? 'All Tracks' : activeTrackFilter,
        totalQuestions: totalAttempted,
        correctCount,
        wrongCount,
        accuracyPercent,
        timeSpentFormatted,
        timeSeconds: timerSeconds,
        markedAnswers: { ...userAnswers },
        wrongQuestions,
        correctQuestionIds
      };

      const history = loadAttemptsHistory();
      history.unshift(newAttempt);
      saveAttemptsHistory(history);

      trackEvent('finish_attempt', { score: correctCount, total: totalAttempted, accuracy: accuracyPercent });

      showAttemptResultModal(newAttempt);
    }

    function showAttemptResultModal(attempt) {
      const modal = document.getElementById('attemptResultModal');
      document.getElementById('modalScoreVal').textContent = `${attempt.correctCount} / ${attempt.totalQuestions}`;
      document.getElementById('modalAccuracyVal').textContent = `${attempt.accuracyPercent}%`;
      document.getElementById('modalTimeVal').textContent = attempt.timeSpentFormatted;

      const emoji = attempt.accuracyPercent >= 90 ? '🏆' : attempt.accuracyPercent >= 75 ? '🎉' : attempt.accuracyPercent >= 50 ? '📚' : '💪';
      const title = attempt.accuracyPercent >= 90 ? 'Outstanding Mastery!' : attempt.accuracyPercent >= 75 ? 'Great Performance!' : 'Attempt Recorded!';
      document.getElementById('modalResultEmoji').textContent = emoji;
      document.getElementById('modalResultTitle').textContent = title;

      const mistakesBanner = document.getElementById('modalMistakesBanner');
      const mistakesBtn = document.getElementById('modalRetestMistakesBtn');

      if (attempt.wrongCount > 0) {
        mistakesBanner.style.display = 'block';
        document.getElementById('modalMistakesCountText').textContent = `${attempt.wrongCount} question(s) need revision`;
        mistakesBtn.style.display = 'block';
      } else {
        mistakesBanner.style.display = 'none';
        mistakesBtn.style.display = 'none';
      }

      modal.classList.add('active');
    }

    function closeAttemptResultModal() {
      document.getElementById('attemptResultModal').classList.remove('active');
    }

    function retestMistakesFromModal() {
      closeAttemptResultModal();
      practiceCurrentMistakes();
    }

    function viewAttemptInHistoryFromModal() {
      closeAttemptResultModal();
      switchTab('attempts');
    }

    function practiceCurrentMistakes() {
      activeStatusFilter = 'mistakes';
      updateFilterPillStates();
      currentIndex = 0;
      applyFilters();
      switchTab('quiz');
      showToast('Loaded mistake review mode: Practice questions answered incorrectly!');
    }

    function renderAttemptsDashboard() {
      const history = loadAttemptsHistory();
      const tbody = document.getElementById('attemptsTableBody');

      document.getElementById('metricTotalAttempts').textContent = history.length;

      if (history.length > 0) {
        const bestAcc = Math.max(...history.map(a => a.accuracyPercent));
        const avgAcc = Math.round(history.reduce((acc, a) => acc + a.accuracyPercent, 0) / history.length);
        document.getElementById('metricBestScore').textContent = `${bestAcc}%`;
        document.getElementById('metricAvgScore').textContent = `${avgAcc}%`;

        tbody.innerHTML = '';
        history.forEach((att, idx) => {
          const tr = document.createElement('tr');
          const grade = att.accuracyPercent >= 90 ? '🏆 Distinction' : att.accuracyPercent >= 70 ? '🎯 Pass' : '📚 Review Needed';
          const gradeColor = att.accuracyPercent >= 90 ? 'var(--success)' : att.accuracyPercent >= 70 ? 'var(--info)' : 'var(--warning)';

          tr.innerHTML = `
            <td style="font-weight: 700;">#${history.length - idx}</td>
            <td style="font-size: 0.825rem; color: var(--text-muted);">${att.dateString}</td>
            <td><span class="badge-tag badge-track">${att.trackFilter}</span></td>
            <td style="font-weight: 700;">${att.correctCount} / ${att.totalQuestions}</td>
            <td style="font-weight: 800; color: ${gradeColor};">${att.accuracyPercent}%</td>
            <td style="font-family: monospace; font-size: 0.85rem;">${att.timeSpentFormatted}</td>
            <td><span style="font-weight: 700; color: ${gradeColor};">${grade}</span></td>
            <td>
              <div style="display: flex; gap: 0.35rem;">
                <button class="btn btn-sm" onclick="inspectAttempt('${att.id}')">👁️ Inspect</button>
                ${att.wrongCount > 0 ? `<button class="btn btn-sm btn-warning" onclick="retestMistakesFromAttempt('${att.id}')">🔄 Retest</button>` : ''}
                <button class="btn btn-sm btn-danger" onclick="deleteAttempt('${att.id}')">🗑️</button>
              </div>
            </td>
          `;
          tbody.appendChild(tr);
        });
      } else {
        document.getElementById('metricBestScore').textContent = '0%';
        document.getElementById('metricAvgScore').textContent = '0%';
        tbody.innerHTML = `
          <tr>
            <td colspan="8" style="text-align: center; color: var(--text-muted); padding: 2rem;">
              No attempts recorded yet. Answer questions in the MCQ Quiz and click "Finish & Record Attempt"!
            </td>
          </tr>
        `;
      }
    }

    function inspectAttempt(attemptId) {
      const history = loadAttemptsHistory();
      const att = history.find(a => a.id === attemptId);
      if (!att) return;

      document.getElementById('inspectModalTitle').textContent = `Attempt #${attemptId.replace('att_', '')} Breakdown`;
      document.getElementById('inspectModalSubtitle').textContent = `${att.dateString} • Score: ${att.correctCount}/${att.totalQuestions} (${att.accuracyPercent}%) • Duration: ${att.timeSpentFormatted}`;

      const listContainer = document.getElementById('inspectQuestionsList');
      listContainer.innerHTML = '';

      RAW_QUESTIONS.forEach(q => {
        const userChoice = att.markedAnswers ? att.markedAnswers[q.id] : undefined;
        if (userChoice !== undefined) {
          const isCorrect = userChoice === q.correctIndex;
          const card = document.createElement('div');
          card.className = `explanation-box ${isCorrect ? '' : 'wrong-banner'}`;
          card.style.margin = '0';
          card.innerHTML = `
            <div class="explanation-title ${isCorrect ? '' : 'wrong'}">
              <span>Q${q.id}. ${isCorrect ? '✅ Answered Correctly' : '❌ Answered Incorrectly'}</span>
              <span class="badge-tag badge-track">${q.track}</span>
            </div>
            <div style="font-weight: 600; margin-bottom: 0.5rem; color: var(--text);">${q.question}</div>
            <div style="font-size: 0.85rem; margin-bottom: 0.35rem;">
              <strong>Your Pick:</strong> <span style="color: ${isCorrect ? 'var(--success)' : 'var(--error)'};">${q.options[userChoice]}</span>
            </div>
            ${!isCorrect ? `<div style="font-size: 0.85rem; margin-bottom: 0.5rem; color: var(--success);"><strong>Correct Answer:</strong> ${q.options[q.correctIndex]}</div>` : ''}
            <div style="font-size: 0.825rem; color: var(--text-muted); line-height: 1.5; border-top: 1px solid var(--surface-border); padding-top: 0.4rem; margin-top: 0.4rem;">
              <strong>Explanation:</strong> ${q.explanation}
            </div>
          `;
          listContainer.appendChild(card);
        }
      });

      const retestBtn = document.getElementById('inspectRetestBtn');
      if (att.wrongCount > 0) {
        retestBtn.style.display = 'inline-flex';
        retestBtn.onclick = () => {
          closeInspectModal();
          retestMistakesFromAttempt(att.id);
        };
      } else {
        retestBtn.style.display = 'none';
      }

      document.getElementById('attemptInspectModal').classList.add('active');
    }

    function closeInspectModal() {
      document.getElementById('attemptInspectModal').classList.remove('active');
    }

    function retestMistakesFromAttempt(attemptId) {
      const history = loadAttemptsHistory();
      const att = history.find(a => a.id === attemptId);
      if (!att || !att.wrongQuestions || att.wrongQuestions.length === 0) return;

      const wrongIds = new Set(att.wrongQuestions.map(w => w.id));
      
      wrongIds.forEach(id => {
        delete userAnswers[id];
      });

      currentQuestions = RAW_QUESTIONS.filter(q => wrongIds.has(q.id));
      currentIndex = 0;
      activeStatusFilter = 'all';
      saveActiveSession();
      updateScoreboard();
      renderCurrentCard();
      renderJumpGrid();
      switchTab('quiz');
      showToast(`Loaded ${wrongIds.size} mistake questions from Attempt for retesting! 🎯`);
    }

    function deleteAttempt(attemptId) {
      if (confirm('Delete this attempt record from your device?')) {
        let history = loadAttemptsHistory();
        history = history.filter(a => a.id !== attemptId);
        saveAttemptsHistory(history);
        renderAttemptsDashboard();
        showToast('Attempt record deleted');
      }
    }

    function clearAllAttemptsHistory() {
      if (confirm('Are you sure you want to delete ALL past attempts stored on this device? This cannot be undone.')) {
        localStorage.removeItem(KEY_ATTEMPTS);
        renderAttemptsDashboard();
        showToast('All attempt history cleared');
      }
    }

    function exportAttemptsData() {
      const history = loadAttemptsHistory();
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(history, null, 2));
      const dlAnchor = document.createElement('a');
      dlAnchor.setAttribute("href", dataStr);
      dlAnchor.setAttribute("download", `CDAC_ML_Quiz_Attempts_${Date.now()}.json`);
      document.body.appendChild(dlAnchor);
      dlAnchor.click();
      dlAnchor.remove();
      showToast('Exported attempts data as JSON file! 📥');
    }

    function importAttemptsData(event) {
      const file = event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const imported = JSON.parse(e.target.result);
          if (Array.isArray(imported)) {
            saveAttemptsHistory(imported);
            renderAttemptsDashboard();
            showToast('Successfully imported attempt history! 📤');
          } else {
            alert('Invalid JSON structure. Expected an array of attempts.');
          }
        } catch (err) {
          alert('Failed to parse JSON file.');
        }
      };
      reader.readAsText(file);
    }

    // ==========================================
    // INTERACTIVE FILL-IN-THE-BLANKS ENGINE      //
    // ==========================================

    let blanksUserAnswers = {};
    let allBlanksRevealed = false;

    function renderInteractiveBlanks() {
      try {
        const saved = localStorage.getItem(KEY_BLANKS);
        if (saved) blanksUserAnswers = JSON.parse(saved);
      } catch (e) {}

      const container = document.getElementById('interactiveBlanksContainer');
      container.innerHTML = '';

      const sections = {};
      RAW_BLANKS.forEach(b => {
        if (!sections[b.sectionNum]) {
          sections[b.sectionNum] = { title: b.sectionTitle, source: b.source, items: [] };
        }
        sections[b.sectionNum].items.push(b);
      });

      Object.keys(sections).forEach(sNum => {
        const sec = sections[sNum];
        const secBlock = document.createElement('div');
        secBlock.id = `fib-sec-${sNum}`;
        secBlock.style.marginBottom = '2.5rem';

        const headerBox = document.createElement('div');
        headerBox.className = 'scoreboard-box';
        headerBox.style.margin = '0 0 1.25rem 0';
        headerBox.innerHTML = `
          <div>
            <h3 style="font-size: 1.15rem; font-weight: 800;">${sec.title}</h3>
            <span style="font-size: 0.75rem; color: var(--text-muted);">${sec.source}</span>
          </div>
          <span class="status-badge">${sec.items.length} Practice Items</span>
        `;
        secBlock.appendChild(headerBox);

        sec.items.forEach(item => {
          const card = document.createElement('article');
          card.className = 'fib-card';
          const savedAnswer = blanksUserAnswers[item.id] || '';

          const displayPrompt = item.prompt.replace('______', '<span style="color: var(--accent); font-weight: 800;">[ ______ ]</span>');

          card.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
              <span class="badge-tag badge-track">Item ${item.id}</span>
              <span style="font-size: 0.75rem; color: var(--text-muted);">${item.source}</span>
            </div>
            <p class="fib-prompt">${displayPrompt}</p>
            <div class="blank-input-group">
              <input type="text" id="blank_input_${item.id.replace('.', '_')}" class="blank-input" placeholder="Type your answer..." value="${savedAnswer}" onkeydown="if(event.key==='Enter') checkBlankAnswer('${item.id}')">
              <button class="btn btn-sm btn-primary" onclick="checkBlankAnswer('${item.id}')">Check Answer</button>
              <button class="btn btn-sm" onclick="toggleBlankExplanation('${item.id}')">👁️ Toggle Notes</button>
              <span id="blank_feedback_${item.id.replace('.', '_')}" style="font-size: 0.85rem; font-weight: 700; margin-left: 0.5rem;"></span>
            </div>
            <div id="blank_expl_${item.id.replace('.', '_')}" class="explanation-box" style="display: ${allBlanksRevealed ? 'block' : 'none'};">
              <div class="explanation-title">
                <span>💡 Key Solution: <strong style="color: var(--success);">${item.answer}</strong></span>
              </div>
              <div class="explanation-body" style="white-space: pre-line; font-size: 0.85rem;">
                ${item.explanation}
              </div>
            </div>
          `;
          secBlock.appendChild(card);
        });

        container.appendChild(secBlock);
      });
    }

    function checkBlankAnswer(itemId) {
      const safeId = itemId.replace('.', '_');
      const inputEl = document.getElementById(`blank_input_${safeId}`);
      const feedbackEl = document.getElementById(`blank_feedback_${safeId}`);
      const explEl = document.getElementById(`blank_expl_${safeId}`);

      if (!inputEl) return;
      const val = inputEl.value.trim().toLowerCase();
      blanksUserAnswers[itemId] = inputEl.value.trim();
      localStorage.setItem(KEY_BLANKS, JSON.stringify(blanksUserAnswers));

      const item = RAW_BLANKS.find(b => b.id === itemId);
      if (!item) return;

      const isCorrect = item.aliases.some(alias => {
        const cleanAlias = alias.toLowerCase().trim();
        return val === cleanAlias || val.includes(cleanAlias) || cleanAlias.includes(val);
      }) && val.length > 1;

      if (isCorrect) {
        inputEl.className = 'blank-input correct';
        feedbackEl.innerHTML = '✅ Correct!';
        feedbackEl.style.color = 'var(--success)';
        explEl.style.display = 'block';
        showToast(`Item ${itemId}: Correct! 🎉`, 'success');
      } else {
        inputEl.className = 'blank-input wrong';
        feedbackEl.innerHTML = '❌ Try again or toggle notes';
        feedbackEl.style.color = 'var(--error)';
        explEl.style.display = 'block';
      }
    }

    function toggleBlankExplanation(itemId) {
      const safeId = itemId.replace('.', '_');
      const explEl = document.getElementById(`blank_expl_${safeId}`);
      if (explEl) {
        explEl.style.display = explEl.style.display === 'none' ? 'block' : 'none';
      }
    }

    function toggleAllBlanksAnswers() {
      allBlanksRevealed = !allBlanksRevealed;
      document.querySelectorAll('[id^="blank_expl_"]').forEach(el => {
        el.style.display = allBlanksRevealed ? 'block' : 'none';
      });
      document.getElementById('toggleBlanksLabel').textContent = allBlanksRevealed ? 'Hide All Explanations' : 'Reveal All Answers';
    }

    function resetBlanksPractice() {
      if (confirm('Clear all typed answers in the blanks section?')) {
        blanksUserAnswers = {};
        localStorage.removeItem(KEY_BLANKS);
        renderInteractiveBlanks();
        showToast('Cleared blanks inputs');
      }
    }

    // ==========================================
    // INTERACTIVE STATISTICAL TEST SELECTOR      //
    // ==========================================

    const TEST_SCENARIOS = {
      '1-sample': {
        title: 'One-Sample t-Test (or Z-Test if σ is known)',
        outcome: 'Continuous numeric metric (e.g., student test scores, process cycle time)',
        formula: 't = (X̄ - μ₀) / (s / √n)',
        assumptions: 'Data normally distributed (or n ≥ 30 by Central Limit Theorem). Random sample.',
        python: 'from scipy import stats\\nstat, pval = stats.ttest_1samp(sample_data, popmean=mu_0)',
        nonParametric: 'Wilcoxon Signed-Rank Test (if data is severely skewed/non-normal)'
      },
      '2-independent': {
        title: 'Two-Sample Independent t-Test (Student t-test or Welch t-test)',
        outcome: 'Continuous numeric metric compared across 2 independent distinct cohorts (e.g. Website A vs Website B)',
        formula: 't = (X̄₁ - X̄₂) / √(s₁²/n₁ + s₂²/n₂)',
        assumptions: 'Independent samples, approximate normality in each group, equal variances (use Welch t-test if variances unequal).',
        python: 'from scipy import stats\\nstat, pval = stats.ttest_ind(group_a, group_b, equal_var=False)  # Welch t-test',
        nonParametric: 'Mann-Whitney U Test (Wilcoxon Rank-Sum Test)'
      },
      '2-paired': {
        title: 'Paired Samples t-Test (Dependent / Repeated Measures)',
        outcome: 'Continuous measurement collected twice on the EXACT SAME subjects (e.g., Patient blood pressure Before vs After medication)',
        formula: 't = d̄ / (s_d / √n), where d = (X_after - X_before)',
        assumptions: 'The pair-wise differences (d) are approximately normally distributed.',
        python: 'from scipy import stats\\nstat, pval = stats.ttest_rel(before_scores, after_scores)',
        nonParametric: 'Wilcoxon Signed-Rank Test for Paired Samples'
      },
      '3-plus': {
        title: 'One-Way Analysis of Variance (ANOVA)',
        outcome: 'Continuous numeric target compared across 3 or more independent categorical groups (e.g., Fertilizer A vs B vs C)',
        formula: 'F = MS_between / MS_within = (SS_between / (k-1)) / (SS_within / (N-k))',
        assumptions: 'Independent samples, normal distribution within each group, homogeneity of variances (homoscedasticity).',
        python: 'from scipy import stats\\nstat, pval = stats.f_oneway(group_1, group_2, group_3)',
        nonParametric: 'Kruskal-Wallis H-Test'
      },
      'categorical': {
        title: 'Chi-Square (χ²) Test of Independence',
        outcome: 'Relationship between two categorical features (e.g., Device Type [Mobile, Desktop] vs Purchase Decision [Yes, No])',
        formula: 'χ² = Σ [(O - E)² / E], where E = (Row_Total × Col_Total) / Grand_Total',
        assumptions: 'Observations are independent counts; expected frequency in each contingency cell is at least 5.',
        python: 'from scipy import stats\\ncontingency_table = pd.crosstab(df["Device"], df["Purchased"])\\nchi2, pval, dof, expected = stats.chi2_contingency(contingency_table)',
        nonParametric: 'Fisher Exact Test (for 2x2 tables with small expected cell counts < 5)'
      }
    };

    function selectTestScenario(key) {
      const data = TEST_SCENARIOS[key];
      if (!data) return;
      const box = document.getElementById('testScenarioResult');
      box.style.display = 'block';
      box.innerHTML = `
        <div class="explanation-title">
          <span>🎯 Recommended Test: <strong>${data.title}</strong></span>
        </div>
        <div class="explanation-body">
          <p style="margin-bottom: 0.4rem;"><strong>Data Scenario:</strong> ${data.outcome}</p>
          <p style="margin-bottom: 0.4rem; font-family: monospace;"><strong>Mathematical Formula:</strong> ${data.formula}</p>
          <p style="margin-bottom: 0.4rem;"><strong>Key Assumptions:</strong> ${data.assumptions}</p>
          <p style="margin-bottom: 0.4rem; color: var(--info);"><strong>Non-Parametric Alternative:</strong> ${data.nonParametric}</p>
          <div style="background: var(--bg); border: 1px solid var(--surface-border); border-radius: 8px; padding: 0.75rem; margin-top: 0.5rem; font-family: monospace; font-size: 0.8rem; overflow-x: auto;">
            ${data.python.replace(/\\n/g, '<br/>')}
          </div>
        </div>
      `;
    }

    // ==========================================
    // GLOBAL SEARCH & WEB RESEARCH HUB           //
    // ==========================================

    document.getElementById('globalSearchBtn').addEventListener('click', () => {
      switchTab('research');
      document.getElementById('liveSearchInput').focus();
    });

    function applySearchTerm(term) {
      const input = document.getElementById('liveSearchInput');
      input.value = term;
      executeLiveSearch(term);
    }

    function executeLiveSearch(query) {
      const q = query.trim().toLowerCase();
      const resultsContainer = document.getElementById('liveSearchResults');
      const resultsList = document.getElementById('searchResultsList');
      const countBadge = document.getElementById('searchResultsCount');

      if (!q || q.length < 2) {
        resultsContainer.style.display = 'none';
        return;
      }

      resultsContainer.style.display = 'block';
      resultsList.innerHTML = '';

      let matchCount = 0;

      // 1. Search MCQs
      RAW_QUESTIONS.forEach(item => {
        const fullText = (item.question + ' ' + item.options.join(' ') + ' ' + item.explanation + ' ' + item.session + ' ' + item.track).toLowerCase();
        if (fullText.includes(q)) {
          matchCount++;
          const card = document.createElement('div');
          card.className = 'quiz-card';
          card.style.padding = '1.25rem';
          card.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
              <span class="badge-tag badge-track">MCQ Q${item.id} • ${item.track}</span>
              <button class="btn btn-sm btn-primary" onclick="jumpToSpecificMCQ(${item.id})">Jump to Quiz</button>
            </div>
            <div style="font-weight: 700; margin-bottom: 0.5rem;">${highlightText(item.question, q)}</div>
            <div style="font-size: 0.85rem; color: var(--success); font-weight: 600; margin-bottom: 0.4rem;">
              Correct: ${highlightText(item.options[item.correctIndex], q)}
            </div>
            <div style="font-size: 0.825rem; color: var(--text-muted); line-height: 1.5;">
              ${highlightText(item.explanation, q)}
            </div>
          `;
          resultsList.appendChild(card);
        }
      });

      // 2. Search Blanks
      RAW_BLANKS.forEach(item => {
        const fullText = (item.prompt + ' ' + item.answer + ' ' + item.explanation + ' ' + item.sectionTitle).toLowerCase();
        if (fullText.includes(q)) {
          matchCount++;
          const card = document.createElement('div');
          card.className = 'quiz-card';
          card.style.padding = '1.25rem';
          card.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
              <span class="badge-tag badge-source">Blank Item ${item.id} • ${item.sectionTitle}</span>
              <button class="btn btn-sm" onclick="switchTab('blanks')">Go to Blanks</button>
            </div>
            <div style="font-weight: 700; margin-bottom: 0.5rem;">${highlightText(item.prompt, q)}</div>
            <div style="font-size: 0.85rem; color: var(--success); font-weight: 700; margin-bottom: 0.4rem;">
              Key Answer: ${highlightText(item.answer, q)}
            </div>
            <div style="font-size: 0.825rem; color: var(--text-muted); line-height: 1.5;">
              ${highlightText(item.explanation, q)}
            </div>
          `;
          resultsList.appendChild(card);
        }
      });

      countBadge.textContent = `${matchCount} matches found`;
      if (matchCount === 0) {
        resultsList.innerHTML = `
          <div style="text-align: center; padding: 2rem; color: var(--text-muted);">
            No local matches found for "${query}". Click below to launch an external web search!
            <div style="margin-top: 1rem;">
              <button class="btn btn-primary" onclick="launchWebSearch('google')">Search "${query}" on Google</button>
            </div>
          </div>
        `;
      }
    }

    function highlightText(text, keyword) {
      if (!keyword) return text;
      const regex = new RegExp(`(${keyword.replace(/[-[\\]{}()*+?.,\\\\^$|#\\s]/g, '\\\\$&')})`, 'gi');
      return text.replace(regex, '<mark style="background: rgba(99,102,241,0.35); color: #fff; padding: 0.1rem 0.2rem; border-radius: 3px;">$1</mark>');
    }

    function jumpToSpecificMCQ(qId) {
      activeTrackFilter = 'all';
      activeSessionFilter = 'all';
      activeStatusFilter = 'all';
      currentQuestions = [...RAW_QUESTIONS];
      const targetIdx = currentQuestions.findIndex(q => q.id === qId);
      if (targetIdx !== -1) {
        currentIndex = targetIdx;
        switchTab('quiz');
        renderCurrentCard();
        renderJumpGrid();
      }
    }

    function launchWebSearch(engine) {
      const query = document.getElementById('liveSearchInput').value.trim() || 'CDAC Machine Learning Statistics Exam Revision';
      let url = '';
      if (engine === 'google') {
        url = `https://www.google.com/search?q=${encodeURIComponent(query + ' data science')}`;
      } else if (engine === 'wikipedia') {
        url = `https://en.wikipedia.org/wiki/Special:Search?search=${encodeURIComponent(query)}`;
      } else if (engine === 'arxiv') {
        url = `https://arxiv.org/search/?query=${encodeURIComponent(query)}&searchtype=all`;
      } else if (engine === 'sklearn') {
        url = `https://scikit-learn.org/stable/search.html?q=${encodeURIComponent(query)}`;
      } else if (engine === 'statquest') {
        url = `https://www.youtube.com/results?search_query=${encodeURIComponent('StatQuest ' + query)}`;
      }
      window.open(url, '_blank');
      trackEvent('web_search', { engine, query });
    }

    // Keyboard Shortcuts
    document.addEventListener('keydown', (e) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;

      if (e.key === 'ArrowLeft') {
        goToPrevQuestion();
      } else if (e.key === 'ArrowRight') {
        goToNextQuestion();
      } else if (['1', '2', '3', '4'].includes(e.key)) {
        const optIdx = parseInt(e.key, 10) - 1;
        const q = currentQuestions[currentIndex];
        if (q && userAnswers[q.id] === undefined && optIdx < q.options.length) {
          handleOptionClick(q.id, optIdx);
        }
      } else if (['a', 'b', 'c', 'd', 'A', 'B', 'C', 'D'].includes(e.key)) {
        const optIdx = e.key.toUpperCase().charCodeAt(0) - 65;
        const q = currentQuestions[currentIndex];
        if (q && userAnswers[q.id] === undefined && optIdx < q.options.length) {
          handleOptionClick(q.id, optIdx);
        }
      } else if (e.key === 'm' || e.key === 'M') {
        toggleCurrentBookmark();
      }
    });

    // Initialize Application
    initTheme();
    populateSessionDropdown();
    restoreActiveSession();
    applyFilters();
    startTimer();
    renderInteractiveBlanks();
  </script>
</body>
</html>'''

# Replace placeholders
final_html = template.replace('__QUESTIONS_JSON__', questions_json_str)
final_html = final_html.replace('__BLANKS_JSON__', blanks_json_str)
final_html = final_html.replace('__REV_BODY__', rev_body)
final_html = final_html.replace('__FIB_EXTRA__', fib_extra_sections)

with open(os.path.join(BASE_DIR, "index.html"), "w", encoding="utf-8") as f:
    f.write(final_html)

print(f"Generated index.html successfully! Size: {len(final_html)} characters.")
