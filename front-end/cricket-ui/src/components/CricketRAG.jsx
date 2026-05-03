import { useState, useRef, useEffect, useCallback, Component } from "react";

// Issue #14 — env-var driven URL so deployment outside localhost works.
// Vite: set VITE_BACKEND_URL in .env  |  CRA: set REACT_APP_BACKEND_URL
const BACKEND_URL =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_BACKEND_URL) ||
  (typeof process !== "undefined" && process.env?.REACT_APP_BACKEND_URL) ||
  "http://localhost:8000";

const PLAYFAIR = "'Playfair Display', Georgia, serif";
const JETBRAINS = "'JetBrains Mono', monospace";

const styles = `
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400;1,700&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600;1,8..60,300;1,8..60,400&family=JetBrains+Mono:wght@400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #fff;
    color: #000;
    font-family: 'Source Serif 4', Georgia, serif;
    height: 100vh;
    overflow: hidden;
  }

  .app {
    display: flex;
    height: 100vh;
    background: #fff;
  }

  /* SIDEBAR */
  .sidebar {
    width: 260px;
    min-width: 260px;
    border-right: 2px solid #000;
    display: flex;
    flex-direction: column;
    background: #000;
    color: #fff;
    position: relative;
    overflow: hidden;
  }

  .sidebar::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image: repeating-linear-gradient(90deg, transparent, transparent 1px, #fff 1px, #fff 2px);
    background-size: 4px 100%;
    opacity: 0.03;
    pointer-events: none;
  }

  .sidebar-header {
    padding: 24px 20px 16px;
    border-bottom: 1px solid #333;
  }

  .sidebar-logo {
    font-family: ${PLAYFAIR};
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 20px;
  }

  .sidebar-logo span { opacity: 0.5; font-weight: 400; }

  .new-chat-btn {
    width: 100%;
    padding: 10px 14px;
    background: transparent;
    border: 1px solid #444;
    color: #fff;
    font-family: ${JETBRAINS};
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: all 80ms;
  }

  .new-chat-btn:hover {
    background: #fff;
    color: #000;
    border-color: #fff;
  }

  .new-chat-btn svg { flex-shrink: 0; }

  /* SIDEBAR NAV */
  .sidebar-nav {
    padding: 12px 0 4px;
    border-bottom: 1px solid #222;
  }

  .sidebar-nav-label {
    font-family: ${JETBRAINS};
    font-size: 9px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #555;
    padding: 4px 20px 8px;
  }

  .sidebar-nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 20px;
    cursor: pointer;
    font-family: ${JETBRAINS};
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #666;
    border-left: 2px solid transparent;
    transition: all 80ms;
  }

  .sidebar-nav-item:hover { color: #fff; background: #111; border-left-color: #555; }
  .sidebar-nav-item.active { color: #fff; background: #1a1a1a; border-left-color: #fff; }
  .sidebar-nav-item svg { flex-shrink: 0; opacity: 0.7; }

  .chat-list {
    flex: 1;
    overflow-y: auto;
    padding: 12px 0;
    scrollbar-width: thin;
    scrollbar-color: #333 transparent;
  }

  .chat-list-label {
    font-family: ${JETBRAINS};
    font-size: 9px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #555;
    padding: 4px 20px 8px;
  }

  .chat-item {
    display: flex;
    align-items: center;
    gap: 0;
    cursor: pointer;
    border-left: 2px solid transparent;
    transition: all 80ms;
    position: relative;
  }

  .chat-item:hover { border-left-color: #555; background: #111; }
  .chat-item.active { border-left-color: #fff; background: #1a1a1a; }

  .chat-item-content {
    flex: 1;
    padding: 10px 14px 10px 16px;
    overflow: hidden;
  }

  .chat-item-title {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 13px;
    color: #fff;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.4;
  }

  .chat-item.active .chat-item-title { color: #fff; }

  .chat-item-meta {
    font-family: ${JETBRAINS};
    font-size: 10px;
    color: #555;
    margin-top: 2px;
  }

  .chat-item-delete {
    padding: 10px 12px;
    background: transparent;
    border: none;
    color: #444;
    cursor: pointer;
    opacity: 0;
    transition: all 80ms;
    flex-shrink: 0;
    display: flex;
    align-items: center;
  }

  .chat-item:hover .chat-item-delete { opacity: 1; }
  .chat-item-delete:hover { color: #fff; }

  /* MAIN */
  .main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
  }

  .main-header {
    padding: 20px 40px 16px;
    border-bottom: 1px solid #e5e5e5;
    display: flex;
    align-items: baseline;
    gap: 16px;
    flex-shrink: 0;
  }

  .main-title {
    font-family: ${PLAYFAIR};
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #000;
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .main-title-empty {
    font-family: ${PLAYFAIR};
    font-size: 22px;
    font-weight: 400;
    font-style: italic;
    color: #bbb;
  }

  .status-pill {
    font-family: ${JETBRAINS};
    font-size: 10px;
    letter-spacing: 0.08em;
    padding: 4px 10px;
    border: 1px solid #e5e5e5;
    color: #888;
    text-transform: uppercase;
    flex-shrink: 0;
  }

  .status-pill.healthy { border-color: #000; color: #000; }

  /* MESSAGES */
  .messages-area {
    flex: 1;
    overflow-y: auto;
    padding: 32px 40px;
    scrollbar-width: thin;
    scrollbar-color: #e5e5e5 transparent;
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    text-align: center;
    padding: 40px;
  }

  .empty-headline {
    font-family: ${PLAYFAIR};
    font-size: 56px;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1;
    color: #000;
    margin-bottom: 4px;
  }

  .empty-sub {
    font-family: ${PLAYFAIR};
    font-size: 22px;
    font-weight: 400;
    font-style: italic;
    color: #888;
    margin-bottom: 32px;
  }

  .empty-rule {
    width: 60px;
    height: 4px;
    background: #000;
    margin-bottom: 24px;
  }

  .empty-hints {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-width: 480px;
    width: 100%;
  }

  .hint-btn {
    padding: 12px 20px;
    border: 1px solid #e5e5e5;
    background: transparent;
    text-align: left;
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 14px;
    color: #555;
    cursor: pointer;
    transition: all 80ms;
    line-height: 1.4;
  }

  .hint-btn:hover {
    border-color: #000;
    color: #000;
    background: #f5f5f5;
  }

  /* MESSAGE BUBBLES */
  .message { margin-bottom: 32px; }

  .msg-user {
    display: flex;
    justify-content: flex-end;
  }

  .msg-user-bubble {
    max-width: 70%;
    padding: 14px 20px;
    background: #000;
    color: #fff;
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 15px;
    line-height: 1.6;
  }

  .msg-assistant { display: flex; flex-direction: column; gap: 0; }

  .msg-label {
    font-family: ${JETBRAINS};
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .msg-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e5e5e5;
  }

  .msg-answer {
    border-left: 4px solid #000;
    padding: 16px 24px;
    background: #fafafa;
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 16px;
    line-height: 1.75;
    color: #000;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .msg-answer .phase-breakdown {
    display: inline-block;
    background: #f0f0f0;
    padding: 4px 8px;
    border-radius: 4px;
    margin: 2px 4px 2px 0;
    font-family: ${JETBRAINS};
    font-size: 12px;
    color: #333;
  }

  .msg-answer .phase-label {
    font-weight: 600;
    color: #000;
  }

  .msg-meta {
    display: flex;
    gap: 16px;
    margin-top: 10px;
    padding-left: 4px;
    flex-wrap: wrap;
  }

  .meta-tag {
    font-family: ${JETBRAINS};
    font-size: 10px;
    letter-spacing: 0.08em;
    color: #888;
    text-transform: uppercase;
    border: 1px solid #e5e5e5;
    padding: 3px 8px;
  }

  .meta-tag.aggregate { border-color: #000; color: #000; }

  /* LOADING */
  .msg-loading {
    border-left: 4px solid #000;
    padding: 16px 24px;
    background: #fafafa;
  }

  .loading-dots {
    display: flex;
    gap: 6px;
    align-items: center;
  }

  .loading-dot {
    width: 6px;
    height: 6px;
    background: #000;
    animation: blink 1.2s infinite;
  }

  .loading-dot:nth-child(2) { animation-delay: 0.2s; }
  .loading-dot:nth-child(3) { animation-delay: 0.4s; }

  @keyframes blink {
    0%, 80%, 100% { opacity: 0.2; }
    40% { opacity: 1; }
  }

  /* ERROR */
  .msg-error {
    border-left: 4px solid #000;
    padding: 16px 24px;
    background: #fafafa;
    font-family: ${JETBRAINS};
    font-size: 12px;
    color: #666;
    letter-spacing: 0.02em;
  }

  /* INPUT */
  .input-area {
    border-top: 2px solid #000;
    padding: 20px 40px 24px;
    flex-shrink: 0;
    background: #fff;
  }

  .input-row {
    display: flex;
    gap: 0;
    border: 2px solid #000;
    transition: border-color 80ms;
  }

  .input-row:focus-within { border-color: #000; }

  .query-input {
    flex: 1;
    padding: 14px 18px;
    border: none;
    outline: none;
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 15px;
    color: #000;
    background: transparent;
    resize: none;
    min-height: 52px;
    max-height: 200px;
    line-height: 1.5;
  }

  .query-input::placeholder { color: #bbb; font-style: italic; }

  .send-btn {
    padding: 0 22px;
    background: #000;
    border: none;
    color: #fff;
    cursor: pointer;
    font-family: ${JETBRAINS};
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    transition: all 80ms;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
    border-left: 2px solid #000;
  }

  .send-btn:hover:not(:disabled) { background: #333; }
  .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .input-hint {
    margin-top: 8px;
    font-family: ${JETBRAINS};
    font-size: 10px;
    color: #bbb;
    letter-spacing: 0.08em;
  }

  /* SCROLLBAR */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #e5e5e5; }

  /* ============================================================
     LEADERBOARD PANEL
  ============================================================ */
  .panel {
    flex: 1;
    overflow-y: auto;
    padding: 32px 40px;
    scrollbar-width: thin;
    scrollbar-color: #e5e5e5 transparent;
  }

  .panel-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 28px;
    padding-bottom: 16px;
    border-bottom: 2px solid #000;
  }

  .panel-title {
    font-family: ${PLAYFAIR};
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #000;
  }

  .panel-subtitle {
    font-family: ${JETBRAINS};
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    margin-top: 4px;
  }

  .refresh-btn {
    padding: 8px 16px;
    background: transparent;
    border: 1px solid #000;
    font-family: ${JETBRAINS};
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: all 80ms;
  }

  .refresh-btn:hover { background: #000; color: #fff; }
  .refresh-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .leaderboard-section {
    margin-bottom: 40px;
  }

  .leaderboard-section-title {
    font-family: ${JETBRAINS};
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #e5e5e5;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .leaderboard-section-title .title-badge {
    background: #000;
    color: #fff;
    padding: 2px 8px;
    font-size: 9px;
    letter-spacing: 0.15em;
  }

  .leaderboard-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 14px;
  }

  .leaderboard-table th {
    text-align: left;
    padding: 10px 12px;
    font-family: ${JETBRAINS};
    font-size: 9px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-bottom: 2px solid #000;
    color: #555;
    background: #fafafa;
  }

  .leaderboard-table th:not(:first-child) { text-align: right; }

  .leaderboard-table td {
    padding: 11px 12px;
    border-bottom: 1px solid #f0f0f0;
    vertical-align: middle;
  }

  .leaderboard-table td:not(:first-child):not(:nth-child(2)) { text-align: right; }

  .leaderboard-table tr:hover td { background: #fafafa; }

  .leaderboard-table tr:nth-child(1) .rank-num { color: #a67c52; font-weight: 500; }
  .leaderboard-table tr:nth-child(2) .rank-num { color: #888; font-weight: 500; }
  .leaderboard-table tr:nth-child(3) .rank-num { color: #b08050; font-weight: 500; }

  .rank-num {
    font-family: ${JETBRAINS};
    font-size: 11px;
    color: #ccc;
    width: 32px;
    display: inline-block;
  }

  .rank-bar {
    position: relative;
    display: inline-block;
    width: 12px;
    height: 12px;
    vertical-align: middle;
    margin-right: 4px;
  }

  .player-name {
    font-weight: 600;
    font-size: 14px;
    color: #000;
  }

  .player-name-btn {
    background: none;
    border: none;
    font-family: inherit;
    font-size: inherit;
    font-weight: inherit;
    color: inherit;
    cursor: pointer;
    text-decoration: underline;
    text-decoration-color: #e0e0e0;
    padding: 0;
    transition: text-decoration-color 80ms;
  }

  .player-name-btn:hover { text-decoration-color: #000; }

  .stat-num {
    font-family: ${JETBRAINS};
    font-size: 12px;
    color: #333;
  }

  .stat-highlight {
    font-family: ${JETBRAINS};
    font-size: 13px;
    font-weight: 500;
    color: #000;
  }

  .stat-dim {
    font-family: ${JETBRAINS};
    font-size: 11px;
    color: #aaa;
  }

  .mini-bar-cell { width: 80px; }

  .mini-bar {
    height: 3px;
    background: #000;
    display: block;
  }

  .loading-placeholder {
    text-align: center;
    padding: 60px 0;
    color: #999;
    font-family: ${JETBRAINS};
    font-size: 11px;
    letter-spacing: 0.1em;
  }

  .loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 2px solid #e5e5e5;
    border-top-color: #000;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-bottom: 12px;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  /* ============================================================
     COMPARE PANEL
  ============================================================ */
  .compare-panel {
    flex: 1;
    overflow-y: auto;
    padding: 32px 40px;
  }

  .compare-form {
    display: flex;
    gap: 12px;
    margin-bottom: 32px;
    align-items: flex-end;
  }

  .compare-field {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .compare-field-label {
    font-family: ${JETBRAINS};
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #666;
  }

  .compare-input {
    padding: 12px 16px;
    border: 2px solid #e5e5e5;
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 14px;
    outline: none;
    transition: border-color 80ms;
    width: 100%;
  }

  .compare-input:focus { border-color: #000; }
  .compare-input::placeholder { color: #ccc; font-style: italic; }

  .compare-btn {
    padding: 12px 28px;
    background: #000;
    border: none;
    color: #fff;
    font-family: ${JETBRAINS};
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 80ms;
    height: 47px;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .compare-btn:hover:not(:disabled) { background: #333; }
  .compare-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .vs-divider {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
    padding-bottom: 2px;
    flex-shrink: 0;
  }

  .vs-text {
    font-family: ${PLAYFAIR};
    font-size: 13px;
    font-weight: 700;
    font-style: italic;
    color: #ccc;
    height: 47px;
    display: flex;
    align-items: center;
  }

  /* Comparison result grid */
  .comparison-result {
    animation: fadeIn 300ms ease;
  }

  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

  .comparison-summary {
    background: #000;
    color: #fff;
    padding: 16px 24px;
    margin-bottom: 24px;
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 15px;
    line-height: 1.6;
    border-left: 4px solid #fff;
  }

  .comparison-grid {
    display: grid;
    grid-template-columns: 1fr 40px 1fr;
    gap: 0;
  }

  .player-col {
    display: flex;
    flex-direction: column;
  }

  .player-col-header {
    padding: 14px 20px;
    background: #000;
    color: #fff;
    font-family: ${PLAYFAIR};
    font-size: 16px;
    font-weight: 700;
    border-bottom: 2px solid #000;
  }

  .player-col-header.right { text-align: right; }

  .vs-col {
    display: flex;
    align-items: stretch;
    justify-content: center;
    position: relative;
  }

  .vs-col::before {
    content: '';
    position: absolute;
    top: 0;
    bottom: 0;
    left: 50%;
    width: 2px;
    background: #000;
    transform: translateX(-50%);
  }

  .stat-section-header {
    font-family: ${JETBRAINS};
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #888;
    padding: 12px 20px 8px;
    background: #f5f5f5;
    border-bottom: 1px solid #e5e5e5;
    border-top: 2px solid #000;
    margin-top: 16px;
  }

  .stat-section-header:first-of-type { margin-top: 0; border-top: none; }

  .compare-stat-row {
    display: contents;
  }

  .compare-stat-cell {
    padding: 12px 20px;
    border-bottom: 1px solid #f0f0f0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .compare-stat-cell.right { text-align: right; }

  .compare-stat-label {
    font-family: ${JETBRAINS};
    font-size: 9px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #aaa;
  }

  .compare-stat-value {
    font-family: ${JETBRAINS};
    font-size: 16px;
    font-weight: 500;
    color: #000;
  }

  .compare-stat-value.winner {
    color: #000;
    position: relative;
  }

  .compare-stat-value.winner::after {
    content: '▲';
    font-size: 8px;
    color: #000;
    margin-left: 4px;
    vertical-align: middle;
  }

  .compare-stat-value.loser {
    color: #bbb;
  }

  .vs-col-spacer {
    border-bottom: 1px solid #f0f0f0;
    background: #fff;
    position: relative;
    z-index: 1;
  }

  /* Career lookup */
  .career-lookup {
    margin-top: 40px;
    padding-top: 32px;
    border-top: 2px solid #000;
  }

  .career-lookup-title {
    font-family: ${PLAYFAIR};
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 16px;
  }

  .career-result {
    margin-top: 20px;
    animation: fadeIn 300ms ease;
  }

  .career-player-name {
    font-family: ${PLAYFAIR};
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 4px;
  }

  .career-matches-badge {
    font-family: ${JETBRAINS};
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    border: 1px solid #e5e5e5;
    padding: 3px 10px;
    display: inline-block;
    margin-bottom: 20px;
  }

  .career-stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #000;
    margin-bottom: 16px;
  }

  .career-stat-box {
    background: #fff;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .career-stat-box-label {
    font-family: ${JETBRAINS};
    font-size: 9px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
  }

  .career-stat-box-value {
    font-family: ${PLAYFAIR};
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #000;
    line-height: 1;
  }

  .career-section-label {
    font-family: ${JETBRAINS};
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #fff;
    background: #000;
    padding: 6px 16px;
    display: inline-block;
    margin-bottom: 4px;
  }

  /* ============================================================
     ANALYTICS PANEL - Pace vs Spin
  ============================================================ */
  .analytics-panel {
    flex: 1;
    overflow-y: auto;
    padding: 32px 40px;
    scrollbar-width: thin;
    scrollbar-color: #e5e5e5 transparent;
  }

  .phase-selector {
    display: flex;
    gap: 0;
    margin-bottom: 32px;
    border: 2px solid #000;
    width: fit-content;
  }

  .phase-btn {
    padding: 10px 24px;
    background: transparent;
    border: none;
    color: #666;
    font-family: ${JETBRAINS};
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 80ms;
    border-right: 1px solid #000;
  }

  .phase-btn:last-child { border-right: none; }
  .phase-btn:hover { background: #f5f5f5; color: #000; }
  .phase-btn.active { background: #000; color: #fff; }

  .analytics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 40px;
  }

  .stat-card {
    background: #fff;
    border: 2px solid #000;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .stat-card-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding-bottom: 12px;
    border-bottom: 1px solid #e5e5e5;
  }

  .stat-card-title {
    font-family: ${PLAYFAIR};
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #000;
  }

  .stat-card-subtitle {
    font-family: ${JETBRAINS};
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #888;
  }

  .stat-card-body {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .stat-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 16px;
  }

  .stat-row-label {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 14px;
    color: #555;
  }

  .stat-row-value {
    font-family: ${JETBRAINS};
    font-size: 16px;
    font-weight: 500;
    color: #000;
  }

  .stat-row-value.highlight {
    font-size: 20px;
    font-weight: 700;
  }

  .stat-row-value.dim {
    color: #888;
  }

  .comparison-bar {
    height: 4px;
    background: #f0f0f0;
    border-radius: 2px;
    overflow: hidden;
    display: flex;
  }

  .comparison-bar-fill {
    height: 100%;
    background: #000;
    transition: width 300ms ease;
  }

  .comparison-bar-fill.pace { background: #000; }
  .comparison-bar-fill.spin { background: #333; }

  /* Top Bowlers List - Newspaper Style */
  .top-bowlers-section {
    margin-top: 40px;
    padding-top: 32px;
    border-top: 2px solid #000;
  }

  .top-bowlers-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 24px;
  }

  .top-bowlers-title {
    font-family: ${PLAYFAIR};
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #000;
  }

  .top-bowlers-subtitle {
    font-family: ${JETBRAINS};
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
  }

  .top-bowlers-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }

  .bowler-list {
    border: 1px solid #e5e5e5;
  }

  .bowler-list-header {
    background: #000;
    color: #fff;
    padding: 12px 16px;
    font-family: ${JETBRAINS};
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .bowler-list-header::before {
    content: '';
    width: 8px;
    height: 8px;
    background: #fff;
    border-radius: 50%;
  }

  .bowler-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 16px;
    border-bottom: 1px solid #f0f0f0;
    transition: background 80ms;
  }

  .bowler-item:last-child { border-bottom: none; }
  .bowler-item:hover { background: #fafafa; }

  .bowler-rank {
    font-family: ${JETBRAINS};
    font-size: 12px;
    font-weight: 500;
    color: #888;
    width: 24px;
    text-align: center;
  }

  .bowler-rank.top-3 {
    color: #000;
    font-weight: 700;
  }

  .bowler-info {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .bowler-name {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 14px;
    font-weight: 600;
    color: #000;
  }

  .bowler-stats {
    font-family: ${JETBRAINS};
    font-size: 11px;
    color: #666;
    display: flex;
    gap: 12px;
  }

  .bowler-stat-item {
    display: flex;
    align-items: baseline;
    gap: 4px;
  }

  .bowler-stat-label {
    color: #999;
    font-size: 9px;
    text-transform: uppercase;
  }

  .bowler-stat-value {
    font-weight: 500;
  }

  .bowler-stat-value.highlight {
    color: #000;
  }

  /* Enhanced phase breakdown in messages */
  .msg-answer .phase-stats-block {
    background: #f5f5f5;
    border-left: 3px solid #000;
    padding: 12px 16px;
    margin: 12px 0;
    font-family: ${JETBRAINS};
    font-size: 12px;
  }

  .msg-answer .phase-stats-header {
    font-weight: 600;
    color: #000;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .msg-answer .phase-stats-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid #e5e5e5;
  }

  .msg-answer .phase-stats-row:last-child { border-bottom: none; }

  .msg-answer .aggregate-summary {
    background: #000;
    color: #fff;
    padding: 12px 16px;
    margin: 12px 0;
    font-family: ${JETBRAINS};
    font-size: 11px;
    line-height: 1.6;
  }

  .msg-answer .aggregate-summary strong {
    font-weight: 700;
    color: #fff;
  }

  .msg-answer .pace-vs-spin-block {
    border: 1px solid #000;
    padding: 16px;
    margin: 16px 0;
    background: #fafafa;
  }

  .msg-answer .pace-vs-spin-header {
    font-family: ${PLAYFAIR};
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .msg-answer .pace-vs-spin-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #e5e5e5;
  }

  .msg-answer .pace-vs-spin-row:last-child { border-bottom: none; }

  .msg-answer .pace-vs-spin-label {
    font-weight: 600;
  }

  .msg-answer .pace-vs-spin-value {
    font-family: ${JETBRAINS};
  }
`;

const HINTS = [
  "Who won the most matches in the dataset?",
  "Which player scored the most runs as top scorer?",
  "List all matches where the winner was determined by runs.",
  "Who was Man of the Match the most number of times?",
];

let chatIdCounter = 1;

function generateId() {
  return `chat-${Date.now()}-${chatIdCounter++}`;
}

function formatTime(ts) {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function LoadingBlock() {
  return (
    <div className="loading-placeholder">
      <div className="loading-spinner" />
      <div>Aggregating player data…</div>
    </div>
  );
}

// Mini bar visualisation for tables
function MiniBar({ value, max }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return <span className="mini-bar" style={{ width: `${pct}%` }} />;
}

// Format phase breakdowns in message content
// Issue #15 — XSS fix: formatMessageContent now returns an array of typed
// segment objects instead of raw HTML strings. The companion component
// FormattedMessage renders them using safe React elements only — no
// dangerouslySetInnerHTML anywhere in the render path.
function formatMessageContent(content) {
  if (!content) return [];

  const segments = [];
  const lines = content.split("\n");

  const phasePattern = /^(Powerplay|Middle|Death):\s*(\d+\/\d+)/i;
  const paceSpinHeader = /PACE vs SPIN/i;
  const paceRow = /^Pace\s*[→:]/i;
  const spinRow = /^Spin\s*[→:]/i;
  const runsHeader = /TOTAL RUNS ACROSS ALL INNINGS/i;
  const wicketsHeader = /TOTAL WICKETS ACROSS ALL INNINGS/i;

  for (const line of lines) {
    if (phasePattern.test(line)) {
      const [, phase, stats] = line.match(phasePattern);
      segments.push({ type: "phase", phase, stats });
    } else if (paceSpinHeader.test(line)) {
      segments.push({ type: "section-header", text: "Pace vs Spin Breakdown" });
    } else if (paceRow.test(line)) {
      segments.push({ type: "pace-row", text: line.replace(/^Pace\s*[→:]/i, "").trim() });
    } else if (spinRow.test(line)) {
      segments.push({ type: "spin-row", text: line.replace(/^Spin\s*[→:]/i, "").trim() });
    } else if (runsHeader.test(line)) {
      segments.push({ type: "stats-header", text: "Total Runs Across All Innings" });
    } else if (wicketsHeader.test(line)) {
      segments.push({ type: "stats-header", text: "Total Wickets Across All Innings" });
    } else {
      segments.push({ type: "text", text: line });
    }
  }

  return segments;
}

// Issue #15 — safe React renderer for formatted message segments.
// Replaces the single dangerouslySetInnerHTML call with typed React elements.
// No HTML strings are ever injected into the DOM.
function FormattedMessage({ content }) {
  const segments = formatMessageContent(content);
  return (
    <div>
      {segments.map((seg, i) => {
        switch (seg.type) {
          case "phase":
            return (
              <span key={i} className="phase-breakdown">
                <span className="phase-label">
                  {seg.phase.charAt(0).toUpperCase() + seg.phase.slice(1).toLowerCase()}:
                </span>{" "}
                {seg.stats}
              </span>
            );
          case "section-header":
            return <div key={i} className="pace-vs-spin-header">{seg.text}</div>;
          case "pace-row":
            return (
              <div key={i} className="pace-vs-spin-row">
                <span className="pace-vs-spin-label">Pace:</span>{" "}
                <span className="pace-vs-spin-value">{seg.text}</span>
              </div>
            );
          case "spin-row":
            return (
              <div key={i} className="pace-vs-spin-row">
                <span className="pace-vs-spin-label">Spin:</span>{" "}
                <span className="pace-vs-spin-value">{seg.text}</span>
              </div>
            );
          case "stats-header":
            return <div key={i} className="phase-stats-header">{seg.text}</div>;
          case "text":
          default:
            return <span key={i}>{seg.text}{i < segments.length - 1 ? "\n" : ""}</span>;
        }
      })}
    </div>
  );
}

// ─── ERROR BOUNDARY (Issue #16) ───────────────────────────────────────────────
// Wraps each analytics panel so a runtime error in one panel does not crash
// the entire app. React error boundaries must be class components.
class PanelErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error("[PanelErrorBoundary] Panel crashed:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          padding: "2rem",
          border: "1px solid #c00",
          borderRadius: "6px",
          color: "#c00",
          background: "#fff5f5",
          fontFamily: "monospace",
          fontSize: "0.85rem",
        }}>
          <strong>⚠ Panel Error</strong>
          <p style={{ marginTop: "0.5rem", color: "#555" }}>
            This panel encountered an unexpected error and could not render.
          </p>
          <p style={{ marginTop: "0.25rem", color: "#888", fontSize: "0.75rem" }}>
            {this.state.error?.message || "Unknown error"}
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              marginTop: "1rem", padding: "0.4rem 1rem",
              background: "#000", color: "#fff",
              border: "none", borderRadius: "4px", cursor: "pointer",
            }}
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// ─── LEADERBOARD PANEL ─────────────────────────────────────────────────────────
function LeaderboardPanel({ leaderboard, loading, onRefresh, onSelectPlayer }) {
  const maxRuns = leaderboard.batting[0]?.runs || 1;
  const maxWickets = leaderboard.bowling[0]?.wickets || 1;

  return (
    <div className="panel">
      <div className="panel-header">
        <div>
          <div className="panel-title">Leaderboard</div>
          <div className="panel-subtitle">Career aggregates across all indexed matches</div>
        </div>
        <button className="refresh-btn" onClick={onRefresh} disabled={loading}>
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path d="M10 6A4 4 0 1 1 6 2a4 4 0 0 1 3.65 2.35" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round"/>
            <polyline points="10,1 10,4.35 6.65,4.35" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>

      {loading ? (
        <LoadingBlock />
      ) : (
        <>
          {/* BATTING */}
          <div className="leaderboard-section">
            <div className="leaderboard-section-title">
              <span className="title-badge">Bat</span>
              Top Run Scorers
            </div>
            {leaderboard.batting.length === 0 ? (
              <div className="loading-placeholder" style={{ padding: "30px 0" }}>No data — click Refresh to load.</div>
            ) : (
              <table className="leaderboard-table">
                <thead>
                  <tr>
                    <th style={{ width: 32 }}>#</th>
                    <th>Player</th>
                    <th>Runs</th>
                    <th>Inns</th>
                    <th>Avg</th>
                    <th>SR</th>
                    <th>4s</th>
                    <th>6s</th>
                    <th style={{ width: 80 }}></th>
                  </tr>
                </thead>
                <tbody>
                  {leaderboard.batting.slice(0, 10).map((p, i) => (
                    <tr key={p.name}>
                      <td><span className="rank-num">{i + 1}</span></td>
                      <td>
                        <button className="player-name player-name-btn" onClick={() => onSelectPlayer(p.name, "compare")}>
                          {p.name}
                        </button>
                      </td>
                      <td><span className="stat-highlight">{p.runs.toLocaleString()}</span></td>
                      <td><span className="stat-num">{p.innings}</span></td>
                      <td><span className="stat-num">{p.average}</span></td>
                      <td><span className="stat-num">{p.strike_rate}</span></td>
                      <td><span className="stat-dim">{p.fours}</span></td>
                      <td><span className="stat-dim">{p.sixes}</span></td>
                      <td className="mini-bar-cell">
                        <MiniBar value={p.runs} max={maxRuns} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          {/* BOWLING */}
          <div className="leaderboard-section">
            <div className="leaderboard-section-title">
              <span className="title-badge">Bowl</span>
              Top Wicket Takers
            </div>
            {leaderboard.bowling.length === 0 ? (
              <div className="loading-placeholder" style={{ padding: "30px 0" }}>No data — click Refresh to load.</div>
            ) : (
              <table className="leaderboard-table">
                <thead>
                  <tr>
                    <th style={{ width: 32 }}>#</th>
                    <th>Player</th>
                    <th>Wkts</th>
                    <th>Inns</th>
                    <th>Avg</th>
                    <th>Econ</th>
                    <th>Runs</th>
                    <th style={{ width: 80 }}></th>
                  </tr>
                </thead>
                <tbody>
                  {leaderboard.bowling.slice(0, 10).map((p, i) => (
                    <tr key={p.name}>
                      <td><span className="rank-num">{i + 1}</span></td>
                      <td>
                        <button className="player-name player-name-btn" onClick={() => onSelectPlayer(p.name, "compare")}>
                          {p.name}
                        </button>
                      </td>
                      <td><span className="stat-highlight">{p.wickets}</span></td>
                      <td><span className="stat-num">{p.innings}</span></td>
                      <td><span className="stat-num">{p.average}</span></td>
                      <td><span className="stat-num">{p.economy}</span></td>
                      <td><span className="stat-dim">{p.runs_conceded.toLocaleString()}</span></td>
                      <td className="mini-bar-cell">
                        <MiniBar value={p.wickets} max={maxWickets} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      )}
    </div>
  );
}

// ─── COMPARE PANEL ──────────────────────────────────────────────────────────────
function ComparePanel({ initialPlayer }) {
  const [p1, setP1] = useState(initialPlayer || "");
  const [p2, setP2] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [careerPlayer, setCareerPlayer] = useState("");
  const [careerResult, setCareerResult] = useState(null);
  const [careerLoading, setCareerLoading] = useState(false);

  // auto-fill initial player from leaderboard click
  useEffect(() => {
    if (initialPlayer) setP1(initialPlayer);
  }, [initialPlayer]);

  const handleCompare = async () => {
    if (!p1.trim() || !p2.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(`${BACKEND_URL}/analytics/compare-players`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ player1: p1.trim(), player2: p2.trim() }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message || "Comparison failed. Is the backend running?");
    }
    setLoading(false);
  };

  const handleCareer = async () => {
    if (!careerPlayer.trim()) return;
    setCareerLoading(true);
    setCareerResult(null);
    try {
      const res = await fetch(`${BACKEND_URL}/analytics/career-totals`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ player_name: careerPlayer.trim() }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setCareerResult(data);
    } catch (e) {
      setCareerResult(null);
    }
    setCareerLoading(false);
  };

  function statWinner(v1, v2, higherIsBetter = true) {
    if (v1 === v2) return [null, null];
    return higherIsBetter
      ? (v1 > v2 ? ["winner", "loser"] : ["loser", "winner"])
      : (v1 < v2 ? ["winner", "loser"] : ["loser", "winner"]);
  }

  const s1 = result?.player1_stats;
  const s2 = result?.player2_stats;

  const statSections = s1 && s2 ? [
    {
      label: "Batting",
      rows: [
        { label: "Total Runs", v1: s1.batting.runs, v2: s2.batting.runs, higher: true },
        { label: "Innings", v1: s1.batting.innings, v2: s2.batting.innings, higher: true },
        { label: "Average", v1: s1.batting.average, v2: s2.batting.average, higher: true },
        { label: "Strike Rate", v1: s1.batting.strike_rate, v2: s2.batting.strike_rate, higher: true },
        { label: "Fours", v1: s1.batting.fours, v2: s2.batting.fours, higher: true },
        { label: "Sixes", v1: s1.batting.sixes, v2: s2.batting.sixes, higher: true },
      ],
    },
    {
      label: "Bowling",
      rows: [
        { label: "Wickets", v1: s1.bowling.wickets, v2: s2.bowling.wickets, higher: true },
        { label: "Innings Bowled", v1: s1.bowling.innings, v2: s2.bowling.innings, higher: true },
        { label: "Economy", v1: s1.bowling.economy, v2: s2.bowling.economy, higher: false },
        { label: "Bowling Avg", v1: s1.bowling.average, v2: s2.bowling.average, higher: false },
        { label: "Runs Conceded", v1: s1.bowling.runs_conceded, v2: s2.bowling.runs_conceded, higher: false },
      ],
    },
    {
      label: "Matches",
      rows: [
        { label: "Matches Played", v1: result.player1_stats.matches_played, v2: result.player2_stats.matches_played, higher: true },
      ],
    },
  ] : [];

  return (
    <div className="compare-panel">
      <div className="panel-header">
        <div>
          <div className="panel-title">Player Comparison</div>
          <div className="panel-subtitle">Career stats head-to-head</div>
        </div>
      </div>

      {/* Compare form */}
      <div className="compare-form">
        <div className="compare-field">
          <div className="compare-field-label">Player 1</div>
          <input
            className="compare-input"
            placeholder="e.g. V Kohli"
            value={p1}
            onChange={e => setP1(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleCompare()}
          />
        </div>
        <div className="vs-divider">
          <div className="vs-text">vs</div>
        </div>
        <div className="compare-field">
          <div className="compare-field-label">Player 2</div>
          <input
            className="compare-input"
            placeholder="e.g. RG Sharma"
            value={p2}
            onChange={e => setP2(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleCompare()}
          />
        </div>
        <button className="compare-btn" onClick={handleCompare} disabled={loading || !p1.trim() || !p2.trim()}>
          {loading ? (
            <>
              <div className="loading-spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />
              Comparing
            </>
          ) : (
            <>
              Compare
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <line x1="1" y1="11" x2="11" y2="1" stroke="currentColor" strokeWidth="1.5"/>
                <polyline points="5,1 11,1 11,7" stroke="currentColor" strokeWidth="1.5" fill="none"/>
              </svg>
            </>
          )}
        </button>
      </div>

      {error && (
        <div style={{ background: "#fafafa", borderLeft: "4px solid #000", padding: "14px 20px", fontFamily: JETBRAINS, fontSize: 12, color: "#666", marginBottom: 24 }}>
          ⚠ {error}
        </div>
      )}

      {result && (
        <div className="comparison-result">
          {result.comparison && (
            <div className="comparison-summary">{result.comparison}</div>
          )}

          <div className="comparison-grid">
            {/* Column headers */}
            <div className="player-col-header">{result.player1}</div>
            <div style={{ background: "#000", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <span style={{ fontFamily: PLAYFAIR, fontWeight: 700, fontSize: 11, color: "#fff", fontStyle: "italic" }}>vs</span>
            </div>
            <div className="player-col-header right">{result.player2}</div>

            {/* Stat rows */}
            {statSections.map(section => (
              <>
                <div className="stat-section-header" key={`sh-${section.label}-1`}>{section.label}</div>
                <div className="stat-section-header" key={`sh-${section.label}-mid`} style={{ background: "#f5f5f5" }}></div>
                <div className="stat-section-header" key={`sh-${section.label}-2`}>{section.label}</div>

                {section.rows.map(row => {
                  const [c1, c2] = statWinner(row.v1, row.v2, row.higher);
                  return (
                    <>
                      <div className="compare-stat-cell" key={`${section.label}-${row.label}-v1`}>
                        <div className="compare-stat-label">{row.label}</div>
                        <div className={`compare-stat-value ${c1 || ""}`}>
                          {typeof row.v1 === "number" ? row.v1.toLocaleString() : row.v1 ?? "—"}
                        </div>
                      </div>
                      <div className="vs-col-spacer" key={`${section.label}-${row.label}-vs`} />
                      <div className="compare-stat-cell right" key={`${section.label}-${row.label}-v2`}>
                        <div className="compare-stat-label">{row.label}</div>
                        <div className={`compare-stat-value ${c2 || ""}`}>
                          {typeof row.v2 === "number" ? row.v2.toLocaleString() : row.v2 ?? "—"}
                        </div>
                      </div>
                    </>
                  );
                })}
              </>
            ))}
          </div>
        </div>
      )}

      {/* Career totals lookup */}
      <div className="career-lookup">
        <div className="career-lookup-title">Career Totals Lookup</div>
        <div className="compare-form" style={{ marginBottom: 0 }}>
          <div className="compare-field">
            <div className="compare-field-label">Player Name</div>
            <input
              className="compare-input"
              placeholder="e.g. Shubman Gill"
              value={careerPlayer}
              onChange={e => setCareerPlayer(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleCareer()}
            />
          </div>
          <button className="compare-btn" onClick={handleCareer} disabled={careerLoading || !careerPlayer.trim()} style={{ marginTop: 0 }}>
            {careerLoading ? "…" : "Look up"}
          </button>
        </div>

        {careerResult && (
          <div className="career-result">
            <div className="career-player-name">{careerResult.player_name}</div>
            <div className="career-matches-badge">{careerResult.matches_played} matches</div>

            <div className="career-section-label">Batting</div>
            <div className="career-stat-grid" style={{ marginBottom: 20 }}>
              {[
                { label: "Runs", value: (careerResult.batting.runs || 0).toLocaleString() },
                { label: "Average", value: careerResult.batting.average ?? "—" },
                { label: "Strike Rate", value: careerResult.batting.strike_rate ?? "—" },
                { label: "Innings", value: careerResult.batting.innings ?? "—" },
              ].map(s => (
                <div className="career-stat-box" key={s.label}>
                  <div className="career-stat-box-label">{s.label}</div>
                  <div className="career-stat-box-value">{s.value}</div>
                </div>
              ))}
            </div>

            <div className="career-section-label">Bowling</div>
            <div className="career-stat-grid">
              {[
                { label: "Wickets", value: careerResult.bowling.wickets ?? "—" },
                { label: "Economy", value: careerResult.bowling.economy ?? "—" },
                { label: "Bowling Avg", value: careerResult.bowling.average ?? "—" },
                { label: "Innings", value: careerResult.bowling.innings ?? "—" },
              ].map(s => (
                <div className="career-stat-box" key={s.label}>
                  <div className="career-stat-box-label">{s.label}</div>
                  <div className="career-stat-box-value">{s.value}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── PACE VS SPIN ANALYTICS PANEL ────────────────────────────────────────────────
function PaceVsSpinPanel() {
  const [phase, setPhase] = useState("all"); // "powerplay" | "middle" | "death" | "all"
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchPaceVsSpin = async (selectedPhase) => {
    setLoading(true);
    setError("");
    setData(null);
    try {
      const res = await fetch(`${BACKEND_URL}/analytics/pace-vs-spin`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phase: selectedPhase }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const responseData = await res.json();
      setData(responseData);
    } catch (e) {
      setError(e.message || "Failed to fetch analytics data. Is the backend running?");
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchPaceVsSpin(phase);
  }, [phase]);

  const paceStats = data?.pace_stats || {};
  const spinStats = data?.spin_stats || {};
  const topPaceBowlers = data?.top_pace_bowlers || [];
  const topSpinBowlers = data?.top_spin_bowlers || [];

  const formatOvers = (balls) => {
    if (!balls) return "0.0";
    const overs = Math.floor(balls / 6);
    const remaining = balls % 6;
    return `${overs}.${remaining}`;
  };

  const getComparisonBar = (paceValue, spinValue, higherIsBetter = true) => {
    const total = paceValue + spinValue;
    if (total === 0) return { pacePct: 50, spinPct: 50 };
    const pacePct = (paceValue / total) * 100;
    const spinPct = (spinValue / total) * 100;
    return { pacePct, spinPct };
  };

  const phaseLabel = phase === "all" ? "All Phases" :
                     phase === "powerplay" ? "Powerplay (Overs 1-10)" :
                     phase === "middle" ? "Middle (Overs 11-40)" :
                     "Death (Overs 41-50)";

  return (
    <div className="analytics-panel">
      <div className="panel-header">
        <div>
          <div className="panel-title">Pace vs Spin Analytics</div>
          <div className="panel-subtitle">Head-to-head bowling performance by match phase</div>
        </div>
      </div>

      {/* Phase Selector */}
      <div className="phase-selector">
        {["all", "powerplay", "middle", "death"].map((p) => (
          <button
            key={p}
            className={`phase-btn ${phase === p ? "active" : ""}`}
            onClick={() => setPhase(p)}
          >
            {p === "all" ? "All" : p.charAt(0).toUpperCase() + p.slice(1)}
          </button>
        ))}
      </div>

      {error && (
        <div style={{ background: "#fafafa", borderLeft: "4px solid #000", padding: "14px 20px", fontFamily: JETBRAINS, fontSize: 12, color: "#666", marginBottom: 24 }}>
          ⚠ {error}
        </div>
      )}

      {loading ? (
        <LoadingBlock />
      ) : data ? (
        <>
          {/* Stats Comparison Grid */}
          <div className="analytics-grid">
            {/* Pace Card */}
            <div className="stat-card">
              <div className="stat-card-header">
                <div className="stat-card-title">Pace Bowling</div>
                <div className="stat-card-subtitle">{phaseLabel}</div>
              </div>
              <div className="stat-card-body">
                <div className="stat-row">
                  <span className="stat-row-label">Wickets</span>
                  <span className="stat-row-value highlight">{paceStats.wickets || 0}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-row-label">Runs Conceded</span>
                  <span className="stat-row-value">{(paceStats.runs || 0).toLocaleString()}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-row-label">Overs</span>
                  <span className="stat-row-value">{formatOvers(paceStats.balls || 0)}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-row-label">Economy</span>
                  <span className="stat-row-value">{paceStats.economy?.toFixed(2) || "0.00"}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-row-label">Dot Ball %</span>
                  <span className="stat-row-value">{paceStats.dot_percentage?.toFixed(1) || "0.0"}%</span>
                </div>
                <div className="stat-row">
                  <span className="stat-row-label">Wickets/Innings</span>
                  <span className="stat-row-value">{paceStats.wickets_per_innings?.toFixed(2) || "0.00"}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-row-label">Innings</span>
                  <span className="stat-row-value dim">{paceStats.innings || 0}</span>
                </div>
              </div>
            </div>

            {/* Spin Card */}
            <div className="stat-card">
              <div className="stat-card-header">
                <div className="stat-card-title">Spin Bowling</div>
                <div className="stat-card-subtitle">{phaseLabel}</div>
              </div>
              <div className="stat-card-body">
                <div className="stat-row">
                  <span className="stat-row-label">Wickets</span>
                  <span className="stat-row-value highlight">{spinStats.wickets || 0}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-row-label">Runs Conceded</span>
                  <span className="stat-row-value">{(spinStats.runs || 0).toLocaleString()}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-row-label">Overs</span>
                  <span className="stat-row-value">{formatOvers(spinStats.balls || 0)}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-row-label">Economy</span>
                  <span className="stat-row-value">{spinStats.economy?.toFixed(2) || "0.00"}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-row-label">Dot Ball %</span>
                  <span className="stat-row-value">{spinStats.dot_percentage?.toFixed(1) || "0.0"}%</span>
                </div>
                <div className="stat-row">
                  <span className="stat-row-label">Wickets/Innings</span>
                  <span className="stat-row-value">{spinStats.wickets_per_innings?.toFixed(2) || "0.00"}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-row-label">Innings</span>
                  <span className="stat-row-value dim">{spinStats.innings || 0}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Comparison Bars */}
          <div style={{ marginBottom: 40 }}>
            <div style={{ fontFamily: JETBRAINS, fontSize: 10, letterSpacing: "0.15em", textTransform: "uppercase", color: "#888", marginBottom: 12 }}>
              Head-to-Head Comparison
            </div>
            {[
              { label: "Wickets", pace: paceStats.wickets || 0, spin: spinStats.wickets || 0, higher: true },
              { label: "Economy", pace: paceStats.economy || 0, spin: spinStats.economy || 0, higher: false },
              { label: "Dot Ball %", pace: paceStats.dot_percentage || 0, spin: spinStats.dot_percentage || 0, higher: true },
            ].map((stat) => {
              const { pacePct, spinPct } = getComparisonBar(stat.pace, stat.spin, stat.higher);
              return (
                <div key={stat.label} style={{ marginBottom: 16 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontFamily: "Source Serif 4", fontSize: 14 }}>
                    <span>{stat.label}</span>
                    <span style={{ fontFamily: JETBRAINS }}>
                      <span style={{ fontWeight: 500 }}>{stat.pace.toFixed(stat.label === "Economy" ? 2 : 1)}</span>
                      <span style={{ color: "#888", margin: "0 8px" }}>vs</span>
                      <span style={{ fontWeight: 500 }}>{stat.spin.toFixed(stat.label === "Economy" ? 2 : 1)}</span>
                    </span>
                  </div>
                  <div className="comparison-bar">
                    <div className="comparison-bar-fill pace" style={{ width: `${pacePct}%` }} />
                    <div className="comparison-bar-fill spin" style={{ width: `${spinPct}%` }} />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Top Bowlers Lists */}
          <div className="top-bowlers-section">
            <div className="top-bowlers-header">
              <div className="top-bowlers-title">Top Performers</div>
              <div className="top-bowlers-subtitle">{phaseLabel}</div>
            </div>

            <div className="top-bowlers-grid">
              {/* Top Pace Bowlers */}
              <div className="bowler-list">
                <div className="bowler-list-header">Pace Bowlers</div>
                {topPaceBowlers.length === 0 ? (
                  <div style={{ padding: "20px", textAlign: "center", color: "#999", fontFamily: JETBRAINS, fontSize: 11 }}>
                    No data available
                  </div>
                ) : (
                  topPaceBowlers.slice(0, 5).map((bowler, i) => (
                    <div key={bowler.name} className="bowler-item">
                      <span className={`bowler-rank ${i < 3 ? "top-3" : ""}`}>{i + 1}</span>
                      <div className="bowler-info">
                        <div className="bowler-name">{bowler.name}</div>
                        <div className="bowler-stats">
                          <div className="bowler-stat-item">
                            <span className="bowler-stat-label">Wkts</span>
                            <span className="bowler-stat-value highlight">{bowler.wickets}</span>
                          </div>
                          <div className="bowler-stat-item">
                            <span className="bowler-stat-label">Runs</span>
                            <span className="bowler-stat-value">{bowler.runs}</span>
                          </div>
                          <div className="bowler-stat-item">
                            <span className="bowler-stat-label">Econ</span>
                            <span className="bowler-stat-value">{bowler.economy?.toFixed(2)}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>

              {/* Top Spin Bowlers */}
              <div className="bowler-list">
                <div className="bowler-list-header">Spin Bowlers</div>
                {topSpinBowlers.length === 0 ? (
                  <div style={{ padding: "20px", textAlign: "center", color: "#999", fontFamily: JETBRAINS, fontSize: 11 }}>
                    No data available
                  </div>
                ) : (
                  topSpinBowlers.slice(0, 5).map((bowler, i) => (
                    <div key={bowler.name} className="bowler-item">
                      <span className={`bowler-rank ${i < 3 ? "top-3" : ""}`}>{i + 1}</span>
                      <div className="bowler-info">
                        <div className="bowler-name">{bowler.name}</div>
                        <div className="bowler-stats">
                          <div className="bowler-stat-item">
                            <span className="bowler-stat-label">Wkts</span>
                            <span className="bowler-stat-value highlight">{bowler.wickets}</span>
                          </div>
                          <div className="bowler-stat-item">
                            <span className="bowler-stat-label">Runs</span>
                            <span className="bowler-stat-value">{bowler.runs}</span>
                          </div>
                          <div className="bowler-stat-item">
                            <span className="bowler-stat-label">Econ</span>
                            <span className="bowler-stat-value">{bowler.economy?.toFixed(2)}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}

// ─── MAIN COMPONENT ────────────────────────────────────────────────────────────
export default function CricketRAG() {
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const [activeTab, setActiveTab] = useState("chat"); // "chat" | "leaderboard" | "compare" | "analytics"
  const [leaderboard, setLeaderboard] = useState({ batting: [], bowling: [] });
  const [leaderboardLoading, setLeaderboardLoading] = useState(false);
  const [compareInitPlayer, setCompareInitPlayer] = useState("");
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const activeChat = chats.find((c) => c.id === activeChatId) || null;

  useEffect(() => {
    fetch(`${BACKEND_URL}/health`)
      .then((r) => r.json())
      .then((d) => setHealth(d))
      .catch(() => setHealth(null));
  }, []);

  const fetchLeaderboard = async () => {
    setLeaderboardLoading(true);
    try {
      const [batRes, bowlRes] = await Promise.all([
        fetch(`${BACKEND_URL}/analytics/leaderboard/batting?limit=15`),
        fetch(`${BACKEND_URL}/analytics/leaderboard/bowling?limit=15`),
      ]);
      const batData = await batRes.json();
      const bowlData = await bowlRes.json();
      setLeaderboard({ batting: batData.leaderboard || [], bowling: bowlData.leaderboard || [] });
    } catch (err) {
      console.error("Failed to fetch leaderboard:", err);
    }
    setLeaderboardLoading(false);
  };

  // Auto-load leaderboard on first visit
  useEffect(() => {
    if (activeTab === "leaderboard" && leaderboard.batting.length === 0) {
      fetchLeaderboard();
    }
  }, [activeTab]);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [activeChat?.messages, loading]);

  const createChat = useCallback(() => {
    const id = generateId();
    const chat = { id, title: "New Conversation", messages: [], createdAt: Date.now() };
    setChats((prev) => [chat, ...prev]);
    setActiveChatId(id);
    setActiveTab("chat");
    setTimeout(() => inputRef.current?.focus(), 50);
  }, []);

  const deleteChat = useCallback(
    (id, e) => {
      e.stopPropagation();
      setChats((prev) => prev.filter((c) => c.id !== id));
      if (activeChatId === id) setActiveChatId(null);
    },
    [activeChatId]
  );

  const sendQuery = useCallback(
    async (questionText) => {
      const q = questionText || query;
      if (!q.trim() || loading) return;

      let chatId = activeChatId;
      if (!chatId) {
        const id = generateId();
        const chat = { id, title: q.slice(0, 40), messages: [], createdAt: Date.now() };
        setChats((prev) => [chat, ...prev]);
        setActiveChatId(id);
        chatId = id;
      }

      setActiveTab("chat");
      const userMsg = { role: "user", content: q.trim(), ts: Date.now() };
      setChats((prev) =>
        prev.map((c) => {
          if (c.id !== chatId) return c;
          const msgs = [...c.messages, userMsg];
          const title = msgs.length === 1 ? q.slice(0, 40) : c.title;
          return { ...c, messages: msgs, title };
        })
      );
      setQuery("");
      setLoading(true);

      try {
        const res = await fetch(`${BACKEND_URL}/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q.trim() }),
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        const assistantMsg = {
          role: "assistant",
          content: data.answer,
          query_type: data.query_type,
          records_scanned: data.records_scanned,
          ts: Date.now(),
        };
        setChats((prev) =>
          prev.map((c) => (c.id === chatId ? { ...c, messages: [...c.messages, assistantMsg] } : c))
        );
      } catch (err) {
        const errMsg = {
          role: "error",
          content: err.message || "Connection failed. Is the backend running?",
          ts: Date.now(),
        };
        setChats((prev) =>
          prev.map((c) => (c.id === chatId ? { ...c, messages: [...c.messages, errMsg] } : c))
        );
      } finally {
        setLoading(false);
      }
    },
    [query, loading, activeChatId]
  );

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  };

  // When user clicks a player name in leaderboard → go to compare tab pre-filled
  const handleSelectPlayer = (name, tab) => {
    setCompareInitPlayer(name);
    setActiveTab(tab);
  };

  return (
    <>
      <style>{styles}</style>
      <div className="app">
        {/* ── SIDEBAR ─────────────────────────────────────── */}
        <aside className="sidebar">
          <div className="sidebar-header">
            <div className="sidebar-logo">
              <img
                src="/CSAlogo.png.jpeg"
                alt="logo"
                style={{ width: 36, height: 36, objectFit: "cover", borderRadius: "50%", border: "1px solid #444" }}
              />
              GRANDSTAND <span>AI</span>
            </div>
            <button className="new-chat-btn" onClick={createChat}>
              <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                <line x1="6.5" y1="1" x2="6.5" y2="12" stroke="currentColor" strokeWidth="1.5" />
                <line x1="1" y1="6.5" x2="12" y2="6.5" stroke="currentColor" strokeWidth="1.5" />
              </svg>
              New Conversation
            </button>
          </div>

          {/* Navigation */}
          <div className="sidebar-nav">
            <div className="sidebar-nav-label">Tools</div>
            <div
              className={`sidebar-nav-item ${activeTab === "leaderboard" ? "active" : ""}`}
              onClick={() => setActiveTab("leaderboard")}
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <rect x="1" y="7" width="3" height="6" stroke="currentColor" strokeWidth="1.3"/>
                <rect x="5.5" y="4" width="3" height="9" stroke="currentColor" strokeWidth="1.3"/>
                <rect x="10" y="1" width="3" height="12" stroke="currentColor" strokeWidth="1.3"/>
              </svg>
              Leaderboard
            </div>
            <div
              className={`sidebar-nav-item ${activeTab === "analytics" ? "active" : ""}`}
              onClick={() => setActiveTab("analytics")}
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <line x1="1" y1="13" x2="13" y2="1" stroke="currentColor" strokeWidth="1.3"/>
                <line x1="1" y1="9" x2="9" y2="1" stroke="currentColor" strokeWidth="1.3"/>
                <line x1="1" y1="5" x2="5" y2="1" stroke="currentColor" strokeWidth="1.3"/>
              </svg>
              Analytics
            </div>
            <div
              className={`sidebar-nav-item ${activeTab === "compare" ? "active" : ""}`}
              onClick={() => setActiveTab("compare")}
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <circle cx="4" cy="7" r="3" stroke="currentColor" strokeWidth="1.3"/>
                <circle cx="10" cy="7" r="3" stroke="currentColor" strokeWidth="1.3"/>
                <line x1="7" y1="4" x2="7" y2="10" stroke="currentColor" strokeWidth="1.3"/>
              </svg>
              Compare Players
            </div>
          </div>

          {/* Chat history */}
          <div className="chat-list">
            {chats.length > 0 && <div className="chat-list-label">Chat History</div>}
            {chats.map((chat) => (
              <div
                key={chat.id}
                className={`chat-item ${chat.id === activeChatId && activeTab === "chat" ? "active" : ""}`}
                onClick={() => { setActiveChatId(chat.id); setActiveTab("chat"); }}
              >
                <div className="chat-item-content">
                  <div className="chat-item-title">{chat.title}</div>
                  <div className="chat-item-meta">
                    {chat.messages.length} msg{chat.messages.length !== 1 ? "s" : ""} · {formatTime(chat.createdAt)}
                  </div>
                </div>
                <button className="chat-item-delete" onClick={(e) => deleteChat(chat.id, e)} title="Delete">
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                    <line x1="1" y1="1" x2="11" y2="11" stroke="currentColor" strokeWidth="1.5" />
                    <line x1="11" y1="1" x2="1" y2="11" stroke="currentColor" strokeWidth="1.5" />
                  </svg>
                </button>
              </div>
            ))}
          </div>

          {/* Status */}
          <div style={{ padding: "12px 20px", borderTop: "1px solid #222" }}>
            <div className={`status-pill ${health ? "healthy" : ""}`} style={{ display: "block", textAlign: "center" }}>
              {health ? `● ${health.docs ?? "?"} docs indexed` : "○ Backend offline"}
            </div>
          </div>
        </aside>

        {/* ── MAIN ────────────────────────────────────────── */}
        <main className="main">
          {/* Header */}
          <header className="main-header">
            {activeTab === "chat" ? (
              activeChat
                ? <div className="main-title">{activeChat.title}</div>
                : <div className="main-title-empty">Select or start a conversation</div>
            ) : activeTab === "leaderboard" ? (
              <div className="main-title">Leaderboard</div>
            ) : activeTab === "analytics" ? (
              <div className="main-title">Pace vs Spin Analytics</div>
            ) : (
              <div className="main-title">Player Comparison</div>
            )}
          </header>

          {/* ── LEADERBOARD TAB ── */}
          {/* Issue #16 — PanelErrorBoundary prevents one panel crash taking down the whole app */}
          {activeTab === "leaderboard" && (
            <PanelErrorBoundary>
              <LeaderboardPanel
                leaderboard={leaderboard}
                loading={leaderboardLoading}
                onRefresh={fetchLeaderboard}
                onSelectPlayer={handleSelectPlayer}
              />
            </PanelErrorBoundary>
          )}

          {/* ── COMPARE TAB ── */}
          {activeTab === "compare" && (
            <PanelErrorBoundary>
              <ComparePanel initialPlayer={compareInitPlayer} />
            </PanelErrorBoundary>
          )}

          {/* ── ANALYTICS TAB ── */}
          {activeTab === "analytics" && (
            <PanelErrorBoundary>
              <PaceVsSpinPanel />
            </PanelErrorBoundary>
          )}

          {/* ── CHAT TAB ── */}
          {activeTab === "chat" && (
            <>
              <div className="messages-area">
                {!activeChat || activeChat.messages.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-headline">Ask GrandStand</div>
                    <div className="empty-sub">Your AI-powered cricket assistant</div>
                    <div className="empty-rule" />
                    <div className="empty-hints">
                      {HINTS.map((h) => (
                        <button
                          key={h}
                          className="hint-btn"
                          onClick={() => {
                            if (!activeChatId) createChat();
                            setQuery(h);
                            setTimeout(() => sendQuery(h), 50);
                          }}
                        >
                          {h}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <>
                    {activeChat.messages.map((msg, i) => (
                      <div key={i} className="message">
                        {msg.role === "user" && (
                          <div className="msg-user">
                            <div className="msg-user-bubble">{msg.content}</div>
                          </div>
                        )}
                        {msg.role === "assistant" && (
                          <div className="msg-assistant">
                            <div className="msg-label"><span>Cricket RAG</span></div>
                            {/* Issue #15 — safe component replaces dangerouslySetInnerHTML */}
                            <div className="msg-answer"><FormattedMessage content={msg.content} /></div>
                            <div className="msg-meta">
                              <span className={`meta-tag ${msg.query_type === "aggregate" ? "aggregate" : ""}`}>
                                {msg.query_type}
                              </span>
                              <span className="meta-tag">{msg.records_scanned} records scanned</span>
                              <span className="meta-tag">{formatTime(msg.ts)}</span>
                            </div>
                          </div>
                        )}
                        {msg.role === "error" && (
                          <div className="msg-assistant">
                            <div className="msg-label"><span>Error</span></div>
                            <div className="msg-error">⚠ {msg.content}</div>
                          </div>
                        )}
                      </div>
                    ))}
                    {loading && (
                      <div className="message">
                        <div className="msg-assistant">
                          <div className="msg-label"><span>Cricket RAG</span></div>
                          <div className="msg-loading">
                            <div className="loading-dots">
                              <div className="loading-dot" />
                              <div className="loading-dot" />
                              <div className="loading-dot" />
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                    <div ref={messagesEndRef} />
                  </>
                )}
              </div>

              <div className="input-area">
                <div className="input-row">
                  <textarea
                    ref={inputRef}
                    className="query-input"
                    placeholder="Ask about matches, players, stats…"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={handleKey}
                    disabled={loading}
                    rows={1}
                  />
                  <button className="send-btn" onClick={() => sendQuery()} disabled={loading || !query.trim()}>
                    {loading ? (
                      "…"
                    ) : (
                      <>
                        Send
                        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                          <line x1="1" y1="11" x2="11" y2="1" stroke="currentColor" strokeWidth="1.5" />
                          <polyline points="5,1 11,1 11,7" stroke="currentColor" strokeWidth="1.5" fill="none" />
                        </svg>
                      </>
                    )}
                  </button>
                </div>
                <div className="input-hint">↵ Enter to send · Shift+↵ newline · Stats powered by Llama 3 via Groq</div>
              </div>
            </>
          )}
        </main>
      </div>
    </>
  );
}