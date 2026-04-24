import { useState, useRef, useEffect, useCallback } from "react";

const BACKEND_URL = "http://localhost:8000";

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

  .msg-meta {
    display: flex;
    gap: 16px;
    margin-top: 10px;
    padding-left: 4px;
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

export default function CricketRAG() {
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const activeChat = chats.find((c) => c.id === activeChatId) || null;

  useEffect(() => {
    fetch(`${BACKEND_URL}/health`)
      .then((r) => r.json())
      .then((d) => setHealth(d))
      .catch(() => setHealth(null));
  }, []);

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
    setTimeout(() => inputRef.current?.focus(), 50);
  }, []);

  const deleteChat = useCallback(
    (id, e) => {
      e.stopPropagation();
      setChats((prev) => prev.filter((c) => c.id !== id));
      if (activeChatId === id) {
        setActiveChatId(null);
      }
    },
    [activeChatId]
  );

  const updateChatMessages = (chatId, updater) => {
    setChats((prev) =>
      prev.map((c) => (c.id === chatId ? { ...c, ...updater(c) } : c))
    );
  };

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
          prev.map((c) =>
            c.id === chatId ? { ...c, messages: [...c.messages, assistantMsg] } : c
          )
        );
      } catch (err) {
        const errMsg = {
          role: "error",
          content: err.message || "Connection failed. Is the backend running?",
          ts: Date.now(),
        };
        setChats((prev) =>
          prev.map((c) =>
            c.id === chatId ? { ...c, messages: [...c.messages, errMsg] } : c
          )
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

  return (
    <>
      <style>{styles}</style>
      <div className="app">
        {/* SIDEBAR */}
        <aside className="sidebar">
          <div className="sidebar-header">
            <div className="sidebar-logo">
                <img src="/CSAlogo.png.jpeg" alt="logo" style={{width:'36px',height:'36px',objectFit:'cover',borderRadius:'50%',border:'1px solid #444'}} />
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

          <div className="chat-list">
            {chats.length > 0 && (
              <div className="chat-list-label">History</div>
            )}
            {chats.map((chat) => (
              <div
                key={chat.id}
                className={`chat-item ${chat.id === activeChatId ? "active" : ""}`}
                onClick={() => setActiveChatId(chat.id)}
              >
                <div className="chat-item-content">
                  <div className="chat-item-title">{chat.title}</div>
                  <div className="chat-item-meta">
                    {chat.messages.length} msg{chat.messages.length !== 1 ? "s" : ""} · {formatTime(chat.createdAt)}
                  </div>
                </div>
                <button
                  className="chat-item-delete"
                  onClick={(e) => deleteChat(chat.id, e)}
                  title="Delete"
                >
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                    <line x1="1" y1="1" x2="11" y2="11" stroke="currentColor" strokeWidth="1.5" />
                    <line x1="11" y1="1" x2="1" y2="11" stroke="currentColor" strokeWidth="1.5" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        </aside>

        {/* MAIN */}
        <main className="main">
          <header className="main-header">
            {activeChat ? (
              <div className="main-title">{activeChat.title}</div>
            ) : (
              <div className="main-title-empty">Select or start a conversation</div>
            )}
          </header>

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
                        <div className="msg-label">
                          <span>Cricket RAG</span>
                        </div>
                        <div className="msg-answer">{msg.content}</div>
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

          {/* INPUT */}
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
              <button
                className="send-btn"
                onClick={() => sendQuery()}
                disabled={loading || !query.trim()}
              >
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
          </div>
        </main>
      </div>
    </>
  );
}