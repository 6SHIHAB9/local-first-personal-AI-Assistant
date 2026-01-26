import { useState } from "react";
import { Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import ChatMessage from "./ChatMessage";
import { sendMessage, syncVault } from "@/lib/backend";

type Message = {
  role: "user" | "assistant";
  content: string;
};

// Add this prop type
type ChatAreaProps = {
  setVaultStatus?: (status: any) => void;
};

const ChatArea = ({ setVaultStatus }: ChatAreaProps) => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userText = input;
    setInput("");
    setLoading(true);

    setMessages((prev) => [
      ...prev,
      { role: "user", content: userText },
    ]);

    try {
      const res = await sendMessage(userText);
      const answer = res?.answer ?? "No response from assistant.";

      // ✅ NEW: Check for auto-sync performed
      if (res?.sync_performed && setVaultStatus) {
        console.log("✅ Vault auto-synced:", res.sync_performed);
        setVaultStatus(res.sync_performed);
        
        // Optional: Show a toast/notification to user
        // You can add a toast library or just log it
      }

      // ✅ KEEP: Update vault status from manual response (backward compatible)
      if (res?.vault_status && setVaultStatus) {
        setVaultStatus(res.vault_status);
      }

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: answer },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Something went wrong talking to the backend.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSync = async () => {
    setLoading(true);
    try {
      const data = await syncVault();
      
      // ✅ Update vault status from sync
      if (setVaultStatus) {
        setVaultStatus(data);
      }
      
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Vault synced successfully.",
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Vault sync failed.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Enhanced Header */}
      <header className="relative flex-shrink-0 border-b border-white/5 bg-gradient-to-b from-slate-900/50 to-slate-950/50 backdrop-blur-xl px-6 py-3 overflow-hidden">
        {/* Animated gradient mesh background */}
        <div className="absolute inset-0 opacity-30">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-emerald-500/10 rounded-full mix-blend-screen filter blur-3xl animate-pulse" style={{ animationDuration: '4s' }} />
          <div className="absolute top-0 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full mix-blend-screen filter blur-3xl animate-pulse" style={{ animationDuration: '6s', animationDelay: '1s' }} />
        </div>

        {/* Subtle scan line effect */}
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-emerald-500/5 to-transparent opacity-20 animate-scan" />

        <div className="relative">
          {/* Main header row */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-4">
              {/* Premium vault icon */}
              <div className="relative group">
                <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-xl blur-lg group-hover:blur-xl transition-all duration-300" />
                <div className="relative w-11 h-11 rounded-xl bg-gradient-to-br from-emerald-500/10 to-cyan-500/10 backdrop-blur-sm flex items-center justify-center border border-emerald-500/20 shadow-lg shadow-emerald-500/10">
                  <svg
                    className="w-5 h-5 text-emerald-400"
                    fill="none"
                    strokeWidth="2"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z"
                    />
                  </svg>
                </div>
                {/* Animated pulse ring */}
                <div className="absolute -bottom-1 -right-1">
                  <div className="relative flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500 ring-2 ring-slate-900" />
                  </div>
                </div>
              </div>

              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-white via-white to-white/60 bg-clip-text text-transparent tracking-tight">
                  Vault Assistant
                </h1>
                <p className="text-xs text-slate-400 font-medium flex items-center gap-1.5 mt-0.5">
                  <span className="inline-block w-1 h-1 rounded-full bg-emerald-500 animate-pulse" />
                  Private, local AI for your notes
                </p>
              </div>
            </div>

            {/* Status badge */}
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 backdrop-blur-sm">
              <div className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
              </div>
              <span className="text-xs font-semibold text-emerald-400">Ready</span>
            </div>
          </div>

          {/* Feature highlight bar */}
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-emerald-500/5 to-cyan-500/5 rounded-xl blur-xl" />
            <div className="relative flex items-center gap-4 px-4 py-3 rounded-xl bg-slate-800/40 border border-white/5 backdrop-blur-sm">
              {/* AI Icon */}
              <div className="flex-shrink-0">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center">
                  <svg
                    className="w-4 h-4 text-blue-400"
                    fill="none"
                    strokeWidth="2"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z"
                    />
                  </svg>
                </div>
              </div>

              {/* Main text */}
              <div className="flex-1">
                <p className="text-sm text-slate-200 font-medium">
                  Ask questions and get answers directly from your vault files.
                </p>
              </div>

              {/* Privacy badges */}
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-emerald-500/10 border border-emerald-500/20">
                  <svg className="w-3 h-3 text-emerald-400" fill="none" strokeWidth="2.5" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5.636 5.636a9 9 0 1012.728 0M12 3v9" />
                  </svg>
                  <span className="text-xs font-semibold text-emerald-400">Fully offline</span>
                </div>
                <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-blue-500/10 border border-blue-500/20">
                  <svg className="w-3 h-3 text-blue-400" fill="none" strokeWidth="2.5" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
                  </svg>
                  <span className="text-xs font-semibold text-blue-400">Nothing uploaded</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <style>{`
        @keyframes scan {
          0%, 100% {
            transform: translateY(-100%);
          }
          50% {
            transform: translateY(100%);
          }
        }
        .animate-scan {
          animation: scan 8s ease-in-out infinite;
        }
      `}</style>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 ? (
          /* Empty state */
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md space-y-4">
              <div className="relative inline-block">
                <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/20 to-blue-500/20 rounded-2xl blur-2xl" />
                <div className="relative w-16 h-16 mx-auto rounded-2xl bg-gradient-to-br from-emerald-500/10 to-blue-500/10 backdrop-blur-sm flex items-center justify-center border border-emerald-500/20">
                  <svg className="w-8 h-8 text-emerald-400" fill="none" strokeWidth="1.5" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" />
                  </svg>
                </div>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-slate-200 mb-2">Start a conversation</h3>
                <p className="text-sm text-slate-400 leading-relaxed">Ask me anything about your vault files. I'll search through your notes to find the best answer.</p>
              </div>
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`group relative ${
                  msg.role === "user" ? "flex justify-end" : "flex justify-start"
                }`}
              >
                {/* Message container */}
                <div
                  className={`relative max-w-[85%] ${
                    msg.role === "user"
                      ? "ml-auto"
                      : "mr-auto"
                  }`}
                >
                  <div className="relative flex gap-3">
                    {/* Avatar */}
                    {msg.role === "assistant" && (
                      <div className="flex-shrink-0 mt-1">
                        <div className="relative">
                          <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/30 to-blue-500/30 rounded-lg blur-md" />
                          <div className="relative w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500/20 to-blue-500/20 backdrop-blur-sm flex items-center justify-center border border-emerald-500/30">
                            <svg className="w-4 h-4 text-emerald-400" fill="none" strokeWidth="2" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                            </svg>
                          </div>
                        </div>
                      </div>
                    )}
                    
                    {/* Message content */}
                    <div
                      className={`relative flex-1 rounded-2xl px-4 py-3 ${
                        msg.role === "user"
                          ? "bg-gradient-to-br from-indigo-500/90 to-purple-500/90 text-white shadow-lg shadow-indigo-500/25"
                          : "bg-slate-800/60 backdrop-blur-sm border border-slate-700/50 text-slate-200"
                      }`}
                    >
                      <p className="text-sm leading-relaxed whitespace-pre-wrap break-words break-all">{msg.content}</p>
                    </div>
                    
                    {/* User avatar */}
                    {msg.role === "user" && (
                      <div className="flex-shrink-0 mt-1">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center shadow-lg shadow-indigo-500/25">
                          <svg className="w-4 h-4 text-white" fill="none" strokeWidth="2" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                          </svg>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}

            {/* Typing indicator */}
            {loading && (
              <div className="group relative flex justify-start">
                <div className="relative max-w-[85%] mr-auto">
                  <div className="absolute -inset-1 bg-gradient-to-r from-emerald-500/10 via-blue-500/10 to-purple-500/10 rounded-2xl blur-xl opacity-50" />
                  
                  <div className="relative flex gap-3">
                    {/* Avatar */}
                    <div className="flex-shrink-0 mt-1">
                      <div className="relative">
                        <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/30 to-blue-500/30 rounded-lg blur-md animate-pulse" />
                        <div className="relative w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500/20 to-blue-500/20 backdrop-blur-sm flex items-center justify-center border border-emerald-500/30">
                          <svg className="w-4 h-4 text-emerald-400 animate-pulse" fill="none" strokeWidth="2" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                          </svg>
                        </div>
                      </div>
                    </div>
                    
                    {/* Typing animation */}
                    <div className="relative flex-1 rounded-2xl px-4 py-3 bg-slate-800/60 backdrop-blur-sm border border-slate-700/50">
                      <div className="flex items-center gap-1.5">
                        <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                        <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                        <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Input */}
      <div className="relative border-t border-white/5 bg-gradient-to-b from-slate-900/50 to-slate-950/80 backdrop-blur-xl">
        {/* Subtle glow effect */}
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-emerald-500/20 to-transparent" />
        
        <div className="p-4">
          <div className="flex gap-3 items-end">
            <div className="flex-1 relative group">
              {/* Glow effect on focus */}
              <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/0 via-emerald-500/5 to-blue-500/0 rounded-xl blur-xl opacity-0 group-focus-within:opacity-100 transition-opacity duration-300" />
              
              <Textarea
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  // Auto-resize logic
                  e.target.style.height = 'auto';
                  e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
                }}
                placeholder="Ask something from your vault..."
                className="relative min-h-[52px] max-h-[200px] resize-none bg-slate-800/50 border-slate-700/50 hover:border-slate-600/50 focus:border-emerald-500/50 rounded-xl px-4 py-3.5 text-slate-200 placeholder:text-slate-500 focus:ring-2 focus:ring-emerald-500/20 transition-all overflow-y-auto"
                rows={1}
                disabled={false}
                onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();

                  // Reset height after sending
                  setTimeout(() => {
                    const el = e.currentTarget;
                    el.style.height = "auto";
                  }, 0);
                }
                }}
              />
            </div>
            
            <Button
              onClick={() => {
                handleSend();
                // Reset textarea height
                const textarea = document.querySelector('textarea');
                if (textarea) {
                  textarea.style.height = 'auto';
                }
              }}
              disabled={loading || !input.trim()}
              className="h-[52px] w-[52px] flex-shrink-0 rounded-xl bg-gradient-to-br from-emerald-500 to-emerald-600 hover:from-emerald-400 hover:to-emerald-500 disabled:from-slate-700 disabled:to-slate-800 border-0 shadow-lg shadow-emerald-500/20 hover:shadow-emerald-500/40 disabled:shadow-none transition-all duration-200 group"
            >
              {loading ? (
                <svg className="h-5 w-5 animate-spin text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
              ) : (
                <Send className="h-5 w-5 text-white group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
              )}
            </Button>
          </div>
          
          {/* Helper text */}
          <div className="flex items-center justify-between mt-2 px-1">
            <p className="text-xs text-slate-500">
              Press <kbd className="px-1.5 py-0.5 rounded bg-slate-800/50 border border-slate-700/50 text-slate-400 font-mono text-[10px]">Enter</kbd> to send, <kbd className="px-1.5 py-0.5 rounded bg-slate-800/50 border border-slate-700/50 text-slate-400 font-mono text-[10px]">Shift + Enter</kbd> for new line
            </p>
            {input.length > 0 && (
              <p className="text-xs text-slate-500">
                {input.length} characters
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatArea;