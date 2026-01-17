const BACKEND_URL = "http://127.0.0.1:8000";

async function post(endpoint: string, body: any) {
  const res = await fetch(`${BACKEND_URL}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

export function syncVault() {
  return post("/sync", {});
}

export function sendMessage(message: string) {
  return post("/ask", { question: message });
}
