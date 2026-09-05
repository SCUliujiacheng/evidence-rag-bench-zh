const form = document.querySelector("#ask-form");
const result = document.querySelector("#result");
const status = document.querySelector("#status");
const answer = document.querySelector("#answer");
const reason = document.querySelector("#reason");
const latency = document.querySelector("#latency");
const evidence = document.querySelector("#evidence");
const reasonLabels = {
  insufficient_evidence: "检索到的证据不足",
  citation_validation_failed: "引用校验失败",
};

function element(tag, text) {
  const node = document.createElement(tag);
  node.textContent = text;
  return node;
}

function showError(message) {
  result.hidden = false;
  evidence.replaceChildren();
  status.textContent = "请求失败";
  answer.textContent = message;
  reason.textContent = "";
  latency.textContent = "";
}

function apiErrorMessage(body) {
  return typeof body.detail === "string"
    ? body.detail
    : "请求参数不符合要求，请检查后重试。";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = document.querySelector("#question").value.trim();
  if (!question) {
    showError("问题不能只包含空白字符。");
    return;
  }
  const response = await fetch("/v1/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: 3 }),
  });
  const body = await response.json();
  result.hidden = false;
  evidence.replaceChildren();
  if (!response.ok) {
    showError(apiErrorMessage(body));
    return;
  }
  status.textContent = body.status === "answer" ? "已基于证据回答" : "已拒答";
  answer.textContent = body.answer;
  reason.textContent = body.reason
    ? `原因：${reasonLabels[body.reason] || body.reason}`
    : "";
  latency.textContent = `延迟：${body.latency_ms.toFixed(1)} ms`;
  body.evidence.forEach((item) => {
    const card = document.createElement("article");
    card.className = "evidence-card";
    card.append(element("strong", item.chunk_id));
    card.append(element("p", item.text));
    const link = document.createElement("a");
    link.href = item.source_url;
    link.textContent = "打开来源";
    link.target = "_blank";
    link.rel = "noreferrer";
    card.append(link);
    evidence.append(card);
  });
});
