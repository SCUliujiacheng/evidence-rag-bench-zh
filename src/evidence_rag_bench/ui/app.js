const form = document.querySelector("#ask-form");
const profile = document.querySelector("#profile");
const questionInput = document.querySelector("#question");
const result = document.querySelector("#result");
const responseView = document.querySelector("#response");
const emptyState = document.querySelector("#empty-state");
const status = document.querySelector("#status");
const reason = document.querySelector("#reason");
const latency = document.querySelector("#latency");
const evidence = document.querySelector("#evidence");
const resultMeta = document.querySelector("#result-meta");
const submittedQuestion = document.querySelector("#submitted-question");
const examples = document.querySelector("#examples");
const corpusList = document.querySelector("#corpus-list");
const corpusNote = document.querySelector("#corpus-note");
const submitButton = form.querySelector('button[type="submit"]');
const isChinese = document.documentElement.lang === "zh-CN";

const copy = isChinese ? {
  search: "检索",
  searching: "正在检索…",
  found: "找到相关段落，请核对原文。",
  abstained: "这次没有足够证据。",
  failed: "这次检索没有完成。",
  blank: "问题不能只包含空白字符。",
  apiError: "问题格式不对，检查一下再试。",
  networkError: "无法连接本地服务，请确认服务还在运行后重试。",
  invalidResponse: "服务返回的内容不完整，请重新检索。",
  source: "原文",
  original: "原始段落",
  cited: "回答引用",
  score: "排序分数",
  query: "问题：",
  latency: "检索耗时",
  reasonLabels: {
    insufficient_evidence: "返回结果的相关性没有达到当前语料的拒答阈值。",
    citation_validation_failed: "引用校验没有通过。",
  },
  profiles: {
    "zh-v1": {
      label: "中文 · zh-v1",
      documents: [
        ["Milvus", "中文 README · 向量检索、架构与数据处理。"],
        ["PaddlePaddle", "中文 README · 深度学习框架、训练与硬件适配。"],
        ["FlagEmbedding", "中文 README · 文本嵌入、重排与 BGE 模型。"],
      ],
      note: "检索使用仓库内保存的文档版本，不会实时抓取网站。",
    },
    "en-v1": {
      label: "英文 · en-v1",
      documents: [
        ["FAISS", "5 份文档 · 相似度检索、安装、基准与示例。"],
        ["scikit-learn", "4 份文档 · 项目介绍、入门、贡献指南与 FAQ。"],
        ["LangChain", "6 份文档 · 核心组件、文本分块、集成与测试。"],
      ],
      note: "这组语料是英文文档，建议使用英文问题检索。",
    },
  },
} : {
  search: "Search",
  searching: "Searching…",
  found: "Related passages found. Check them against the question.",
  abstained: "Not enough evidence for this question.",
  failed: "The search could not finish.",
  blank: "Enter a question, not just spaces.",
  apiError: "Check the question and try again.",
  networkError: "Could not reach the local service. Check that it is running and try again.",
  invalidResponse: "The service returned an incomplete response. Please try again.",
  source: "Source",
  original: "Original passage",
  cited: "Cited in response",
  score: "Ranking score",
  query: "Question: ",
  latency: "Search time",
  reasonLabels: {
    insufficient_evidence: "The results did not reach this corpus's relevance threshold.",
    citation_validation_failed: "The citation check failed.",
  },
  profiles: {
    "en-v1": {
      label: "English · en-v1",
      documents: [
        ["FAISS", "5 documents · Similarity search, installation, benchmarks and demos."],
        ["scikit-learn", "4 documents · Overview, getting started, contributing and FAQ."],
        ["LangChain", "6 documents · Core components, text splitting, integrations and tests."],
      ],
      note: "Searches use the document snapshots in this repository, not live websites.",
    },
    "zh-v1": {
      label: "中文 · zh-v1",
      documents: [
        ["Milvus", "Chinese README · Vector search, architecture and data processing."],
        ["PaddlePaddle", "Chinese README · Training, inference and hardware support."],
        ["FlagEmbedding", "Chinese README · Embeddings, reranking and BGE models."],
      ],
      note: "These documents are in Chinese. Try asking a question in Chinese.",
    },
  },
};

const sampleQuestions = {
  "zh-v1": [
    "Milvus 为什么同时支持流处理和批处理？",
    "BGE-M3 支持哪些检索方式？",
    "飞桨怎样屏蔽不同芯片软件栈的接口差异？",
  ],
  "en-v1": [
    "How can FAISS implement cosine similarity?",
    "What do LangChain text splitters do?",
    "Which learning paradigms does scikit-learn support?",
  ],
};

const documentTitles = {
  "milvus-readme-zh": "Milvus",
  "paddle-readme-zh": "PaddlePaddle",
  "flagembedding-readme-zh": "FlagEmbedding",
  "faiss-readme": "FAISS",
  "faiss-benchmarks": "FAISS · Benchmarks",
  "faiss-install": "FAISS · Installation",
  "faiss-c-api-install": "FAISS · C API",
  "faiss-demos": "FAISS · Demos",
  "sklearn-readme": "scikit-learn",
  "sklearn-contributing": "scikit-learn · Contributing",
  "sklearn-getting-started": "scikit-learn · Getting started",
  "sklearn-faq": "scikit-learn · FAQ",
  "langchain-readme": "LangChain",
  "langchain-package": "LangChain · Package",
  "langchain-core": "LangChain · Core",
  "langchain-text-splitters": "LangChain · Text splitters",
  "langchain-standard-tests": "LangChain · Standard tests",
  "langchain-openai": "LangChain · OpenAI integration",
};

let requestNumber = 0;
let pendingRequest = null;

function element(tag, text, className) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

// This is only a reading aid. The unmodified chunk is available below it.
function readableSource(text) {
  return text
    .replace(/\r\n?/g, "\n")
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/<[^>]+>/g, " ")
    .replace(/\b(?:details|summary|b|strong|em)\s*>/gi, "")
    .replace(/^\s{0,3}#{1,6}\s+/gm, "")
    .replace(/^\s*[-*]\s+/gm, "• ")
    .replace(/^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/gm, "")
    .replace(/\s+\|\s+/g, " · ")
    .replace(/\x60\x60\x60[a-z0-9_-]*/gi, "")
    .replace(/[*\x60~]/g, "")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/^[ \t]+$/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function setBusy(busy) {
  result.setAttribute("aria-busy", String(busy));
  submitButton.disabled = busy;
  submitButton.textContent = busy ? copy.searching : copy.search;
}

function cancelRequest() {
  requestNumber += 1;
  pendingRequest?.abort();
  pendingRequest = null;
  setBusy(false);
}

function clearResult() {
  cancelRequest();
  evidence.replaceChildren();
  responseView.hidden = true;
  emptyState.hidden = false;
  status.classList.remove("error");
  status.textContent = "";
  reason.hidden = true;
  latency.hidden = true;
  submittedQuestion.hidden = true;
  resultMeta.textContent = copy.profiles[profile.value].label + " · Hybrid · top 3";
}

function syncExampleSelection() {
  for (const button of examples.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.textContent === questionInput.value.trim()));
  }
}

function showProfile() {
  clearResult();
  questionInput.value = "";
  questionInput.placeholder = sampleQuestions[profile.value][0];
  examples.replaceChildren();
  for (const question of sampleQuestions[profile.value]) {
    const button = element("button", question, "example-button");
    button.type = "button";
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => {
      clearResult();
      questionInput.value = question;
      syncExampleSelection();
      questionInput.focus();
    });
    examples.append(button);
  }
  corpusList.replaceChildren();
  for (const [name, description] of copy.profiles[profile.value].documents) {
    corpusList.append(element("dt", name), element("dd", description));
  }
  corpusNote.textContent = copy.profiles[profile.value].note;
}

function revealResultOnMobile() {
  if (!window.matchMedia("(max-width: 700px)").matches) return;
  result.scrollIntoView({
    behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
    block: "start",
  });
}

function showError(message) {
  emptyState.hidden = true;
  responseView.hidden = false;
  evidence.replaceChildren();
  status.classList.add("error");
  status.textContent = copy.failed;
  reason.textContent = message;
  reason.hidden = false;
  latency.hidden = true;
  revealResultOnMobile();
}

function sourceLink(url) {
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "https:" && parsed.protocol !== "http:") return null;
    const link = element("a", copy.source);
    link.href = parsed.href;
    link.target = "_blank";
    link.rel = "noreferrer";
    return link;
  } catch {
    return null;
  }
}

function renderEvidence(item, index, citations) {
  const article = document.createElement("article");
  article.className = "evidence-card";
  const number = element("span", String(index + 1).padStart(2, "0"), "evidence-number");
  number.setAttribute("aria-hidden", "true");
  const content = document.createElement("div");
  content.className = "evidence-content";
  content.append(element("h3", documentTitles[item.doc_id] || item.doc_id));
  const meta = document.createElement("div");
  meta.className = "evidence-meta";
  meta.append(element("code", item.chunk_id));
  const link = sourceLink(item.source_url);
  if (link) meta.append(link);
  if (citations.has(item.chunk_id)) {
    meta.append(element("span", copy.cited, "citation-mark"));
  }
  content.append(meta, element("p", readableSource(item.text), "passage"));

  const original = document.createElement("details");
  original.className = "source-details";
  original.append(element("summary", copy.original), element("pre", item.text));
  if (Number.isFinite(item.score)) {
    original.append(element("p", copy.score + ": " + item.score.toFixed(4), "score-note"));
  }
  content.append(original);
  article.append(number, content);
  return article;
}

function renderResponse(body) {
  if (
    !Array.isArray(body.evidence)
    || !Array.isArray(body.citations)
    || !["answer", "abstain"].includes(body.status)
    || body.evidence.some((item) => typeof item.text !== "string")
  ) {
    throw new Error("invalid-response");
  }
  status.textContent = body.status === "answer" ? copy.found : copy.abstained;
  reason.textContent = body.reason ? (copy.reasonLabels[body.reason] || body.reason) : "";
  reason.hidden = !body.reason;
  latency.textContent = Number.isFinite(body.latency_ms)
    ? copy.latency + ": " + body.latency_ms.toFixed(1) + " ms"
    : "";
  latency.hidden = !latency.textContent;
  const citations = new Set(body.citations.map((citation) => citation.chunk_id));
  evidence.replaceChildren(...body.evidence.map((item, index) => renderEvidence(item, index, citations)));
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  clearResult();
  if (!question) {
    showError(copy.blank);
    questionInput.focus();
    return;
  }
  const selectedProfile = profile.value;
  const activeRequest = requestNumber;
  const controller = new AbortController();
  pendingRequest = controller;
  emptyState.hidden = true;
  responseView.hidden = false;
  submittedQuestion.textContent = copy.query + question;
  submittedQuestion.hidden = false;
  status.textContent = copy.searching;
  setBusy(true);
  revealResultOnMobile();
  try {
    const response = await fetch("/v1/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({ question, top_k: 3, profile: selectedProfile }),
    });
    let body;
    try {
      body = await response.json();
    } catch {
      throw new Error("invalid-response");
    }
    if (activeRequest !== requestNumber) return;
    if (!response.ok) {
      showError(typeof body.detail === "string" ? body.detail : copy.apiError);
      return;
    }
    if (body.profile !== selectedProfile) throw new Error("invalid-response");
    renderResponse(body);
    revealResultOnMobile();
  } catch (error) {
    if (error.name === "AbortError" || activeRequest !== requestNumber) return;
    showError(error.message === "invalid-response" ? copy.invalidResponse : copy.networkError);
  } finally {
    if (activeRequest === requestNumber) {
      pendingRequest = null;
      setBusy(false);
    }
  }
});

profile.addEventListener("change", showProfile);
questionInput.addEventListener("input", () => {
  clearResult();
  syncExampleSelection();
});

showProfile();
