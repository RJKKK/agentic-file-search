export function buildDbParams(dbPath, extra = {}) {
  const params = new URLSearchParams();
  if (String(dbPath || "").trim()) {
    params.set("db_path", String(dbPath).trim());
  }
  for (const [key, value] of Object.entries(extra)) {
    if (value !== undefined && value !== null && value !== "") {
      params.set(key, String(value));
    }
  }
  return params;
}

export function withQuery(url, params) {
  const query = params?.toString?.() ?? "";
  return query ? `${url}?${query}` : url;
}

export async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message =
      payload.message || payload.error || payload.detail || `Request failed (${response.status})`;
    const error = new Error(message);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

export async function streamJsonEvents(url, options = {}, handlers = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const message =
      payload.message || payload.error || payload.detail || `Request failed (${response.status})`;
    const error = new Error(message);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  if (!response.body) {
    throw new Error("Streaming response is not available.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const dispatch = (type, payload) => {
    const handler = handlers[type];
    if (typeof handler === "function") {
      handler(payload);
    }
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let separatorIndex = buffer.indexOf("\n\n");
      while (separatorIndex >= 0) {
        const block = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        const lines = block.split("\n");
        let type = "message";
        const dataLines = [];
        for (const line of lines) {
          if (line.startsWith("event:")) {
            type = line.slice(6).trim() || "message";
            continue;
          }
          if (line.startsWith("data:")) {
            dataLines.push(line.slice(5).trimStart());
          }
        }
        if (!dataLines.length) {
          separatorIndex = buffer.indexOf("\n\n");
          continue;
        }
        let payload = {};
        try {
          payload = JSON.parse(dataLines.join("\n"));
        } catch {
          payload = { raw: dataLines.join("\n") };
        }
        dispatch(type, payload);
        separatorIndex = buffer.indexOf("\n\n");
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export function uploadWithProgress(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      onProgress(Math.min(99, Math.round((event.loaded / event.total) * 100)));
    };
    xhr.onload = () => {
      const payload = JSON.parse(xhr.responseText || "{}");
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(payload);
        return;
      }
      reject(new Error(payload.message || payload.error || payload.detail || `Request failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("上传失败，请检查网络或服务状态"));
    xhr.send(formData);
  });
}

export function documentStatusType(status) {
  if (status === "deleted") return "danger";
  if (status === "failed") return "danger";
  if (status === "processing") return "warning";
  if (status === "queued") return "info";
  if (["pages_ready", "indexed", "completed"].includes(status)) return "success";
  if (status === "uploaded") return "warning";
  return "info";
}

export function formatFileSize(size) {
  const bytes = Number(size || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function formatTime(value) {
  const timestamp = Number(value || 0);
  if (!timestamp) return "-";
  return new Date(timestamp * 1000).toLocaleString();
}

export function formatDuration(value) {
  const duration = Number(value ?? 0);
  if (!Number.isFinite(duration) || duration < 0) return "-";
  if (duration < 1000) return `${duration} ms`;
  if (duration < 60_000) return `${(duration / 1000).toFixed(1)} s`;
  const minutes = Math.floor(duration / 60_000);
  const seconds = Math.round((duration % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

export function shortText(value, maxLength = 48) {
  const text = formatValue(value).replace(/\s+/g, " ").trim();
  if (!text) return "";
  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

export function formatValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    return value.map((item) => (typeof item === "object" ? compactObject(item) : String(item))).join("\n");
  }
  if (typeof value === "object") return compactObject(value);
  return String(value);
}

function compactObject(value) {
  if (!value || typeof value !== "object") return String(value || "");
  return Object.entries(value)
    .map(([key, item]) => `${labelize(key)}: ${typeof item === "object" ? JSON.stringify(item, null, 2) : item}`)
    .join("\n");
}

function labelize(value) {
  return String(value).replaceAll("_", " ");
}
