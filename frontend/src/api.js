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
