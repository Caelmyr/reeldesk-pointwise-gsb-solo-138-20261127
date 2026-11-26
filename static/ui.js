export const $ = (id) => document.getElementById(id);

export function h(tag, attrs = {}, ...children) {
  const element = document.createElement(tag);
  for (const [attribute, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (attribute.startsWith("on") && typeof value === "function") {
      element.addEventListener(attribute.slice(2), value);
    } else if (attribute === "class") {
      element.className = value;
    } else if (value === true) {
      element.setAttribute(attribute, "");
    } else {
      element.setAttribute(attribute, String(value));
    }
  }
  for (const child of children.flat(Infinity)) {
    if (child !== undefined && child !== null) element.append(child);
  }
  return element;
}

export function svg(tag, attrs = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attrs)) element.setAttribute(name, value);
  return element;
}

export function button(text, action, kind = "quiet", attrs = {}) {
  return h("button", {type: "button", class: kind, onclick: action, ...attrs}, text);
}

export function link(text, path, attrs = {}) {
  return h("a", {class: "button", href: path, ...attrs}, text);
}

export function input(name, label, value = "", opts = {}) {
  const { type = "text", ...rest } = opts;
  return h("label", {class: "label-field"}, h("span", {}, label),
    h("input", {name, type, value, ...rest}));
}

export function textarea(name, label, value = "", opts = {}) {
  return h("label", {class: "label-field"}, h("span", {}, label), h("textarea", {name, ...opts}, value));
}

export function select(name, label, options, value = "", attrs = {}) {
  return h("label", {class: "label-field"},
    h("span", {}, label),
    h("select", {name, ...attrs}, options.map(option => {
      const [id, title] = Array.isArray(option) ? option : [option, option];
      return h("option", {value: id, selected: id === value}, title);
    })));
}

export function checkbox(name, label, value) {
  return h("label", {class: "check"}, h("input", {name, type: "checkbox", checked: value}), label);
}

export function numberField(name, label, value, min, max, step = 1) {
  return input(name, label, value, {type: "number", min, max, step, required: true});
}

export function section(title, ...children) {
  return h("section", {class: "panel-block"}, h("h3", {}, title), children);
}

export function message(text) {
  return h("p", {class: "help"}, text);
}

export function properties(className, ...children) {
  return h("div", {class: className}, children);
}

export function formValues(element) {
  return Object.fromEntries(new FormData(element).entries());
}

export async function request(path, {method = "GET", data, file, binary = false, signal} = {}) {
  const headers = {};
  let body;
  if (file instanceof FormData) {
    body = file;
  } else if (data !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(data);
  }
  const response = await fetch(path, { method, headers, body, signal, credentials: "same-origin" });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw Error(error.message || `Request failed (${response.status}).`);
  }
  if (response.status === 204) return null;
  return binary ? response.blob() : response.json();
}

export function prettyNumber(value, digits = 4, suffix = "") {
  if (value === null || value === undefined) return "—";
  if (typeof value !== "number") return String(value);
  return `${Number(value.toFixed(digits))}${suffix}`;
}

export function asPercent(value) {
  return value === null || value === undefined ? "—" : `${prettyNumber(value * 100, 1)}%`;
}

export function describeDate(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat(undefined, {dateStyle: "medium", timeStyle: "short"}).format(new Date(value));
}

export function table(headers, rows) {
  if (!rows.length) return h("p", {class: "muted"}, "No matching entries.");
  return h("div", {class: "small-table"}, h("table", {},
    h("thead", {}, h("tr", {}, headers.map(label => h("th", {scope: "col"}, label)))),
    h("tbody", {}, rows.map(row => h("tr", {}, row.map(item => h("td", {}, item ?? "—")))))));
}

export function metrics(items) {
  return h("div", {class: "metrics"}, items.map(([label, value]) =>
    h("div", {class: "metric"}, label, h("strong", {}, value ?? "—"))));
}
