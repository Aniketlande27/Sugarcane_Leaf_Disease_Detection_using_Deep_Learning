/* CaneScan frontend. No build step. Uses relative URLs, so the backend must serve this folder
   (it does by default). If you host the frontend elsewhere, set window.CANESCAN_API_BASE in config.js. */
const API_BASE = (window.CANESCAN_API_BASE || "").replace(/\/$/, "");

const view = document.getElementById("view");

/* ---------------------------------------------------------------- helpers */
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

const OFFLINE_MESSAGE = "Can't reach the CaneScan server. Check that the backend is running and that config.js points to it.";

async function api(path, options) {
  let res;
  try {
    res = await fetch(API_BASE + path, options);
  } catch (_) {
    const err = new Error(OFFLINE_MESSAGE);
    err.offline = true;
    throw err;
  }
  if (!res.ok) {
    let message = `Something went wrong (error ${res.status}).`;
    let body = {};
    try { body = await res.json(); if (typeof body.detail === "string") message = body.detail; } catch (_) {}
    const err = new Error(message);
    err.status = res.status;
    err.code = body.code;
    err.hint = body.hint;
    throw err;
  }
  return res.json();
}

const pretty = (s) => s.replace(/[_-]+/g, " ").replace(/^\w/, (c) => c.toUpperCase());
const pct = (x) => `${(x * 100).toFixed(1)}%`;
const fmtDate = (iso) => new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
const mount = (...nodes) => { view.replaceChildren(...nodes); };
const severityLabel = { none: "No disease", low: "Low risk", moderate: "Moderate risk", high: "High risk", unknown: "Risk not rated" };

/* ---------------------------------------------------------------- shared pieces */
function adviceBlock(d) {
  return h("div", { class: "advice" },
    h("dl", {},
      h("dt", { text: "What it is" }), h("dd", { text: d.description }),
      h("dt", { text: "What to look for" }), h("dd", { text: d.symptoms }),
      h("dt", { text: "What to do" }), h("dd", { text: d.treatment }),
      h("dt", { text: "How to prevent it" }), h("dd", { text: d.prevention })
    )
  );
}

function meter(items) {
  return h("div", { class: "meter", role: "list", "aria-label": "Most likely diagnoses" },
    items.map((it, i) =>
      h("div", { class: "meter-row" + (i === 0 ? " first" : ""), role: "listitem" },
        h("span", { text: pretty(it.label) }),
        h("div", { class: "meter-track" }, h("div", { class: "meter-fill", style: `width:${(it.probability * 100).toFixed(1)}%` })),
        h("span", { class: "pct", text: `${Math.round(it.probability * 100)}%` })
      )
    )
  );
}

/* ---------------------------------------------------------------- SCAN view */
function scanView() {
  let file = null;
  let previewUrl = null;

  const galleryInput = h("input", { type: "file", accept: "image/*", hidden: true });
  const cameraInput = h("input", { type: "file", accept: "image/*", capture: "environment", hidden: true });

  const hint = h("div", { class: "hint" },
    h("strong", { text: "Add a leaf photo" }),
    h("span", { text: "Tap here, or drop an image onto this box" })
  );
  const finder = h("div", { class: "finder", role: "button", tabindex: "0", "aria-label": "Choose a leaf photo" },
    h("i", { class: "corner tl" }), h("i", { class: "corner tr" }), h("i", { class: "corner bl" }), h("i", { class: "corner br" }),
    hint
  );

  const analyzeBtn = h("button", { class: "btn primary", disabled: true, text: "Check this leaf" });
  const banner = h("div", { "aria-live": "polite" });
  let modelReady = true;   // assumed until /api/ready says otherwise

  async function refreshReady() {
    let info;
    try {
      const res = await fetch(API_BASE + "/api/ready");     // 200 when ready, 503 (with a JSON body) when not
      info = await res.json();
    } catch (_) {
      info = { ready: false, message: OFFLINE_MESSAGE };
    }
    modelReady = !!info.ready;
    analyzeBtn.disabled = !(modelReady && file);
    if (modelReady) { banner.replaceChildren(); return; }
    banner.replaceChildren(
      h("div", { class: "error", role: "alert" },
        h("strong", { text: "Diagnosis is unavailable right now. " }),
        info.message || "",
        info.hint ? ` ${info.hint}` : "",
        h("div", { class: "btn-row" }, h("button", { class: "btn", text: "Check again", onclick: refreshReady }))
      )
    );
  }
  const resultBox = h("div", { class: "result", "aria-live": "polite" },
    h("h2", { text: "Result" }),
    h("p", { class: "empty", text: "Your diagnosis and treatment advice will appear here." })
  );

  function setFile(f) {
    if (!f) return;
    if (!f.type.startsWith("image/")) { showError("Choose an image file (JPG, PNG or WEBP)."); return; }
    file = f;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(f);
    finder.querySelector("img")?.remove();
    finder.append(h("img", { src: previewUrl, alt: "Selected leaf photo" }));
    hint.style.display = "none";
    analyzeBtn.disabled = !modelReady;
  }

  function showError(message) {
    resultBox.className = "result bad";
    resultBox.replaceChildren(h("h2", { text: "Result" }), h("div", { class: "error", role: "alert", text: message }));
  }

  finder.addEventListener("click", () => galleryInput.click());
  finder.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); galleryInput.click(); } });
  finder.addEventListener("dragover", (e) => { e.preventDefault(); finder.classList.add("drag"); });
  finder.addEventListener("dragleave", () => finder.classList.remove("drag"));
  finder.addEventListener("drop", (e) => { e.preventDefault(); finder.classList.remove("drag"); setFile(e.dataTransfer.files[0]); });
  galleryInput.addEventListener("change", () => setFile(galleryInput.files[0]));
  cameraInput.addEventListener("change", () => setFile(cameraInput.files[0]));

  analyzeBtn.addEventListener("click", async () => {
    if (!file) return;
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Checking...";
    finder.classList.add("scanning");
    try {
      const fd = new FormData();
      fd.append("file", file);
      const data = await api("/api/predict", { method: "POST", body: fd });
      renderResult(data);
    } catch (err) {
      showError(err.message);
      if (err.status === 503 || err.offline) refreshReady();    // show the banner and stop further attempts
    } finally {
      finder.classList.remove("scanning");
      analyzeBtn.disabled = !modelReady;
      analyzeBtn.textContent = "Check this leaf";
    }
  });

  function renderResult(data) {
    const d = data.disease;
    const isHealthy = d && d.severity === "none";
    const sevClass = d ? d.severity : "unknown";
    resultBox.className = "result " + (data.is_uncertain ? "warn" : (isHealthy || sevClass === "low" ? "" : "bad"));

    const parts = [h("h2", { text: "Result" })];

    if (data.is_uncertain) {
      parts.push(
        h("div", { class: "notice" },
          h("strong", { text: "We're not sure about this one. " }),
          `The best match is below ${Math.round(data.threshold * 100)}% confidence. Take a new photo of a single leaf in daylight, with the leaf filling the frame and in focus.`
        )
      );
    }

    parts.push(
      h("p", { class: "verdict" },
        h("span", { class: "verdict", text: d ? d.display_name : pretty(data.label) }),
        d && h("span", { class: `sev ${sevClass}`, text: severityLabel[sevClass] || severityLabel.unknown })
      ),
      h("p", { class: "muted", text: `Confidence ${pct(data.confidence)}` }),
      meter(data.top3)
    );

    if (d && !data.is_uncertain) parts.push(adviceBlock(d));
    parts.push(feedbackBlock(data));
    resultBox.replaceChildren(...parts);
    resultBox.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  function feedbackBlock(data) {
    const box = h("div", { class: "feedback" });
    const ask = () => {
      box.replaceChildren(
        h("strong", { text: "Was this diagnosis right?" }),
        h("div", { class: "btn-row" },
          h("button", { class: "btn", text: "Yes, correct", onclick: () => send({ correct: true }) }),
          h("button", { class: "btn", text: "No, it's wrong", onclick: pickCorrect })
        )
      );
    };
    const pickCorrect = async () => {
      let classes = [];
      try { classes = await api("/api/diseases"); } catch (e) { box.append(h("p", { class: "error", text: e.message })); return; }
      const select = h("select", { "aria-label": "Correct disease" },
        h("option", { value: "", text: "Choose the correct one" }),
        classes.map((c) => h("option", { value: c.name, text: c.display_name }))
      );
      box.replaceChildren(
        h("strong", { text: "What is it really?" }),
        h("div", { class: "btn-row" }, select,
          h("button", { class: "btn primary", text: "Send correction", onclick: () => send({ correct: false, correct_label: select.value }) }))
      );
    };
    const send = async (payload) => {
      try {
        await api(`/api/predictions/${data.id}/feedback`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
        });
        box.replaceChildren(h("p", { class: "muted", text: "Thanks. Your answer helps improve the model." }));
      } catch (e) {
        box.append(h("p", { class: "error", role: "alert", text: e.message }));
      }
    };
    ask();
    return box;
  }

  refreshReady();
  return h("div", {},
    h("h1", { text: "Is your cane leaf healthy?" }),
    h("p", { class: "lede", text: "Photograph one leaf, and get a likely diagnosis with what to do next." }),
    banner,
    h("div", { class: "scan-grid" },
      h("div", {},
        finder, galleryInput, cameraInput,
        h("div", { class: "btn-row" },
          h("button", { class: "btn", text: "Take a photo", onclick: () => cameraInput.click() }),
          h("button", { class: "btn", text: "Choose from gallery", onclick: () => galleryInput.click() }),
          analyzeBtn
        ),
        h("ul", { class: "tips" },
          h("li", { text: "Photograph a single leaf, filling most of the frame." }),
          h("li", { text: "Use daylight and keep the leaf in focus." }),
          h("li", { text: "Include the spotted or discoloured area." })
        )
      ),
      resultBox
    )
  );
}

/* ---------------------------------------------------------------- HISTORY view */
function historyView() {
  const PAGE = 12;
  let offset = 0;
  const list = h("div", { class: "grid-list" });
  const moreBtn = h("button", { class: "btn", text: "Load more", hidden: true });
  const status = h("p", { class: "muted", text: "Loading..." });

  function feedbackTag(it) {
    if (it.feedback_correct === true) return h("span", { class: "tag ok", text: "Confirmed correct" });
    if (it.feedback_correct === false) return h("span", { class: "tag no", text: `Corrected to ${pretty(it.feedback_label || "")}` });
    if (it.is_uncertain) return h("span", { class: "tag unsure", text: "Low confidence" });
    return null;
  }

  async function load() {
    try {
      const data = await api(`/api/history?limit=${PAGE}&offset=${offset}`);
      status.textContent = data.total === 0 ? "No scans yet. Check a leaf on the Scan page." : `${data.total} scan${data.total === 1 ? "" : "s"} so far`;
      for (const it of data.items) {
        list.append(h("article", { class: "scan-item" },
          h("img", { src: API_BASE + it.image_url, alt: `Leaf scanned as ${pretty(it.label)}`, loading: "lazy" }),
          h("div", {},
            h("div", { class: "name", text: pretty(it.label) }),
            h("div", { class: "meta", text: `${pct(it.confidence)} confidence` }),
            h("div", { class: "meta", text: fmtDate(it.created_at) }),
            feedbackTag(it)
          )
        ));
      }
      offset += data.items.length;
      moreBtn.hidden = offset >= data.total;
    } catch (e) {
      status.replaceChildren(h("span", { class: "error", role: "alert", text: e.message }));
    }
  }
  moreBtn.addEventListener("click", load);
  load();

  return h("div", {}, h("h1", { text: "Past scans" }), status, list, h("div", { class: "center" }, moreBtn));
}

/* ---------------------------------------------------------------- GUIDE view */
function guideView() {
  const box = h("div", { class: "guide" }, h("p", { class: "muted", text: "Loading..." }));
  api("/api/diseases")
    .then((items) => {
      box.replaceChildren(...items.map((d) =>
        h("details", {},
          h("summary", {}, d.display_name, h("span", { class: `sev ${d.severity}`, text: severityLabel[d.severity] || severityLabel.unknown })),
          adviceBlock(d)
        )
      ));
    })
    .catch((e) => box.replaceChildren(h("p", { class: "error", role: "alert", text: e.message })));
  return h("div", {},
    h("h1", { text: "Disease guide" }),
    h("p", { class: "lede", text: "Every condition the model can recognise, with symptoms and what to do." }),
    box
  );
}

/* ---------------------------------------------------------------- INSIGHTS view */
function insightsView() {
  const box = h("div", {}, h("p", { class: "muted", text: "Loading..." }));
  api("/api/stats")
    .then((s) => {
      const max = Math.max(1, ...s.by_label.map((x) => x.count));
      const accuracy = s.reviewed ? pct(s.confirmed_correct / s.reviewed) : "No feedback yet";
      box.replaceChildren(
        h("div", { class: "stat-line" },
          h("div", { class: "stat" }, h("b", { text: s.total_scans }), h("span", { text: "scans" })),
          h("div", { class: "stat" }, h("b", { text: s.total_scans ? pct(s.average_confidence) : "-" }), h("span", { text: "average confidence" })),
          h("div", { class: "stat" }, h("b", { text: s.uncertain_scans }), h("span", { text: "low-confidence scans" })),
          h("div", { class: "stat" }, h("b", { text: accuracy }), h("span", { text: `rated correct (${s.reviewed} reviewed)` }))
        ),
        h("h2", { text: "Scans by diagnosis" }),
        s.by_label.length === 0
          ? h("p", { class: "muted", text: "Nothing to show yet." })
          : h("div", { class: "meter" }, s.by_label.map((x) =>
              h("div", { class: "meter-row" },
                h("span", { text: pretty(x.label) }),
                h("div", { class: "meter-track" }, h("div", { class: "meter-fill", style: `width:${(x.count / max) * 100}%` })),
                h("span", { class: "pct", text: x.count })
              )))
      );
    })
    .catch((e) => box.replaceChildren(h("p", { class: "error", role: "alert", text: e.message })));
  return h("div", {}, h("h1", { text: "Insights" }), box);
}

/* ---------------------------------------------------------------- router */
const routes = { scan: scanView, history: historyView, guide: guideView, insights: insightsView };

function route() {
  const name = (location.hash.replace(/^#\//, "") || "scan");
  const render = routes[name] || routes.scan;
  const active = routes[name] ? name : "scan";
  document.querySelectorAll("#nav a").forEach((a) => {
    if (a.dataset.route === active) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  mount(render());
  view.focus({ preventScroll: true });
  window.scrollTo(0, 0);
}

window.addEventListener("hashchange", route);
route();
