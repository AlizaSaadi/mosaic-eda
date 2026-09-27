// The Mosaic office and team: plays office actions from the server, handles clicks, and
// keeps the day/night and color-blind settings.
// Gradio renders components after this script loads, so everything is found lazily.
(function () {
  "use strict";
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const SPEED = 240; // SVG units per second when walking
  let run = null;
  let seen = -1;
  let queue = [];
  let busy = false;

  const $ = (sel) => document.querySelector(sel);
  const member = (name) => document.getElementById("m-" + name);
  const spots = () => {
    const svg = $("#office-svg");
    return svg ? JSON.parse(svg.dataset.spots || "{}") : {};
  };
  const wait = (ms) => new Promise((r) => setTimeout(r, reduce ? Math.min(ms, 300) : ms));
  const pick = (list) => list[Math.floor(Math.random() * list.length)];

  // ---- day/night and color-blind mode (remembered in this browser) ----
  function remember(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch (err) {
      /* private mode: the setting just isn't remembered */
    }
  }
  function recall(key) {
    try {
      return localStorage.getItem(key);
    } catch (err) {
      return null;
    }
  }
  function applyTheme(dark) {
    document.documentElement.classList.toggle("dark", dark);
    document.body.classList.toggle("dark", dark);
    $("#m-theme-btn")?.setAttribute("aria-pressed", String(dark));
  }
  function applyColorBlind(on) {
    document.body.classList.toggle("cb-safe", on);
    $("#m-cb-btn")?.setAttribute("aria-pressed", String(on));
  }
  // The toolbar button sets what the visitor wants; a hidden Gradio checkbox carries it to
  // the server (which recolors the charts). Gradio can render that checkbox seconds after
  // the page loads (slower on Spaces), so the two are kept in step on a timer.
  let wantCb = recall("mosaic-cb") === "1";
  let syncing = 0;
  function colorBlindBox() {
    return document.querySelector("#m-cb input[type=checkbox]");
  }
  function syncColorBlind() {
    applyColorBlind(wantCb);
    const box = colorBlindBox();
    if (box && box.checked !== wantCb && Date.now() - syncing > 1500) {
      syncing = Date.now();
      box.click(); // Gradio sends the change to the server, which redraws the charts
    }
  }
  // called by the hidden checkbox's change event
  window.mosaicColorBlind = function (on) {
    wantCb = on;
    applyColorBlind(on);
    remember("mosaic-cb", on ? "1" : "0");
  };
  function toast(text) {
    let el = document.getElementById("m-toast");
    if (!el) {
      el = document.createElement("div");
      el.id = "m-toast";
      el.setAttribute("role", "status");
      document.body.appendChild(el);
    }
    el.textContent = text;
    el.classList.add("show");
    clearTimeout(el._t);
    el._t = setTimeout(() => el.classList.remove("show"), 3800);
  }
  document.addEventListener("click", (ev) => {
    if (!ev.target.closest) return;
    if (ev.target.closest("#m-theme-btn")) {
      const dark = !document.body.classList.contains("dark");
      applyTheme(dark);
      remember("mosaic-theme", dark ? "dark" : "light");
    } else if (ev.target.closest("#m-cb-btn")) {
      wantCb = !wantCb;
      remember("mosaic-cb", wantCb ? "1" : "0");
      syncing = 0;
      syncColorBlind();
      toast(
        wantCb
          ? "Color-blind colors on: charts and highlights now use colors that stay distinct " +
              "with red-green and blue-yellow color blindness."
          : "Color-blind colors off."
      );
    }
  });
  const loaded = Date.now();
  function restoreSettings() {
    const theme = recall("mosaic-theme");
    // Gradio sets its own theme class while it starts up: apply the saved one after it
    if (Date.now() - loaded < 15000) {
      applyTheme(theme ? theme === "dark" : document.body.classList.contains("dark"));
    }
    syncColorBlind();
  }
  document.addEventListener("DOMContentLoaded", restoreSettings);
  setInterval(restoreSettings, 800);

  // ---- the office ----
  function place(el, x, y) {
    el.setAttribute("transform", `translate(${x},${y})`);
    el.dataset.x = x;
    el.dataset.y = y;
  }

  function at(el) {
    return [parseFloat(el.dataset.x), parseFloat(el.dataset.y)];
  }

  async function walk(el, points) {
    const crt = el.querySelector(".crt");
    el.parentNode.appendChild(el); // walk in front of the desks and other rooms
    crt.classList.add("walking");
    for (const [x, y] of points) {
      const [x0, y0] = at(el);
      const dist = Math.hypot(x - x0, y - y0);
      const ms = reduce ? 1 : (dist / SPEED) * 1000;
      const start = performance.now();
      await new Promise((resolve) => {
        function step(now) {
          const t = Math.min((now - start) / ms, 1);
          const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
          el.setAttribute("transform", `translate(${x0 + (x - x0) * e},${y0 + (y - y0) * e})`);
          if (t < 1) requestAnimationFrame(step);
          else resolve();
        }
        requestAnimationFrame(step);
      });
      el.dataset.x = x;
      el.dataset.y = y;
    }
    crt.classList.remove("walking");
  }

  function behindDesks(el) {
    const svg = $("#office-svg");
    const desk = svg && svg.querySelector(".desk");
    if (desk) svg.insertBefore(el, desk);
  }

  // a speech bubble that grows to fit what's said
  function speak(el, text, ms = 1800) {
    const bubble = el && el.querySelector(".bubble");
    if (!bubble) return;
    const label = bubble.querySelector(".say");
    label.textContent = text;
    const rect = bubble.querySelector("rect");
    let width = text.length * 6.2 + 20;
    try {
      width = Math.max(label.getComputedTextLength() + 22, 60);
    } catch (err) {
      /* not rendered yet: keep the estimate */
    }
    rect.setAttribute("width", width);
    rect.setAttribute("x", -width / 2);
    bubble.classList.add("show");
    clearTimeout(bubble._t);
    bubble._t = setTimeout(() => bubble.classList.remove("show"), ms);
  }
  const say = (name, text, ms) => speak(member(name), text, ms);

  function caption(text) {
    const c = $("#office-caption");
    if (c && text) c.textContent = text;
  }

  function setWorking(name) {
    document.querySelectorAll("#office-svg .member .crt").forEach((c) => c.classList.remove("working"));
    document.querySelectorAll("#office-svg .desk").forEach((d) => d.classList.remove("on"));
    if (!name) return;
    member(name)?.querySelector(".crt").classList.add("working");
    document.querySelector(`#office-svg .desk[data-name="${name}"]`)?.classList.add("on");
  }

  async function visit(from, to, text, opts = {}) {
    const el = member(from);
    const s = spots();
    if (!el || !s[from] || !s[to]) return;
    const crt = el.querySelector(".crt");
    crt.classList.add("carrying");
    if (opts.red) crt.classList.add("marked");
    const home = s[from];
    const there = s[to];
    await walk(el, [home.door, there.door, there[opts.slot || "guest"]]);
    say(from, text);
    if (opts.onArrive) opts.onArrive();
    await wait(opts.stay || 1500);
    crt.classList.remove("carrying", "marked");
    if (opts.stayThere) return;
    await walk(el, [there.door, home.door, home.seat]);
    behindDesks(el);
  }

  async function hop(names, gap = 140) {
    for (const name of names) {
      const crt = member(name)?.querySelector(".crt");
      if (!crt) continue;
      crt.classList.add("hop");
      setTimeout(() => crt.classList.remove("hop"), 450);
      await wait(gap);
    }
  }

  // the last scene: everyone brings their part to Quill, who delivers the report
  async function finale(a) {
    setWorking(null);
    caption("Pip brings the final findings to Rex");
    await visit("Pip", "Rex", "Final findings!", { stay: 1100 });
    caption("Rex signs off");
    say("Rex", "Checked. Approved!", 1600);
    await hop(["Rex"]);
    await wait(1300);
    caption("Rex takes the approved findings to Quill");
    await visit("Rex", "Quill", "All yours, Quill!", { stay: 1100 });
    caption("Tilly and Mop add their notes");
    await Promise.all([
      visit("Tilly", "Quill", "Data notes!", { stay: 1200 }),
      visit("Mop", "Quill", "Cleaning log!", { slot: "guest2", stay: 1200 }),
    ]);
    caption("Quill puts the report together");
    setWorking("Quill");
    say("Quill", "Writing...", 1600);
    await wait(1900);
    setWorking(null);
    say("Quill", "Done!", 1000);
    await wait(700);
    caption("Quill brings you the report");
    await visit("Quill", "lounge", a.say || "Your report!", {
      stay: 900,
      stayThere: true,
      onArrive: () => document.querySelector("#office-svg .visitor-report")?.classList.add("show"),
    });
    caption("Your report is ready");
    const s = spots();
    const names = ["Tilly", "Mop", "Pip", "Rex", "Quill"];
    await Promise.all(
      names.map((name, i) => {
        const el = member(name);
        const home = s[name];
        const lounge = s.lounge;
        if (!el || !home || !lounge || !s.gather) return null;
        const path = name === "Quill" ? [s.gather[i]] : [home.door, lounge.door, s.gather[i]];
        return wait(i * 180).then(() => walk(el, path));
      })
    );
    say("Pip", "Ta-da!", 2200);
    await hop(names, 120);
    await hop(names, 120);
    await wait(1400);
    // then on to the results (the skip button does the same thing sooner)
    if ($("#office-svg")) document.querySelector("#to-results")?.click();
  }

  const handlers = {
    async deliver(a) {
      await visit(a.who, a.to, a.kind === "tea" ? "Your tea!" : "Your coffee!", {
        onArrive: () => {
          const cup = document.querySelector(`#office-svg .visitor-drink[data-kind="${a.kind}"]`);
          if (cup) cup.classList.add("show");
        },
      });
    },
    async work(a) {
      setWorking(a.who);
      caption(a.caption);
      await wait(400);
    },
    async handoff(a) {
      setWorking(null);
      await visit(a.from, a.to, a.say, { red: a.say === "Please revise" });
    },
    async reject(a) {
      const crt = member(a.who)?.querySelector(".crt");
      if (!crt) return;
      crt.classList.add("carrying", "marked", "stressed");
      say(a.who, a.say);
      caption(`${a.who} is fixing something the fact check caught`);
      await wait(1400);
      crt.classList.remove("carrying", "marked", "stressed");
    },
    async say(a) {
      say(a.who, a.say);
      await wait(900);
    },
    async approve(a) {
      const crt = member(a.who)?.querySelector(".crt");
      say(a.who, a.say);
      caption("Rex approved the findings");
      if (crt) {
        crt.classList.add("hop");
        await wait(450);
        crt.classList.remove("hop");
      }
      await wait(600);
    },
    finale,
    async done(a) {
      await finale(a); // older recordings
    },
    async fail(a) {
      setWorking(null);
      caption("The team hit a problem");
      document.querySelectorAll("#office-svg .member .crt").forEach((m) => m.classList.add("worried"));
      say(a.who, a.say, 3000);
    },
  };

  function reset() {
    const s = spots();
    setWorking(null);
    document.querySelectorAll("#office-svg .member").forEach((el) => {
      const p = s[el.dataset.name];
      if (p) place(el, ...p.seat);
      el.querySelector(".crt").className.baseVal = "crt";
      behindDesks(el);
    });
    document
      .querySelectorAll("#office-svg .visitor-drink, #office-svg .visitor-report")
      .forEach((d) => d.classList.remove("show"));
    caption("The team is getting ready");
    queue = [];
    seen = -1;
  }

  async function drain() {
    if (busy) return;
    busy = true;
    while (queue.length) {
      const a = queue.shift();
      const handler = handlers[a.type];
      try {
        if (handler) await handler(a);
      } catch (err) {
        console.warn("office", err);
      }
    }
    busy = false;
  }

  function poll() {
    const box = document.querySelector("#office-state textarea, #office-state input");
    if (!box || !box.value) return;
    let state;
    try {
      state = JSON.parse(box.value);
    } catch (err) {
      return;
    }
    if (state.run !== run) {
      run = state.run;
      reset();
    }
    // a long run can outpace the animations: skip ahead, but keep the latest few
    // (and always the ending)
    const fresh = state.actions.filter((a) => a.id > seen);
    if (fresh.length) {
      seen = fresh[fresh.length - 1].id;
      queue.push(...fresh);
      if (queue.length > 8) queue = [queue[0], ...queue.slice(-6)];
      drain();
    }
  }
  setInterval(poll, 400);

  // ---- clicking a creature: a hop, or a small meltdown after five quick clicks ----
  const LINES = {
    Tilly: ["I'm doing my best!", "Still sorting your files", "Triage takes a steady hand", "One file at a time, please"],
    Mop: ["Yes boss, I'm on it!", "Scrubbing as fast as I can", "Clean data takes elbow grease", "Mind the wet floor!"],
    Pip: ["Good reports take time", "I'm onto something here", "Numbers don't rush, and neither do I", "Shh, I'm counting"],
    Rex: ["Rushing leads to mistakes", "I will check this twice", "Please take a number", "Patience is a virtue"],
    Quill: ["Genius can't be hurried", "Writer's block incoming...", "Every word matters", "Chapter one is almost done"],
  };
  const clicks = new Map();
  document.addEventListener("click", (ev) => {
    const crt = ev.target.closest && ev.target.closest(".crt");
    if (!crt) return;
    const now = Date.now();
    const recent = (clicks.get(crt) || []).filter((t) => now - t < 3000).concat(now);
    clicks.set(crt, recent);
    if (crt.classList.contains("stressed")) return;
    if (recent.length >= 5) {
      crt.classList.add("stressed");
      const name = crt.dataset.name;
      speak(crt.closest(".member"), pick(LINES[name] || ["Too many clicks!"]), 2400);
      setTimeout(() => {
        crt.classList.remove("stressed");
        clicks.set(crt, []);
      }, 2600);
    } else {
      crt.classList.remove("hop");
      void crt.getBBox();
      crt.classList.add("hop");
      setTimeout(() => crt.classList.remove("hop"), 420);
    }
  });
})();
