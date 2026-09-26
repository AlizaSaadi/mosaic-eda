// The Mosaic office and team: plays office actions from the server, and handles clicks.
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

  function say(name, text, ms = 1800) {
    const el = member(name);
    if (!el) return;
    const bubble = el.querySelector(".bubble");
    bubble.querySelector(".say").textContent = text;
    bubble.classList.add("show");
    clearTimeout(bubble._t);
    bubble._t = setTimeout(() => bubble.classList.remove("show"), ms);
  }

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
    await walk(el, [home.door, there.door, there.guest]);
    say(from, text);
    if (opts.onArrive) opts.onArrive();
    await wait(1500);
    crt.classList.remove("carrying", "marked");
    await walk(el, [there.door, home.door, home.seat]);
    const svg = $("#office-svg");
    svg.insertBefore(el, svg.querySelector(".desk")); // back behind the desks
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
    async done(a) {
      setWorking(null);
      caption("Your report is ready");
      say(a.who, a.say, 2500);
      for (const m of document.querySelectorAll("#office-svg .member .crt")) {
        m.classList.add("hop");
        setTimeout(() => m.classList.remove("hop"), 450);
        await wait(140);
      }
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
    });
    document.querySelectorAll("#office-svg .visitor-drink").forEach((d) => d.classList.remove("show"));
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
    const fresh = state.actions.filter((a) => a.id > seen);
    if (fresh.length) {
      seen = fresh[fresh.length - 1].id;
      queue.push(...fresh);
      if (queue.length > 8) queue = [queue[0], ...queue.slice(-6)];
      drain();
    }
  }
  setInterval(poll, 400);

  // clicking a creature: a hop, or a small meltdown after five quick clicks
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
      const bubble = crt.closest(".member")?.querySelector(".bubble");
      if (bubble) {
        const lines = ["Too many clicks!", "I'm working!", "Please stop", "Deep breaths..."];
        bubble.querySelector(".say").textContent = lines[Math.floor(Math.random() * lines.length)];
        bubble.classList.add("show");
        clearTimeout(bubble._t);
        bubble._t = setTimeout(() => bubble.classList.remove("show"), 2200);
      }
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
