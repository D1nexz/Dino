"""Dino NEAT — обучение динозаврика играть в Chrome T-Rex Runner.

Всё в одном файле: NEAT обучает нейросеть прыгать через кактусы и птиц.
Обучение идёт в быстрой детерминированной симуляции (тот же движок игры,
но с фиксированным шагом времени), показ — в настоящем окне браузера.

Запуск:
    python3 dino.py            # обучение до 15 000 шагов
    python3 dino.py --play     # показать обученного динозавра
"""

import argparse
import asyncio
import copy
import logging
import os
import pickle
import signal
import time

import neat
from playwright.async_api import async_playwright

logger = logging.getLogger("dino")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ── настройка ──────────────────────────────────────────────────────────
GAME_URL = "https://wayou.github.io/t-rex-runner/"
# Браузер для запуска игры. По умолчанию — Яндекс на macOS, если он есть;
# иначе Playwright возьмёт свой Chromium. Можно задать свой путь:
#   DINO_BROWSER=/path/to/browser python3 dino.py --play
YANDEX_PATH = "/Applications/Yandex.app/Contents/MacOS/Yandex"
BROWSER_PATH = os.environ.get("DINO_BROWSER") or (
    YANDEX_PATH if os.path.exists(YANDEX_PATH) else None
)

TARGET_STEPS = 15000      # цель обучения: агент должен прожить столько шагов
MAX_GENERATIONS = 10000   # предохранитель; реально обучение идёт до TARGET_STEPS
SAVE_INTERVAL = 5         # каждые N поколений сохранять чекпоинт
NUM_SIM_PAGES = 3
SEEDS = (1, 2, 3, 101, 202, 303)
FEATURE_VERSION = 7
BEST_PATH = "best_genome.pkl"
BEST_STEPS_PATH = "best_steps.txt"
CHECKPOINT_DIR = "checkpoints"

SEED_PATH = os.path.join("archive", "best_genome_6654.pkl")




JS_RESTART = """
() => {
    const r = window.Runner?.instance_;
    if (!r) return {status: "NO_RUNNER"};
    // Останавливаем текущий цикл анимации.
    try { cancelAnimationFrame(r.raqId); } catch (e) {}
    r.raqId = 0;
    r.updatePending = false;

    // Сбрасываем состояние забега вручную: нажатие Space при уже идущей игре
    // делает ПРЫЖОК, а не рестарт, поэтому агент стартовал посреди чужого
    // забега (наблюдали разброс 251..1265 шагов у одного генома).
    r.playing = true;
    r.crashed = false;
    r.paused = false;
    // activated управляет ходом времени: при false deltaTime = 0 и игра
    // стоит на месте, ожидая первого прыжка. Для замеров он должен быть true.
    r.activated = true;
    r.playingIntro = false;
    r.distanceRan = 0;
    r.runningTime = 0;
    r.invertTimer = 0;
    r.invertTrigger = false;
    r.currentSpeed = r.config.SPEED;
    r.time = performance.now();
    try { r.containerEl.classList.remove('crashed'); } catch (e) {}
    if (r.tRex) {
        r.tRex.playingIntro = false;
        r.tRex.reset();
    }
    if (r.horizon) {
        r.horizon.reset();
        r.horizon.obstacles = [];
    }
    if (r.distanceMeter) {
        try { r.distanceMeter.reset(r.highestScore); } catch (e) {}
    }
    window.__dinoTracker = null;

    // Запускаем игровой цикл заново.
    r.update();
    return {status: "OK"};
}
"""

JS_STATE = """
() => {
    const runner = window.Runner?.instance_;
    if (!runner) return {status: "NO_RUNNER"};
    if (runner.crashed) return {status: "DEAD"};
    if (!runner.playing) return {status: "NOT_PLAYING"};
    const dino = runner.tRex;
    const speed = runner.currentSpeed || 1;
    const score = runner.distanceRan || 0;
    // Высота над землёй: на земле yPos === groundYPos, в прыжке меньше.
    // ВАЖНО: нельзя писать (dino.yPos || groundY) — на апексе yPos === 0,
    // и тогда подставляется земля, высота ошибочно = 0.
    const groundY = (typeof dino.groundYPos === 'number') ? dino.groundYPos : dino.yPos;
    const yPos = (typeof dino.yPos === 'number') ? dino.yPos : groundY;
    const height = Math.max(0, groundY - yPos);
    const obs = runner.horizon?.obstacles || [];

    // Счётчик пройденных препятствий — чистая награда за прогресс.
    const trk = window.__dinoTracker ||
        (window.__dinoTracker = {seen: new WeakSet(), passed: 0});
    for (let i = 0; i < obs.length; i++) {
        const o = obs[i];
        if (trk.seen.has(o)) continue;
        const w = o.typeConfig ? o.typeConfig.width * o.size - 2 : 0;
        if (o.xPos + w < dino.xPos) { trk.seen.add(o); trk.passed++; }
    }

    const base = {status: "OK", height, ducking: !!dino.ducking, speed,
                  score, passed: trk.passed,
                  // Сколько кадров осталось до препятствия. Это единственный
                  // линейно разделимый признак «когда прыгать»: порог по нему
                  // не зависит от скорости, а порог по расстоянию — зависит.
                  ttc: 60,
                  vy: (typeof dino.jumpVelocity === 'number') ? dino.jumpVelocity : 0};
    // Только препятствия впереди (obs[0] может быть уже перепрыгнутым).
    const ahead = obs.filter(function (o) {
        return o.xPos + (o.width || 0) > dino.xPos;
    });
    if (ahead.length === 0) {
        return Object.assign(base, {distance: 1000, otype: 0, oy: 0, ow: 0, oh: 0});
    }
    // Ближайшее по x.
    const o = ahead.reduce(function (a, b) {
        return a.xPos <= b.xPos ? a : b;
    });
    // Реальные значения: o.typeConfig.type (у самого o поля type нет) —
    // CACTUS_SMALL / CACTUS_LARGE / PTERODACTYL.
    const cfg = o.typeConfig || {};
    const t = cfg.type || '';
    let otype = 0;
    if (t === 'PTERODACTYL') otype = 2;
    else if (t.indexOf('CACTUS') === 0) otype = 1;
    return Object.assign(base, {
        distance: o.xPos - dino.xPos,
        // Время до столкновения в кадрах: расстояние / шаг за кадр.
        ttc: Math.max(0, (o.xPos - dino.xPos) / Math.max(1, speed)),
        otype,
        // Верхняя кромка препятствия по y (у птицы: 50 / 75 / 100).
        oy: (typeof o.yPos === 'number') ? o.yPos : 0,
        // Ширина с учётом группы кактусов.
        ow: cfg.width ? cfg.width * (o.size || 1) : (o.width || 0),
        oh: cfg.height || 0,
    });
}
"""

# ── быстрый детерминированный забег ────────────────────────────────────
# Гоняем ТОТ ЖЕ движок игры, что и в браузере, но:
#   1) время идёт фиксированным шагом 1000/60 мс вместо реального (не ждём
#      requestAnimationFrame) — забег на 15 000 кадров считается за доли
#      секунды вместо 4.5 минут;
#   2) Math.random заменён на seeded xorshift32 — все геномы поколения
#      проходят ОДИНАКОВЫЕ препятствия, поэтому отбор идёт по умению, а не
#      по везению (раньше один геном давал 251, 949 и 1265 шагов);
#   3) отрисовка отключена (2D-контексты заменены заглушкой) — на состояние
#      она не влияет.
# Физика, столкновения и появление препятствий остаются кодом самой игры,
# поэтому политика, обученная здесь, работает и в реальном браузере.
JS_EPISODE = """
([spec, seeds, maxSteps]) => {
    const r = window.Runner && window.Runner.instance_;
    if (!r) return {status: "NO_RUNNER"};

    // 1. Рисование не нужно и стоит дорого: настоящий canvas.drawImage на
    //    каждый кадр даёт ~100 мкс против ~1 мкс у физики (1500 мс против
    //    18 мс на забег). Поэтому и заглушаем контексты, и снимаем сами
    //    draw-функции: до контекста дело просто не доходит.
    const noop = function () {};
    const ctxStub = new Proxy({}, {
        get: function () { return noop; },
        set: function () { return true; },
    });
    const kill = function (o) { try { if (o) o.canvasCtx = ctxStub; } catch (e) {} };
    kill(r); kill(r.tRex); kill(r.horizon); kill(r.distanceMeter);
    if (r.horizon) { kill(r.horizon.horizonLine); kill(r.horizon.nightMode); }
    // Облака создают свой контекст из canvas (canvas.getContext('2d')), и
    // рисовались по-настоящему — это и было 95% времени кадра. Отдаём и им
    // заглушку, чтобы новые облака/препятствия тоже её получали.
    try { r.canvas.getContext = function () { return ctxStub; }; } catch (e) {}
    r.clearCanvas = noop;
    r.tRex.draw = noop;
    r.tRex.blink = noop;
    r.horizon.draw = noop;
    if (r.horizon.horizonLine) r.horizon.horizonLine.draw = noop;
    if (r.horizon.nightMode) {
        r.horizon.nightMode.draw = noop;
        r.horizon.nightMode.placeStars = noop;
    }
    r.distanceMeter.draw = noop;
    r.playSound = noop;
    r.gameOver = function () {
        this.stop();
        this.crashed = true;
        this.tRex.update(100, 'CRASHED');
    };

    // 2. Фиксированный шаг времени: ровно один кадр 60 fps.
    const FRAME = 1000 / 60;
    let clock = 0;
    try {
        Object.defineProperty(performance, 'now',
            {value: function () { return clock; }, configurable: true, writable: true});
    } catch (e) { return {status: "NO_CLOCK"}; }
    window.requestAnimationFrame = function () { return 0; };
    window.cancelAnimationFrame = function () {};

    // 3. Детерминированный генератор: одинаковые препятствия для всех геномов.
    const origRandom = Math.random;
    const seedRandom = function (seed) {
        let st = (seed >>> 0) || 0x9E3779B9;
        Math.random = function () {
            st ^= st << 13; st >>>= 0;
            st ^= st >>> 17;
            st ^= st << 5;  st >>>= 0;
            return st / 4294967296;
        };
    };

    // 4. Сеть: плоский список узлов в топологическом порядке.
    const nin = spec.inputs.length;
    const v = new Float64Array(spec.n);
    const activate = function (x) {
        for (let i = 0; i < nin; i++) v[spec.inputs[i]] = x[i];
        const steps = spec.steps;
        // Активация обязана совпадать с NEAT: gain 2.5 и зажим в ±60.
        // Обычный Math.tanh здесь означал бы оценку другой сети (см. комментарий
        // у NEAT_TANH_GAIN в Python-части).
        const gain = spec.gain, lim = spec.limit;
        for (let k = 0; k < steps.length; k++) {
            const nd = steps[k];
            const links = nd.links;
            let sum = 0;
            for (let j = 0; j < links.length; j++) {
                sum += v[links[j][0]] * links[j][1];
            }
            let z = gain * (nd.b + nd.r * sum);
            if (z > lim) z = lim; else if (z < -lim) z = -lim;
            v[nd.o] = Math.tanh(z);
        }
        const res = [];
        for (let i = 0; i < spec.outputs.length; i++) res.push(v[spec.outputs[i]]);
        return res;
    };

    // 5. Наблюдение — порт JS_STATE (obj нужен только для shaping).
    const readState = function () {
        const dino = r.tRex;
        const out = {
            height: Math.max(0, dino.groundYPos - dino.yPos),
            ducking: !!dino.ducking,
            speed: r.currentSpeed,
            ttc: 60, vy: dino.jumpVelocity, otype: 0, oy: 0, obj: null,
        };
        const obs = (r.horizon && r.horizon.obstacles) || [];
        let best = null;
        for (let i = 0; i < obs.length; i++) {
            const o = obs[i];
            if (o.xPos + (o.width || 0) <= dino.xPos) continue;
            if (best === null || o.xPos < best.xPos) best = o;
        }
        if (best === null) return out;
        const cfg = best.typeConfig || {};
        const t = cfg.type || '';
        out.obj = best;
        out.otype = (t === 'PTERODACTYL') ? 2
            : (t.indexOf('CACTUS') === 0 ? 1 : 0);
        out.oy = (typeof best.yPos === 'number') ? best.yPos : 0;
        out.ttc = Math.max(0, (best.xPos - dino.xPos) / Math.max(1, r.currentSpeed));
        return out;
    };

    // 6. 8 входов — порт inputs_of.
    const inputsOf = function (s) {
        return [
            Math.min(1, Math.max(0, s.ttc / 60)),
            Math.min(1, Math.max(0, s.height / 90)),
            Math.min(1, Math.max(0, (s.vy + 11) / 16)),
            Math.min(1, Math.max(0, (s.speed - 6) / 7)),
            s.otype === 1 ? 1 : 0,
            (s.otype === 2 && s.oy >= 95) ? 1 : 0,
            (s.otype === 2 && s.oy >= 65 && s.oy < 95) ? 1 : 0,
            (s.otype === 2 && s.oy < 65) ? 1 : 0,
        ];
    };

    // 7. Сброс в начало забега. Должен быть ПОЛНЫМ: движок сбрасывает не всё.
    //
    //    Horizon.reset() чистит только препятствия, линию и ночной режим. Он
    //    НЕ обнуляет horizon.runningTime (в конструкторе он вообще не задан,
    //    поэтому после первого забега там NaN) и НЕ убирает облака. Облака
    //    при этом тянут Math.random: каждое новое облако — два вызова. Если их
    //    не убрать, следующий забег стартует с другим числом облаков и
    //    расходует генератор иначе — поток препятствий смещается, и один и тот
    //    же сид даёт 265/266/269 пройденных препятствий. Это ломало честность
    //    сравнения: геномы в поколении бежали по разным трассам.
    const resetRun = function () {
        r.raqId = 0; r.updatePending = false;
        r.playing = true; r.crashed = false; r.activated = true;
        r.playingIntro = false; r.inverted = false;
        r.runningTime = 0; r.distanceRan = 0;
        r.invertTimer = 0; r.invertTrigger = false;
        r.currentSpeed = r.config.SPEED;
        r.tRex.playingIntro = false;
        r.tRex.xPos = 50;              // START_X_POS: позиция после интро
        r.tRex.reset();
        r.horizon.reset();
        r.horizon.obstacles = [];
        r.horizon.obstacleHistory = [];
        r.horizon.runningTime = 0;
        // Облака пересоздаём с нуля: ровно одно стартовое, как при init().
        // Старые облака иначе доживают до следующего забега и сдвигают RNG.
        r.horizon.clouds = [];
        r.horizon.addCloud();
        // Линия горизонта тоже «случайная» (ровная/бугристая) — без сброса
        // она тянет Math.random в первом кадре и сдвигает поток препятствий.
        if (r.horizon.horizonLine) {
            const hl = r.horizon.horizonLine;
            hl.reset();
            // reset() не восстанавливает sourceXPos, а именно он определяет,
            // когда линия «перепрыгнет» и вызовет getRandomType() — ещё один
            // вызов Math.random. Возвращаем оба тайла к исходным спрайтам.
            if (hl.spritePos && hl.dimensions) {
                hl.sourceXPos = [hl.spritePos.x,
                                 hl.spritePos.x + hl.dimensions.WIDTH];
            }
        }
        try { r.distanceMeter.reset(r.highestScore); } catch (e) {}
        clock = 0;
        r.time = 0;
    };

    const SHAPE = 8.0;
    // Shaping награждает ТОЛЬКО за реально пройденное препятствие, к которому
    // динозавр подошёл не «стоя на земле». Раньше здесь была зашита гипотеза,
    // какое действие «правильное» для каждого типа препятствия (кактус — прыжок,
    // птица oy=75 — присед, птица oy=50 — бездействие). Замер на движке её
    // опроверг: чемпион проходит ВСЁ прыжком — 268/268 препятствий за забег,
    // 0 приседов, 15 000 шагов. Прыжок поднимает динозавра выше любой птицы,
    // поэтому и низкая, и средняя, и высокая птица проходятся одинаково.
    // Из-за ложных штрафов фитнесс выходил НИЖЕ числа шагов (15 534 при
    // 15 000), то есть отбор наказывал работающую тактику. Теперь штрафов
    // за действия нет вообще: награда — за уклонение, отсутствие награды —
    // единственная «кара». Фитнесс всегда ≥ числа шагов.
    const CONTACT_TTC = 12;
    const results = [];

    try {
        for (let si = 0; si < seeds.length; si++) {
            seedRandom(seeds[si]);
            resetRun();

            // Счётчик пройденных препятствий — порт JS_STATE.
            const seen = new WeakSet();
            // Какие препятствия были «уклонены» (в воздухе или в приседе).
            const averted = new WeakMap();
            let passed = 0;

            let steps = 0, shapeSum = 0, jumps = 0, ducks = 0, avertedCount = 0;
            let prev = null;

            for (;;) {
                if (steps >= maxSteps || r.crashed) break;
                const s = readState();
                prev = s;
                const out = activate(inputsOf(s));

                const dino = r.tRex;
                const onGround = !dino.jumping;
                const wantJump = out[0] > 0 && onGround;
                const wantDuck = out[1] > 0 && onGround && !wantJump;

                // Состояние спрашиваем у движка, а не держим своё: его reset()
                // после приземления сам снимает присед, и локальный флаг
                // рассинхронизировался — динозавр переставал приседать навсегда
                // после первого прыжка и умирал на средней птице.
                if (wantJump) {
                    // Движок игнорирует Space, пока динозавр приседает.
                    if (dino.ducking) dino.setDuck(false);
                    if (!dino.jumping) {
                        dino.startJump(r.currentSpeed);
                        jumps++;
                    }
                } else if (wantDuck) {
                    if (!dino.ducking) { dino.setDuck(true); ducks++; }
                } else if (dino.ducking && !dino.jumping) {
                    dino.setDuck(false);
                }

                // Shaping: отмечаем уклонение ДО контакта, награду выдаём
                // позже, когда препятствие реально осталось позади.
                // ttc убывает на 1 за кадр, первый кадр контакта наступает
                // при ttc ≈ 42/speed (3-9 кадров), поэтому CONTACT_TTC=12
                // гарантированно раньше — иначе вердикт никогда не выносится:
                // забег обрывается на столкновении.
                if (s.obj && !averted.has(s.obj) && s.ttc <= CONTACT_TTC) {
                    // «Уклонение» = динозавр не стоит на земле в этот момент:
                    // либо в прыжке, либо в приседе. Что именно — не важно,
                    // движок сам решает, прошло ли столкновение.
                    averted.set(s.obj, dino.jumping || dino.ducking);
                }

                steps++;

                // Один кадр движка: deltaTime ровно 1000/60 мс.
                r.time = clock;
                clock += FRAME;
                r.update();

                // Учёт пройденных препятствий.
                const obs = r.horizon.obstacles || [];
                for (let i = 0; i < obs.length; i++) {
                    const o = obs[i];
                    if (seen.has(o)) continue;
                    const w = o.typeConfig ? o.typeConfig.width * o.size - 2 : 0;
                    if (o.xPos + w < r.tRex.xPos) {
                        seen.add(o);
                        passed++;
                        // Награда только за пройденное препятствие, к которому
                        // подошли в воздухе/приседе. Стоя на земле кактус не
                        // пройти, так что это честный признак уклонения.
                        if (averted.get(o)) { shapeSum += SHAPE; avertedCount++; }
                    }
                }
            }

            if (r.tRex.ducking) { try { r.tRex.setDuck(false); } catch (e) {} }

            results.push({
                steps: steps,
                fitness: steps + passed * 5 + shapeSum,
                jumps: jumps, ducks: ducks, passed: passed, averted: avertedCount,
                death: r.crashed ? {
                    steps: steps,
                    otype: prev ? prev.otype : 0,
                    oy: prev ? prev.oy : 0,
                    height: prev ? prev.height : 0,
                } : null,
            });
        }
        return {status: "OK", runs: results};
    } finally {
        Math.random = origRandom;
    }
}
"""


# Функция активации, которой пользуется NEAT. Это НЕ math.tanh: библиотека
# применяет усиление 2.5 и зажимает пре-активацию в ±60:
#
#     z = max(-60, min(60, 2.5 * z)); return math.tanh(z)
#
# Симулятор считал обычный Math.tanh, то есть оценивал ДРУГУЮ сеть, чем та,
# которую эволюционировал NEAT. Из-за этого фитнесс не соответствовал геному:
# отбор шёл по одной функции, а чемпион записывался по другой, и обучение
# стояло на месте. Значение продублировано в JS (apply in JS_EPISODE) —
# менять их надо вместе.
NEAT_TANH_GAIN = 2.5
NEAT_TANH_LIMIT = 60.0


def build_net_spec(genome, neat_config):
    """Сеть NEAT → плоский список шагов для JS-симулятора.

    Формат: узлы уже в топологическом порядке (node_evals строится по
    слоям), поэтому JS считает их одним проходом без рекурсии.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
    idx = {}
    for i, key in enumerate(net.input_nodes):
        idx[key] = i
    steps = []
    for node, act, agg, bias, response, links in net.node_evals:
        # Конфиг разрешает только tanh (activation_mutate_rate = 0), поэтому
        # проверка дешёвая, но защищает от тихой подмены функции.
        if act is not neat.activations.tanh_activation:
            raise ValueError(
                f"build_net_spec знает только tanh, а у узла {node} "
                f"активация {act.__name__}")
        idx[node] = len(idx)
        steps.append({
            "o": idx[node],
            "b": float(bias),
            "r": float(response),
            "links": [[idx[i], float(w)] for i, w in links],
        })
    return {
        "n": len(idx),
        "inputs": [idx[k] for k in net.input_nodes],
        "outputs": [idx[k] for k in net.output_nodes],
        "steps": steps,
        "gain": NEAT_TANH_GAIN,
        "limit": NEAT_TANH_LIMIT,
    }


class Game:
    """Один браузер + одна игра T-Rex Runner."""

    def __init__(self, browser_id: int, browser, page, context=None):
        self.browser_id = browser_id
        self.browser = browser
        self.page = page
        self.context = context
        self.busy = False
        self.broken = False

    async def start(self):
        await self.page.goto(GAME_URL, wait_until="domcontentloaded", timeout=60000)
        await self.page.wait_for_timeout(2000)

    async def get_state(self):
        try:
            # Таймаут обязателен: если рендерер завис, page.evaluate висит
            # бесконечно (наблюдали поколения по 61 минуте).
            r = await asyncio.wait_for(self.page.evaluate(JS_STATE), timeout=5.0)
            return r
        except Exception:
            return {"status": "ERROR"}

    async def start_game(self):
        # ВАЖНО: игру надо ПЕРЕЗАПУСКАТЬ, а не «продолжать».
        #
        # Раньше здесь нажимался Space и ждалось 1.5 с. Если игра уже шла
        # (playing=True, crashed=False), Space делает ПРЫЖОК, а не рестарт —
        # и агент подхватывал забег, который шёл неизвестно сколько. Замер:
        # один и тот же геном давал 251, 949 и 1265 шагов (разброс в 15 раз),
        # а в логах появлялись «смерти» на 29-м шаге, хотя препятствия
        # возникают только через CLEAR_TIME = 3000 мс (180 кадров).
        #
        # Теперь сбрасываем состояние напрямую через движок: останавливаем
        # цикл, обнуляем distanceRan/runningTime/obstacles и запускаем заново.
        # Это даёт одинаковый старт для всех геномов — без него отбор шёл по
        # везению, а не по качеству политики.
        try:
            await self.page.keyboard.up("ArrowDown")
        except Exception:
            pass
        try:
            res = await asyncio.wait_for(self.page.evaluate(JS_RESTART), timeout=5.0)
        except Exception:
            res = {"status": "ERROR"}
        if res.get("status") != "OK":
            logger.warning("Браузер %d: не удалось перезапустить игру (%s)",
                           self.browser_id, res.get("status"))
            return res
        return await self.get_state()

    async def jump(self):
        await self.page.keyboard.press("Space")

    async def set_duck(self, on):
        """Держать/отпустить стрелку вниз (присед)."""
        if on:
            await self.page.keyboard.down("ArrowDown")
        else:
            await self.page.keyboard.up("ArrowDown")

    async def close(self):
        try:
            await self.context.close()
        except Exception:
            pass
        try:
            await self.browser.close()
        except Exception:
            pass


def inputs_of(state):
    """8 входов сети — все линейно разделимые для правила «когда прыгать».

    Главный вход — ttc (время до столкновения в кадрах). Порог по нему не
    зависит от скорости игры, в отличие от порога по расстоянию: раньше
    dist обрезался на 250 px, и при скорости 12 (шаг 12 px/кадр) 250 px
    выглядели как «далеко», хотя это всего 20 кадров — динозавр прыгал
    слишком рано и приземлялся в кактус.

    Высоты птиц проверены экспериментом на движке (по ~85 встреч каждого
    типа): yPos=100 — низкая, нужно ПРЫГАТЬ; yPos=75 — средняя, нужно
    ПРИСЕДАТЬ; yPos=50 — высокая, пролетает над головой, не делать НИЧЕГО.

    Вход bird_high добавлен отдельно, потому что без него высокая птица
    давала те же три нулевых флага, что и «препятствия нет»: сеть не могла
    отличить её от пустой дороги и, выучив «ttc маленькое → прыжок»,
    прыгала прямо в птицу. Именно на этом умирал чемпион версии 5 (3 из 8
    незнакомых сидов, смерть всегда «птица oy=50»).
    """
    # 0..60 кадров → 0..1. Прыжок длится ~36 кадров, взлёт ~18.
    ttc = float(state.get("ttc", 60))
    ttc_n = min(1.0, max(0.0, ttc / 60.0))
    height = min(1.0, max(0.0, float(state.get("height", 0.0)) / 90.0))
    # Вертикальная скорость: -11 (взлёт) .. +5 (падение).
    vy = min(1.0, max(0.0, (float(state.get("vy", 0.0)) + 11.0) / 16.0))
    speed = min(1.0, max(0.0, (float(state.get("speed", 6)) - 6.0) / 7.0))
    otype = int(state.get("otype", 0))
    is_cactus = 1.0 if otype == 1 else 0.0
    oy = float(state.get("oy", 0))
    bird_low = 1.0 if otype == 2 and oy >= 95 else 0.0
    bird_mid = 1.0 if otype == 2 and 65 <= oy < 95 else 0.0
    bird_high = 1.0 if otype == 2 and oy < 65 else 0.0
    return [ttc_n, height, vy, speed, is_cactus, bird_low, bird_mid, bird_high]


class Trainer:
    def __init__(self, max_generations, max_steps):
        self.max_generations = max_generations
        self.max_steps = max_steps
        self.neat_config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
            neat.DefaultStagnation, "neat_config.ini",
        )
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        self.gen = 0
        self.running = True
        self.playwright = None
        self.browser = None
        self.context = None
        self.pages = []
        self.page_pool = None
        self.best_fit = 0.0
        self.best_genome = None
        self.best_steps = 0

    # ── быстрая симуляция ──────────────────────────────────────────────
    async def start_sim(self):
        """Один headless-браузер с несколькими страницами-симуляторами.

        Страницы нужны только как хост JS: игровой цикл в них не крутится
        (requestAnimationFrame заглушён), поэтому память и CPU почти не
        расходуются.
        """
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True, executable_path=BROWSER_PATH,
            args=[
                "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage",
                "--disable-extensions", "--disable-background-networking",
                "--disable-sync", "--no-first-run", "--disable-component-update",
                "--disable-features=Translate,BackForwardCache,MediaRouter",
                "--disable-renderer-backgrounding",
                "--disable-background-timer-throttling",
                "--js-flags=--max-old-space-size=256",
            ],
        )
        self.context = await self.browser.new_context(
            viewport={"width": 800, "height": 600})
        self.pages = []
        for _ in range(NUM_SIM_PAGES):
            page = await self.context.new_page()
            await page.goto(GAME_URL, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(1500)
            if not await page.evaluate("() => !!window.Runner?.instance_"):
                raise RuntimeError("игра не загрузилась в симуляторе")
            self.pages.append(page)
        self.page_pool = asyncio.Queue()
        for page in self.pages:
            self.page_pool.put_nowait(page)
        logger.info("Симулятор готов: %d страниц, сиды %s", len(self.pages), SEEDS)

    async def stop_sim(self):
        for page in getattr(self, "pages", []):
            try:
                await page.close()
            except Exception:
                pass
        for attr in ("context", "browser"):
            obj = getattr(self, attr, None)
            if obj:
                try:
                    await obj.close()
                except Exception:
                    pass
        if self.playwright:
            await self.playwright.stop()
        self.pages = []

    async def sim_run(self, spec, seeds, max_steps):
        """Прогнать геном по списку сидов в симуляторе."""
        page = await self.page_pool.get()
        try:
            for _ in range(2):
                try:
                    res = await asyncio.wait_for(
                        page.evaluate(JS_EPISODE, [spec, seeds, max_steps]),
                        timeout=120.0)
                except Exception as exc:
                    logger.warning("Симулятор: сбой (%s), перезагружаю страницу",
                                   type(exc).__name__)
                    try:
                        await page.goto(GAME_URL, wait_until="domcontentloaded",
                                        timeout=60000)
                        await page.wait_for_timeout(1500)
                    except Exception:
                        pass
                    res = {"status": "ERROR"}
                if res.get("status") == "OK":
                    return res["runs"]
            return [{"steps": 0, "fitness": 0.0, "jumps": 0, "ducks": 0,
                     "passed": 0, "death": None} for _ in seeds]
        finally:
            self.page_pool.put_nowait(page)

    async def eval_genome(self, genome):
        """Оценить геном на наборе фиксированных сидов.

        Один сид = одна и та же последовательность препятствий для всех
        геномов поколения (честное сравнение). Несколько сидов нужны, чтобы
        политика не подгонялась под единственный расклад: итоговый фитнесс —
        среднее, а «шаги» — минимум по сидам (выжить надо везде).
        """
        spec = build_net_spec(genome, self.neat_config)
        runs = await self.sim_run(spec, SEEDS, self.max_steps)
        fits = [r["fitness"] for r in runs]
        steps = [r["steps"] for r in runs]
        best = max(runs, key=lambda r: r["steps"])
        fit = sum(fits) / len(fits)
        genome.fitness = fit
        return {
            "fitness": fit,
            "steps": min(steps),
            "best_steps": max(steps),
            "jumps": best["jumps"],
            "ducks": best["ducks"],
            "passed": best["passed"],
            "death": best.get("death"),
            "reached_target": min(steps) >= TARGET_STEPS,
        }

    @staticmethod
    def _genome_signature(genome):
        """Точная «отпечаток» генома: структура + веса + смещения.

        Нужен, чтобы отличить настоящего чемпиона от потомка с теми же
        ключами узлов, но другими весами.
        """
        conns = tuple(sorted(
            (int(i), int(o), round(float(c.weight), 6), bool(c.enabled))
            for (i, o), c in genome.connections.items()
        ))
        nodes = tuple(sorted(
            (int(k), round(float(n.bias), 6), round(float(n.response), 6))
            for k, n in genome.nodes.items()
        ))
        return (conns, nodes)

    def _inject_champion(self, population):
        """Вернуть текущего чемпиона в популяцию.

        NEAT-элитизм работает по фитнессу *текущего* поколения, а он шумный:
        хороший геном может случайно умереть рано и вылететь, после чего
        популяция деградирует (наблюдали падение 5634 → 75 шагов).
        Поэтому чемпион по лучшему за всё время результату всегда остаётся
        в популяции и заново оценивается каждое поколение.
        """
        if self.best_genome is None:
            return population
        # Ищем структурно И численно идентичный геном. Сравнивать только
        # ключи узлов/связей нельзя: после мутации ключи те же, а веса другие —
        # тогда чемпион считался «уже в популяции» и не инжектировался
        # (наблюдали выпадение чемпиона: лучший фитнесс в поколении 4287
        # вместо 7199).
        key = self._genome_signature(self.best_genome)
        for g in population.values():
            if self._genome_signature(g) == key:
                return population
        new_key = max(population.keys(), default=0) + 1
        champion = copy.deepcopy(self.best_genome)
        # Заранее проставляем известный лучший фитнесс: иначе случайный ранний
        # забег даст низкий балл и чемпиона снова выбросят.
        champion.fitness = max(float(self.best_fit), 1.0)
        # Ключ обязан быть уникальным: GenomeDistanceCache индексирует по нему,
        # а внутренний счётчик NEAT при перезапуске начинается с 1.
        while new_key in population:
            new_key += 1
        champion.key = new_key
        # Вытесняем самого слабого (None считаем нулём).
        weakest = min(population.items(),
                      key=lambda kv: float(kv[1].fitness or 0.0))
        if len(population) >= self.neat_config.pop_size:
            del population[weakest[0]]
        population[new_key] = champion
        return population

    async def eval_generation(self, genomes):
        """Оценить всех геномов поколения в быстрой симуляции.

        Геномы считаются параллельно на пуле страниц, но каждая страница
        синхронно прогоняет свой забег — это доли секунды, поэтому нет смысла
        дробить на большее число задач.
        """
        sem = asyncio.Semaphore(len(self.pages))

        async def run_one(gid, genome):
            async with sem:
                try:
                    res = await self.eval_genome(genome)
                except Exception as exc:
                    logger.error("Геном %d: ошибка оценки (%s)", gid, exc)
                    genome.fitness = 0.0
                    return {"fitness": 0.0, "steps": 0, "best_steps": 0,
                            "jumps": 0, "ducks": 0, "passed": 0,
                            "death": None, "reached_target": False}
                res["gid"] = gid
                return res

        scored = await asyncio.gather(
            *(run_one(gid, g) for gid, g in genomes))
        scored = [r for r in scored if r["fitness"] > 0]

        if scored:
            # Лучший — по прожитым шагам (это и есть цель), при равенстве по
            # фитнессу: он учитывает бонус за пройденные препятствия.
            best = max(scored, key=lambda r: (r["steps"], r["fitness"]))
            best_fit = best["fitness"]
            best_steps = best["steps"]
            death = best.get("death") or {}
            logger.info(
                "Поколение %d: лучший %d шагов (в лучшем забеге %d, фитнесс %.1f, "
                "прыжков %d, приседов %d, смерть: тип=%d oy=%d высота=%.0f)",
                self.gen, best_steps, best.get("best_steps", best_steps),
                best_fit, best.get("jumps", 0), best.get("ducks", 0),
                int(death.get("otype", 0)), int(death.get("oy", 0)),
                float(death.get("height", 0)),
            )
            if (best_steps > self.best_steps or (
                    best_steps == self.best_steps and best_fit > self.best_fit)):
                self.best_fit = best_fit
                self.best_steps = best_steps
                genome = next(g for gid, g in genomes if gid == best["gid"])
                self.best_genome = copy.deepcopy(genome)
                self.best_genome.fitness = best_fit
                with open(BEST_PATH, "wb") as f:
                    pickle.dump({"genome": self.best_genome,
                                 "feature_version": FEATURE_VERSION,
                                 "steps": best_steps,
                                 "fitness": best_fit}, f)
                with open(BEST_STEPS_PATH, "w", encoding="utf-8") as f:
                    f.write(str(best_steps))
                logger.info(
                    "  ^ новый рекорд: %d шагов (геном %d)", best_steps, best["gid"])
            if best.get("reached_target"):
                # reached_target = минимум по ВСЕМ сидам >= TARGET_STEPS.
                logger.info(
                    "Цель достигнута: %d шагов на всех %d сидах (поколение %d)",
                    best_steps, len(SEEDS), self.gen,
                )
                self.running = False
        self.gen += 1

        # Перед размножением закрепляем за чемпионом его лучший фитнесс.
        if self.best_genome is not None:
            key = self._genome_signature(self.best_genome)
            for g in self.pop.population.values():
                if self._genome_signature(g) == key:
                    g.fitness = max(float(self.best_fit), 1.0)
                    break

        new_pop = self.pop.reproduction.reproduce(
            self.neat_config, self.pop.species, self.neat_config.pop_size, self.gen
        )
        if not new_pop:
            new_pop = self.pop.reproduction.create_new(
                self.neat_config.genome_type, self.neat_config.genome_config,
                self.neat_config.pop_size,
            )
        new_pop = self._inject_champion(new_pop)
        self.pop.population = new_pop
        self.pop.species.speciate(self.neat_config, self.pop.population, self.gen)

        if self.gen % SAVE_INTERVAL == 0:
            self.save_checkpoint()

    def _seed_from_archive(self):
        """Подсадить в новую популяцию архивного чемпиона (затравка).

        Нужно, чтобы не начинать с нуля: архивный геном умеет проходить
        кактусы и низких птиц. Его фитнесс сбрасывается — прежний балл
        измерен при другой шкале shaping, и закреплять его нельзя.
        """
        if not os.path.exists(SEED_PATH):
            return
        try:
            with open(SEED_PATH, "rb") as f:
                blob = pickle.load(f)
            seed = blob.get("genome") if isinstance(blob, dict) else blob
            if seed is None or not seed.nodes or not seed.connections:
                return
            # Проверяем совместимость формы сети с текущим конфигом.
            neat.nn.FeedForwardNetwork.create(seed, self.neat_config)
        except Exception as exc:
            logger.warning("Не удалось использовать затравку %s: %s", SEED_PATH, exc)
            return

        seed = copy.deepcopy(seed)
        # Честный старт: балл неизвестен, пусть его измерят в этом поколении.
        seed.fitness = None
        new_key = max(self.pop.population.keys(), default=0) + 1
        while new_key in self.pop.population:
            new_key += 1
        seed.key = new_key
        # Вытесняем самого слабого (у новых геномов фитнесс ещё None).
        weakest = min(self.pop.population.items(),
                      key=lambda kv: float(kv[1].fitness or 0.0))
        if len(self.pop.population) >= self.neat_config.pop_size:
            del self.pop.population[weakest[0]]
        self.pop.population[new_key] = seed
        self.pop.species.speciate(self.neat_config, self.pop.population, self.gen)
        logger.info(
            "Затравка из архива: геном с %s шагами добавлен в популяцию "
            "(фитнесс будет измерен заново)", blob.get("steps", "?"),
        )

    def save_checkpoint(self):
        path = os.path.join(CHECKPOINT_DIR, f"neat_checkpoint_gen_{self.gen}.pkl")
        data = {"generation": self.gen, "best_fit": self.best_fit,
                "best_steps": self.best_steps,
                "target_steps": TARGET_STEPS,
                "feature_version": FEATURE_VERSION,
                "num_inputs": self.neat_config.genome_config.num_inputs,
                "num_outputs": self.neat_config.genome_config.num_outputs,
                "population": self.pop.population if self.pop else {}}
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Чекпоинт: %s", path)

    def _checkpoint_compatible(self, data):
        """Чекпоинт подходит, только если у него та же форма сети и те же признаки."""
        gc = self.neat_config.genome_config
        return (data.get("feature_version") == FEATURE_VERSION
                and data.get("num_inputs") == gc.num_inputs
                and data.get("num_outputs") == gc.num_outputs)

    def load_checkpoint(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.gen = data.get("generation", 0)
        self.best_fit = data.get("best_fit", 0.0)
        self.best_steps = data.get("best_steps", 0)
        self.pop.population = data.get("population", {})
        # Пересоздаём виды для загруженной популяции
        self.pop.species.speciate(self.neat_config, self.pop.population, self.gen)

    def find_latest_checkpoint(self):
        """Вернуть путь к самому свежему совместимому чекпоинту (или None).

        Чекпоинты со старой формой сети (4 входа / 1 выход) пропускаются:
        такие геномы нельзя прогнать через новую сеть.
        """
        if not os.path.isdir(CHECKPOINT_DIR):
            return None
        candidates = []
        for name in os.listdir(CHECKPOINT_DIR):
            if not (name.startswith("neat_checkpoint_gen_") and name.endswith(".pkl")):
                continue
            try:
                num = int(name[len("neat_checkpoint_gen_"):-len(".pkl")])
            except ValueError:
                continue
            candidates.append((num, os.path.join(CHECKPOINT_DIR, name)))
        for num, path in sorted(candidates, reverse=True):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
            except Exception:
                continue
            if self._checkpoint_compatible(data):
                return path
            logger.info("Пропускаю несовместимый чекпоинт %s", path)
        return None

    async def train(self):
        # Корректная остановка по Ctrl+C/SIGTERM: ставим обработчики на текущий loop.
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                asyncio.get_running_loop().add_signal_handler(sig, self.stop)
            except (NotImplementedError, RuntimeError):
                pass
        await self.start_sim()
        try:
            self.pop = neat.Population(self.neat_config)

            cp = self.find_latest_checkpoint()
            if cp:
                self.load_checkpoint(cp)
                logger.info(
                    "Загружен чекпоинт %s (поколение %d, лучшие шаги %d)",
                    cp, self.gen, self.best_steps,
                )
                if self.best_steps >= TARGET_STEPS:
                    logger.info("Цель %d шагов уже достигнута — обучение не требуется", TARGET_STEPS)
                    return
            else:
                logger.info("Новая популяция NEAT")

            # Подхватываем чемпиона с прошлого запуска. Файл читаем только если
            # он сохранён на ТЕХ ЖЕ признаках: иначе его фитнесс и «рекорд
            # шагов» относятся к другому пространству входов и врут.
            if self.best_genome is None and os.path.exists(BEST_PATH):
                try:
                    with open(BEST_PATH, "rb") as f:
                        blob = pickle.load(f)
                    if isinstance(blob, dict):
                        champ = blob.get("genome")
                        fv = blob.get("feature_version")
                        champ_steps = blob.get("steps")
                    else:
                        # Старый формат (голый геном) — версия неизвестна.
                        champ, fv, champ_steps = blob, None, None
                    if fv != FEATURE_VERSION:
                        logger.warning(
                            "Игнорирую %s: версия признаков %s, ожидается %d",
                            BEST_PATH, fv, FEATURE_VERSION,
                        )
                        champ = None
                    if champ is not None and len(champ.nodes) > 0 and len(champ.connections) > 0:
                        # Проверяем, что форма сети подходит.
                        neat.nn.FeedForwardNetwork.create(champ, self.neat_config)
                        self.best_genome = champ
                        self.best_fit = float(champ.fitness or 0.0)
                        self.best_steps = int(champ_steps or 0)
                        logger.info(
                            "Подхвачен чемпион из %s (фитнесс %.1f, шагов %d)",
                            BEST_PATH, self.best_fit, self.best_steps,
                        )
                except Exception as exc:
                    logger.warning("Не удалось загрузить %s: %s", BEST_PATH, exc)

            # Чемпион должен участвовать уже в первом поколении, иначе он
            # появится только после размножения и потеряет целый цикл.
            if self.best_genome is not None:
                self.pop.population = self._inject_champion(dict(self.pop.population))
                self.pop.species.speciate(self.neat_config, self.pop.population, self.gen)
            else:
                # Нет чемпиона текущей версии — берём архивного как затравку.
                # Он умеет проходить кактусы (6654 шага), но не умеет приседать.
                # Фитнесс НЕ закрепляем: старый балл 7199 измерен при другой
                # шкале shaping и несравним с новыми числами. Пусть конкурирует
                # честно — так его навык не потеряется, но и не заблокирует
                # поиск, как это случилось с монотонным фитнессом.
                self._seed_from_archive()

            logger.info("Цель обучения: %d шагов без смерти", TARGET_STEPS)
            while self.running and self.gen < self.max_generations:
                logger.info("=== Поколение %d ===", self.gen)
                await self.eval_generation(list(self.pop.population.items()))
                if not self.running:
                    break
        finally:
            await self.stop_sim()
        logger.info(
            "Обучение завершено. Лучшие шаги: %d, лучший фитнесс: %.2f",
            self.best_steps, self.best_fit,
        )

    def stop(self):
        self.running = False


# ── автопилот для показа в реальном окне ───────────────────────────────
# Прошлый --play опрашивал состояние из Python через Playwright и успевал
# ~25 раз в секунду, тогда как игра идёт 60 кадров в секунду. Решение
# запаздывало на 2-3 кадра, и чемпион гибнул на 134-м шаге — ещё ДО появления
# препятствий (они начинаются после CLEAR_TIME = 3000 мс = 180 кадров).
#
# Поэтому решение принимается ВНУТРИ страницы, в обёртке вокруг Runner.update():
# ровно один раз на кадр движка и ДО расчёта физики. Python только наблюдает.
JS_PLAY_INSTALL = """
(spec) => {
    const r = window.Runner && window.Runner.instance_;
    if (!r) return {status: "NO_RUNNER"};
    if (window.__auto) window.__auto.stop();

    // Сеть: тот же проход, что в симуляторе, с усилением и зажимом NEAT.
    const nin = spec.inputs.length;
    const v = new Float64Array(spec.n);
    const gain = spec.gain, lim = spec.limit;
    const activate = function (x) {
        for (let i = 0; i < nin; i++) v[spec.inputs[i]] = x[i];
        for (let k = 0; k < spec.steps.length; k++) {
            const nd = spec.steps[k], links = nd.links;
            let sum = 0;
            for (let j = 0; j < links.length; j++) sum += v[links[j][0]] * links[j][1];
            let z = gain * (nd.b + nd.r * sum);
            if (z > lim) z = lim; else if (z < -lim) z = -lim;
            v[nd.o] = Math.tanh(z);
        }
        const res = [];
        for (let i = 0; i < spec.outputs.length; i++) res.push(v[spec.outputs[i]]);
        return res;
    };

    const stats = {steps: 0, jumps: 0, ducks: 0, err: null};

    // Наблюдение — порт readState + inputs_of.
    const decide = function () {
        const dino = r.tRex;
        const speed = r.currentSpeed;
        const height = Math.max(0, dino.groundYPos - dino.yPos);
        const obs = (r.horizon && r.horizon.obstacles) || [];
        let best = null;
        for (let i = 0; i < obs.length; i++) {
            const o = obs[i];
            if (o.xPos + (o.width || 0) <= dino.xPos) continue;
            if (best === null || o.xPos < best.xPos) best = o;
        }
        const ttc = best ? Math.max(0, (best.xPos - dino.xPos) / Math.max(1, speed)) : 60;
        let otype = 0, oy = 0;
        if (best && best.typeConfig) {
            const t = best.typeConfig.type || '';
            if (t === 'PTERODACTYL') otype = 2;
            else if (t.indexOf('CACTUS') === 0) otype = 1;
            oy = (typeof best.yPos === 'number') ? best.yPos : 0;
        }
        const out = activate([
            Math.min(1, Math.max(0, ttc / 60)),
            Math.min(1, Math.max(0, height / 90)),
            Math.min(1, Math.max(0, (dino.jumpVelocity + 11) / 16)),
            Math.min(1, Math.max(0, (speed - 6) / 7)),
            otype === 1 ? 1 : 0,
            (otype === 2 && oy >= 95) ? 1 : 0,
            (otype === 2 && oy >= 65 && oy < 95) ? 1 : 0,
            (otype === 2 && oy < 65) ? 1 : 0,
        ]);
        const onGround = !dino.jumping;
        const wantJump = out[0] > 0 && onGround;
        const wantDuck = out[1] > 0 && onGround && !wantJump;
        if (wantJump) {
            if (dino.ducking) dino.setDuck(false);
            if (!dino.jumping) { dino.startJump(r.currentSpeed); stats.jumps++; }
        } else if (wantDuck) {
            if (!dino.ducking) { dino.setDuck(true); stats.ducks++; }
        } else if (dino.ducking && !dino.jumping) {
            dino.setDuck(false);
        }
    };

    // Синтетические часы с ФИКСИРОВАННЫМ шагом.
    //
    // В реальном браузере deltaTime = getTimeStamp() - this.time, то есть
    // столько, сколько реально прошло между кадрами. Если машина загружена и
    // кадр запаздывает на 50 мс, физика за один вызов прыгает на три кадра
    // вперёд: препятствие проскакивает мимо решения, и динозавр гибнет
    // случайно (наблюдали 7756 шагов вместо 15 000 при том же геноме).
    // Поэтому подменяем часы: каждый кадр двигает время ровно на 1000/60 мс,
    // как в обучающей симуляции. Кадры при этом остаются настоящими (rAF),
    // так что картинка плавная; при просадке fps игра просто чуть замедлится,
    // а не «перескочит» в столкновение.
    let clock = performance.now();
    try {
        Object.defineProperty(performance, 'now',
            {value: function () { return clock; }, configurable: true, writable: true});
    } catch (e) { return {status: "NO_CLOCK"}; }
    const FRAME = 1000 / 60;

    const orig = r.update.bind(r);
    r.update = function () {
        if (!r.crashed && r.playing) {
            try { decide(); stats.steps++; } catch (e) { stats.err = String(e); }
            // r.time ставим на кадр назад: тогда update посчитает
            // deltaTime = clock - (clock - FRAME) = ровно FRAME.
            r.time = clock - FRAME;
            clock += FRAME;
        }
        return orig();
    };
    // Подхватываем цикл. ВАЖНО: scheduleNextUpdate() делает
    // requestAnimationFrame(this.update.bind(this)) — bind захватывает
    // ТЕКУЩИЙ update. Кадр, запланированный до подмены, всё ещё указывает на
    // старую функцию, поэтому его надо отменить и запланировать заново,
    // иначе автопилот не вызывается ни разу (наблюдали «шагов 0» навсегда).
    try { cancelAnimationFrame(r.raqId); } catch (e) {}
    r.raqId = 0;
    r.updatePending = false;
    r.scheduleNextUpdate();

    window.__auto = {stats: stats, stop: function () { r.update = orig; }};
    return {status: "OK"};
}
"""

JS_PLAY_STATUS = """
() => {
    const r = window.Runner && window.Runner.instance_;
    const a = window.__auto;
    if (!r || !a) return {status: "NO_RUNNER"};
    const s = a.stats;
    return {status: r.crashed ? "DEAD" : "OK",
            steps: s.steps, jumps: s.jumps, ducks: s.ducks,
            score: Math.round(r.distanceRan || 0), err: s.err || null};
}
"""

# ── «полная» картинка для показа ───────────────────────────────────────
# В CSS страницы .runner-container имеет width: 44px и overflow: hidden, а до
# полной ширины 600px его доводит только playIntro(). Но playIntro срабатывает
# лишь при activated == false, а наш сброс забега сразу ставит activated = true
# (иначе время в движке стоит). В итоге контейнер навсегда остаётся 44px и
# обрезает канвас — виден только динозавр слева, без земли, кактусов и счёта.
# Плюс поверх висит HTML-надпись «Press Space to start»: это заглушка страницы,
# игру мы запускаем из кода, а не клавишей.
#
# Здесь доводим контейнер до размеров канваса, убираем заглушки и вписываем
# картинку в окно: масштаб = min(ширина/600, высота/150), по центру. Окно при
# этом делается чуть больше самой игры (см. viewport в play_best), поэтому
# пустых полей почти нет.
JS_PLAY_VIEW = """
() => {
    const r = window.Runner && window.Runner.instance_;
    if (!r) return {status: "NO_RUNNER"};

    const W = r.dimensions.WIDTH, H = r.dimensions.HEIGHT;
    const el = r.containerEl;

    // Контейнер должен быть по размеру канваса, иначе overflow:hidden режет.
    el.style.width = W + 'px';
    el.style.height = H + 'px';
    el.style.position = 'absolute';
    el.style.transformOrigin = 'top left';

    // Заглушки страницы: надпись «Press Space to start» и иконка офлайна.
    const mb = document.getElementById('messageBox');
    if (mb) mb.style.display = 'none';
    const icon = document.querySelector('.icon-offline');
    if (icon) icon.style.display = 'none';

    // Пиксельная графика при увеличении остаётся чёткой (как в Chrome).
    r.canvas.style.imageRendering = 'pixelated';

    const fit = function () {
        // Небольшой отступ, чтобы картинка не упиралась в край окна.
        const pad = 8;
        const scale = Math.min((window.innerWidth - pad * 2) / W,
                               (window.innerHeight - pad * 2) / H);
        const s = scale > 0 ? scale : 1;
        el.style.left = Math.round((window.innerWidth - W * s) / 2) + 'px';
        el.style.top = Math.round((window.innerHeight - H * s) / 2) + 'px';
        el.style.transform = 'scale(' + s + ')';
    };
    fit();
    if (!window.__viewResize) {
        window.__viewResize = fit;
        window.addEventListener('resize', window.__viewResize);
    }

    const b = el.getBoundingClientRect();
    return {status: "OK", viewport: {w: window.innerWidth, h: window.innerHeight},
            rect: {x: Math.round(b.x), y: Math.round(b.y),
                   w: Math.round(b.width), h: Math.round(b.height)}};
}
"""

async def play_best():
    """Показать лучшего динозавра в одном окне (реальное время)."""
    if not os.path.exists(BEST_PATH):
        logger.error("best_genome.pkl не найден. Сначала запустите обучение.")
        return
    with open(BEST_PATH, "rb") as f:
        blob = pickle.load(f)
    # Новый формат — словарь с геномом и версией признаков; старый — голый геном.
    genome = blob.get("genome") if isinstance(blob, dict) else blob
    if isinstance(blob, dict) and blob.get("feature_version") != FEATURE_VERSION:
        logger.warning(
            "best_genome.pkl сохранён на другой версии признаков (%s != %d) — "
            "поведение может быть некорректным, переобучите модель.",
            blob.get("feature_version"), FEATURE_VERSION,
        )

    neat_config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
        neat.DefaultStagnation, "neat_config.ini",
    )
    spec = build_net_spec(genome, neat_config)

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        headless=False, executable_path=BROWSER_PATH,
        args=["--disable-gpu", "--no-sandbox"],
    )
    # Окно под размер игры: 600x150 — родное разрешение движка, плюс поля,
    # чтобы картинка не упиралась в края (масштаб ~1.04).
    context = await browser.new_context(viewport={"width": 640, "height": 200})
    page = await context.new_page()
    game = Game(0, browser, page)
    game.context = context
    await game.start()
    await game.start_game()

    inst = await page.evaluate(JS_PLAY_INSTALL, spec)
    if inst.get("status") != "OK":
        logger.error("Не удалось включить автопилот: %s", inst.get("status"))
        await game.close()
        await playwright.stop()
        return
    await page.evaluate(JS_PLAY_VIEW)

    # Python только наблюдает: все решения принимаются в странице.
    # Предела шагов нет — игра идёт, пока динозавр не столкнётся.
    try:
        while True:
            st = await page.evaluate(JS_PLAY_STATUS)
            if st.get("status") == "DEAD":
                print(f"шагов: {st.get('steps', 0)}")
                break
            if st.get("err"):
                logger.error("Ошибка автопилота: %s", st["err"])
                break
            await asyncio.sleep(0.25)
    except Exception:
        # Окно закрыли вручную — это штатный конец просмотра, не ошибка.
        pass

    await game.close()
    await playwright.stop()


def main():
    global TARGET_STEPS
    parser = argparse.ArgumentParser(description="Dino NEAT — обучение в быстрой симуляции")
    parser.add_argument("--play", action="store_true", help="показать лучшего в одном окне")
    parser.add_argument("--generations", type=int, default=MAX_GENERATIONS)
    parser.add_argument("--target-steps", type=int, default=TARGET_STEPS,
                        help="сколько шагов (кадров) должен прожить агент")
    args = parser.parse_args()

    if args.play:
        asyncio.run(play_best())
        return

    TARGET_STEPS = args.target_steps

    trainer = Trainer(args.generations, args.target_steps)

    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()
